"""Latent-space surrogate of molecular dynamics with an exact decoder.

Input: ensembles of MD trajectories of a complex (Cartesian coordinates of every
atom at every saved time step). The model learns, jointly,

* a representation ``z = f(x)`` of every snapshot (an SE(3)-invariant encoder),
* a deterministic decoder ``x_hat = g(z)`` (so every latent element *is* a
  structure and any spatial observable is defined on it by composition),
* a stochastic transition rule ``pi(z' | z)`` at lag ``tau`` -- a latent
  Langevin "policy": the state is the current latent point and the action is the
  distribution of the next latent point, driven by a learned scalar landscape
  ``F(z)`` (the label that makes the latent ensemble interpretable), and
* a label head ``h(z) = E[label | z]`` that attaches per-frame descriptors of the
  MD run (potential energy, native-contact fractions, state indicators, ...) to
  every latent element.

The target is a commuting diagram: encoding then propagating in latent space
must agree with propagating in coordinate space then encoding, and decoding must
undo encoding (up to rigid motions) and encoding must undo decoding on the
latent region the dynamics visits. The pieces are trained together, so the
representation adapts to the rule and the rule to the representation; whitening
removes the scale freedom that would otherwise let the likelihood be gamed.
See README.md ("Latent-space MD model") for the design rationale and limits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn
from torch.distributions import Independent, Normal

# --------------------------------------------------------------------------- #
# Exact geometry: invariants, matrix representations, alignment
# --------------------------------------------------------------------------- #


def center(x: Tensor) -> Tensor:
    """Remove the centroid (translation gauge) of coordinates (..., N, 3)."""
    return x - x.mean(dim=-2, keepdim=True)


def pair_distances(x: Tensor) -> Tensor:
    """Upper-triangular pairwise distances, shape (..., N * (N - 1) / 2).

    Exactly invariant to rotations, reflections and translations; smooth and
    safe to differentiate. This is the encoder input.
    """
    n = x.shape[-2]
    i, j = torch.triu_indices(n, n, offset=1, device=x.device)
    return ((x[..., i, :] - x[..., j, :]).pow(2).sum(-1) + 1e-12).sqrt()


def distance_matrix(x: Tensor) -> Tensor:
    """Euclidean distance matrix (..., N, N) with an exactly zero diagonal."""
    d2 = (x.unsqueeze(-2) - x.unsqueeze(-3)).pow(2).sum(-1)
    eye = torch.eye(x.shape[-2], dtype=torch.bool, device=x.device)
    return torch.where(eye, torch.zeros_like(d2), d2.masked_fill(eye, 1.0).sqrt())


def gram_matrix(x: Tensor) -> Tensor:
    """Gram matrix of centred coordinates; invariant to rotations/translations."""
    xc = center(x)
    return xc @ xc.transpose(-1, -2)


def coords_from_gram(gram: Tensor, dim: int = 3) -> Tensor:
    """Coordinates from a Gram matrix by classical MDS.

    Exact inverse of :func:`gram_matrix` up to an orthogonal transformation: any
    symmetric matrix is mapped to its nearest rank-``dim`` PSD embedding, so the
    result is always a valid structure.
    """
    evals, evecs = torch.linalg.eigh(gram)
    evals = evals[..., -dim:].flip(-1).clamp_min(0.0)
    evecs = evecs[..., -dim:].flip(-1)
    return evecs * evals.sqrt().unsqueeze(-2)


def coords_from_distances(dist: Tensor, dim: int = 3) -> Tensor:
    """Coordinates from a distance matrix (Torgerson/Schoenberg classical MDS).

    Exact for Euclidean-embeddable matrices, up to rotation/reflection (the
    mirror image has the same distances; fix chirality with :func:`kabsch_align`
    against a reference).
    """
    n = dist.shape[-1]
    j = torch.eye(n, dtype=dist.dtype, device=dist.device) - 1.0 / n
    return coords_from_gram(-0.5 * j @ dist.pow(2) @ j, dim)


def kabsch_rotation(mobile: Tensor, target: Tensor) -> Tensor:
    """Proper rotation R (..., 3, 3) minimising ``|center(mobile) R^T - center(target)|``.

    Computed without gradients. Mirror images are *not* superimposed
    (determinant +1), so chirality is distinguished.
    """
    with torch.no_grad():
        u, _, vh = torch.linalg.svd(center(mobile).transpose(-1, -2) @ center(target))
        v = vh.transpose(-1, -2)
        sign = torch.sign(torch.linalg.det(v @ u.transpose(-1, -2)))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        scale = torch.stack((torch.ones_like(sign), torch.ones_like(sign), sign), dim=-1)
        return v @ torch.diag_embed(scale) @ u.transpose(-1, -2)


def kabsch_align(mobile: Tensor, target: Tensor) -> Tensor:
    """Superimpose ``mobile`` on ``target`` (optimal proper rotation + translation).

    Gradients flow through ``mobile`` with the rotation held fixed; because the
    rotation minimises the deviation, this is the exact gradient of the
    minimised deviation (envelope theorem) and avoids differentiating the SVD.
    """
    rot = kabsch_rotation(mobile, target)
    return center(mobile) @ rot.transpose(-1, -2) + target.mean(dim=-2, keepdim=True)


def rmsd(a: Tensor, b: Tensor) -> Tensor:
    """Root-mean-square deviation after optimal superposition, shape (...,)."""
    return (kabsch_align(a, b) - b).pow(2).sum(-1).mean(-1).sqrt()


def contact_map(x: Tensor, cutoff: float) -> Tensor:
    """Boolean contact map (..., N, N): distance below ``cutoff``, no self-contacts."""
    eye = torch.eye(x.shape[-2], dtype=torch.bool, device=x.device)
    return (distance_matrix(x) < cutoff) & ~eye


@torch.no_grad()
def procrustes_mean(coords: Tensor, n_iter: int = 5) -> Tensor:
    """Mean structure of (M, N, 3) coordinates after iterative superposition."""
    mean = center(coords[0])
    for _ in range(n_iter):
        mean = center(kabsch_align(coords, mean.expand_as(coords)).mean(0))
    return mean


# --------------------------------------------------------------------------- #
# Encoder / decoder / transition policy
# --------------------------------------------------------------------------- #


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class StructureEncoder(nn.Module):
    """SE(3)-invariant map from coordinates (..., N, 3) to latent points (..., d).

    Standardised pairwise distances feed an MLP, so snapshots expressed in
    different lab frames (different ensembles, unaligned trajectories) land on
    the same latent point. The distance matrix is a lossless, invariant matrix
    representation of the structure up to reflection; the bottleneck is the
    only lossy step.
    """

    def __init__(self, n_atoms: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        n_pairs = n_atoms * (n_atoms - 1) // 2
        self.register_buffer("feature_mean", torch.zeros(n_pairs))
        self.register_buffer("feature_scale", torch.ones(n_pairs))
        self.net = _mlp(n_pairs, hidden_dim, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.net((pair_distances(x) - self.feature_mean) / self.feature_scale)


class StructureDecoder(nn.Module):
    """Deterministic map from latent points (..., d) to centred coordinates.

    ``x_hat = reference + length_scale * MLP(z)``. It is a plain function on the
    whole latent space, so every latent element is a structure; coordinates are
    expressed near the reference frame (the gauge is only fixed up to rigid
    motion by the loss). Fluctuations that the latent does not encode are
    averaged out: ``g(z)`` approximates the mean structure given ``z``.
    """

    def __init__(self, n_atoms: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.n_atoms = n_atoms
        self.register_buffer("reference", torch.zeros(n_atoms, 3))
        self.register_buffer("length_scale", torch.ones(()))
        self.net = _mlp(latent_dim, hidden_dim, 3 * n_atoms)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: Tensor) -> Tensor:
        delta = self.net(z).reshape(*z.shape[:-1], self.n_atoms, 3)
        return center(self.reference + self.length_scale * delta)


class LatentLangevinPolicy(nn.Module):
    """Stochastic transition rule on the latent space (the "agent").

    State: current latent point ``z``. Action: the distribution of the next
    latent point after one lag, a diagonal Gaussian

        pi(z' | z) = N(z + b(z), 2 M(z)),   b_i = -M_i dF/dz_i + dM_i/dz_i,

    i.e. the Euler-Maruyama kernel of overdamped Langevin dynamics with a learned
    landscape ``F(z) = confinement/2 |z|^2 + MLP(z)`` (units of k_B T) and a
    learned, *state-dependent* positive diagonal mobility ``M(z)``. The Ito
    correction ``dM_i/dz_i`` keeps the stationary density of the continuous-time
    dynamics proportional to ``exp(-F)``: the landscape that drives the dynamics
    is the same object that labels the latent elements. The mobility has to
    depend on the state because stiff metastable basins and soft broad basins
    need very different step sizes. The quadratic term makes the stationary
    density normalisable and keeps rollouts in the region where the decoder is
    trusted. An optional ``bias`` potential (a callable ``z -> (batch,)``) is
    added to ``F`` to steer the dynamics.

    ``min_mobility`` is the resolution of the description: the step noise cannot
    drop below ``sqrt(2 min_mobility)`` (in whitened latent units), so the
    likelihood cannot be improved by contracting latent regions, which would
    otherwise discard exactly the information the decoder needs. ``max_step``
    softly caps the drift per lag (``tanh``), a numerical safeguard that is
    inactive on the data but stops explicit-Euler blow-ups where the learned
    landscape is an extrapolation.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        initial_mobility: float = 0.05,
        confinement: float = 1.0,
        min_mobility: float = 1e-3,
        max_mobility: float = 2.0,
        max_step: float = 3.0,
    ) -> None:
        super().__init__()
        lo, hi = math.log(min_mobility), math.log(max_mobility)
        if not 0.0 < (math.log(initial_mobility) - lo) / (hi - lo) < 1.0:
            raise ValueError("initial_mobility must lie strictly between min_mobility and max_mobility")
        self.latent_dim = latent_dim
        self.max_step = max_step
        self.log_mobility_range = (lo, hi)
        self.energy_net = _mlp(latent_dim, hidden_dim, 1)
        self.mobility_net = _mlp(latent_dim, hidden_dim, latent_dim)
        fraction = (math.log(initial_mobility) - lo) / (hi - lo)
        nn.init.zeros_(self.mobility_net[-1].weight)
        nn.init.constant_(self.mobility_net[-1].bias, math.log(fraction / (1.0 - fraction)))
        self.confinement = confinement

    def free_energy(self, z: Tensor) -> Tensor:
        """Landscape F(z) in k_B T (up to an additive constant), shape (...,)."""
        return 0.5 * self.confinement * z.pow(2).sum(-1) + self.energy_net(z).squeeze(-1)

    def log_mobility(self, z: Tensor) -> Tensor:
        """ln M(z), smoothly confined to ``log_mobility_range``; shape (..., d)."""
        lo, hi = self.log_mobility_range
        return lo + (hi - lo) * torch.sigmoid(self.mobility_net(z))

    def mobility(self, z: Tensor) -> Tensor:
        return self.log_mobility(z).exp()

    def drift(
        self,
        z: Tensor,
        bias: Callable[[Tensor], Tensor] | None = None,
        create_graph: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Mean displacement ``b(z)`` and mobility ``M(z)``; usable under ``torch.no_grad()``."""
        with torch.enable_grad():
            if not z.requires_grad:
                z = z.detach().requires_grad_(True)
            energy = self.free_energy(z)
            if bias is not None:
                energy = energy + bias(z)
            log_m = self.log_mobility(z)
            (grad_f,) = torch.autograd.grad(
                energy.sum(), z, create_graph=create_graph, retain_graph=True
            )
            dlog_m = torch.stack(
                [
                    torch.autograd.grad(
                        log_m[..., i].sum(), z, create_graph=create_graph, retain_graph=True
                    )[0][..., i]
                    for i in range(self.latent_dim)
                ],
                dim=-1,
            )
        m = log_m.exp()
        step = m * (dlog_m - grad_f)
        return self.max_step * torch.tanh(step / self.max_step), m

    def action(
        self,
        z: Tensor,
        bias: Callable[[Tensor], Tensor] | None = None,
        create_graph: bool = False,
    ) -> Independent:
        """Distribution of the next latent point given ``z`` (the policy output)."""
        drift, m = self.drift(z, bias, create_graph)
        return Independent(Normal(z + drift, (2.0 * m).sqrt()), 1)

    def log_prob(self, z: Tensor, z_next: Tensor, create_graph: bool = False) -> Tensor:
        return self.action(z, create_graph=create_graph).log_prob(z_next)

    def sample(
        self,
        z: Tensor,
        bias: Callable[[Tensor], Tensor] | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        dist = self.action(z, bias)
        noise = torch.randn(dist.mean.shape, generator=generator, dtype=z.dtype, device=z.device)
        return (dist.mean + dist.base_dist.scale * noise).detach()


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #


class LatentMD(nn.Module):
    """Encoder, exact decoder, latent transition policy and label head.

    Args:
        n_atoms: Number of atoms of the (fixed-topology) complex.
        latent_dim: Dimension d of the latent space. Reconstruction improves with
            d; the latent dynamics is easiest to learn for small d (see README).
        hidden_dim: Width of the encoder/decoder MLPs.
        policy_hidden_dim: Width of the landscape, mobility and label MLPs.
        initial_mobility: Initial diagonal mobility of the latent Langevin kernel.
        confinement: Strength of the quadratic term in the landscape.
        n_labels: Number of per-frame label columns regressed on the latent point
            (0 disables the label head).
        min_mobility: Floor of the latent mobility (resolution of the description).

    Call :meth:`initialize_statistics` (done by :func:`fit`) before use so that
    feature scaling, the decoder reference and the label scales are set from data.
    """

    def __init__(
        self,
        n_atoms: int,
        latent_dim: int = 4,
        hidden_dim: int = 128,
        policy_hidden_dim: int = 64,
        initial_mobility: float = 0.05,
        confinement: float = 1.0,
        n_labels: int = 0,
        min_mobility: float = 1e-3,
    ) -> None:
        super().__init__()
        self.n_atoms = n_atoms
        self.latent_dim = latent_dim
        self.encoder = StructureEncoder(n_atoms, latent_dim, hidden_dim)
        self.decoder = StructureDecoder(n_atoms, latent_dim, hidden_dim)
        self.policy = LatentLangevinPolicy(
            latent_dim, policy_hidden_dim, initial_mobility, confinement, min_mobility
        )
        self.n_labels = n_labels
        self.label_head = _mlp(latent_dim, policy_hidden_dim, n_labels) if n_labels else None
        self.register_buffer("label_mean", torch.zeros(n_labels))
        self.register_buffer("label_scale", torch.ones(n_labels))
        self.register_buffer("free_energy_offset", torch.zeros(()))
        self.register_buffer("reference_msd", torch.ones(()))

    # -- statistics ---------------------------------------------------------- #

    @torch.no_grad()
    def initialize_statistics(
        self, coords: Tensor, labels: Tensor | None = None, max_frames: int = 20000
    ) -> None:
        """Set feature scaling, decoder reference and label scales from (M, N, 3) data.

        Structure statistics use an evenly strided subset of at most ``max_frames``
        frames and chunked accumulation, so memory stays bounded for long
        trajectories of large complexes.
        """
        subset = coords[:: -(-coords.shape[0] // max_frames)]
        total = squares = 0.0
        for chunk in subset.split(1024):
            features = pair_distances(chunk).double()
            total = total + features.sum(0)
            squares = squares + features.pow(2).sum(0)
        feature_mean = total / subset.shape[0]
        feature_std = (squares / subset.shape[0] - feature_mean.pow(2)).clamp_min(0.0).sqrt()
        self.encoder.feature_mean.copy_(feature_mean)
        self.encoder.feature_scale.copy_(feature_std.clamp_min(1e-3 * float(feature_mean.mean())))
        mean = procrustes_mean(subset)
        deviation = kabsch_align(subset, mean.expand_as(subset)) - mean
        self.reference_msd.copy_(deviation.pow(2).sum(-1).mean())
        self.decoder.reference.copy_(mean)
        self.decoder.length_scale.copy_(self.reference_msd.sqrt())
        if labels is not None:
            self.label_mean.copy_(labels.mean(0))
            self.label_scale.copy_(labels.std(0).clamp_min(1e-8))

    # -- maps ---------------------------------------------------------------- #

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    @torch.no_grad()
    def encode_frames(self, x: Tensor, chunk_size: int = 2048) -> Tensor:
        """:meth:`encode` for long trajectories (M, N, 3) in chunks, without gradients."""
        return torch.cat([self.encoder(c) for c in x.split(chunk_size)])

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Round trip g(f(x)): returns the latent points and reconstructed structures."""
        z = self.encode(x)
        return z, self.decode(z)

    def pullback(self, z: Tensor, observable: Callable[[Tensor], Tensor]) -> Tensor:
        """Define a spatial observable on latent elements: ``O(z) = O(g(z))``.

        Contact maps, distances, RMSD to a reference, force-field energies and
        any other function of coordinates are available on the latent space
        through this one rule, so latent ensembles can be analysed exactly like
        MD ensembles.
        """
        return observable(self.decode(z))

    # -- labels -------------------------------------------------------------- #

    def labels(self, z: Tensor) -> Tensor:
        """Learned labels ``E[label | z]`` of latent elements, shape (..., n_labels).

        Unlike a pullback ``O(g(z))`` (an observable of the *mean* structure given
        ``z``), this is the conditional mean of the observable over the structures
        that map to ``z``; the two differ by the fluctuations the latent averages out.
        """
        if self.label_head is None:
            raise RuntimeError("the model was built with n_labels=0")
        return self.label_head(z) * self.label_scale + self.label_mean

    def free_energy(self, z: Tensor) -> Tensor:
        """Landscape F(z) in k_B T that drives the latent dynamics (min ~ 0 on data)."""
        return self.policy.free_energy(z) - self.free_energy_offset

    # -- the transition rule ------------------------------------------------- #

    def action(self, z: Tensor, bias: Callable[[Tensor], Tensor] | None = None) -> Independent:
        """Policy: distribution of the next latent point after one lag."""
        return self.policy.action(z, bias)

    @torch.no_grad()
    def rollout(
        self,
        z0: Tensor,
        n_steps: int,
        bias: Callable[[Tensor], Tensor] | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Evolve latent points directly with the policy.

        Returns (n_steps + 1, batch, d); every step advances one lag of the
        original MD. With ``bias`` the dynamics is steered by ``F + bias``.
        """
        path = [z0]
        for _ in range(n_steps):
            path.append(self.policy.sample(path[-1], bias, generator))
        return torch.stack(path)

    # -- joint objective ----------------------------------------------------- #

    def _relative_msd(self, x_hat: Tensor, x: Tensor) -> Tensor:
        """Mean squared deviation after superposition, relative to the data variance."""
        aligned = kabsch_align(x_hat, x)
        return (aligned - x).pow(2).sum(-1).mean(-1) / self.reference_msd

    def loss_terms(
        self,
        x_now: Tensor,
        x_next: Tensor,
        labels_now: Tensor | None = None,
        cycle_steps: int = 4,
        tube_scale: float = 1.0,
    ) -> dict[str, Tensor]:
        """Terms of the joint objective for a batch of lag pairs (x_t, x_{t+lag}).

        reconstruction: g(f(x)) ~ x up to rigid motion (fraction of variance unexplained).
        dynamics:       negative log-likelihood per latent dimension of f(x_{t+lag})
                        under pi(. | f(x_t)); its gradient reaches the encoder, so the
                        representation is shaped by the rule it must obey.
        whitening:      zero-mean, identity-covariance latents; removes the scale and
                        shear freedom that would let the likelihood be improved by
                        merely shrinking the latent space.
        cycle:          f(g(z)) ~ z on points visited by the latent dynamics, the
                        other half of a well-defined latent <-> structure correspondence.
        tube:           g(z + noise) ~ x for noise at the scale of the policy's own
                        one-step noise, so that latent states the dynamics reaches
                        slightly off the encoded data still decode to valid structures
                        (the resolution of the decoder is tied to the rule's noise).
        labels:         regression of the per-frame labels (energy, indicators, ...)
                        on the latent point, i.e. E[label | z].
        """
        z0, z1 = self.encode(x_now), self.encode(x_next)
        reconstruction = 0.5 * (
            self._relative_msd(self.decode(z0), x_now) + self._relative_msd(self.decode(z1), x_next)
        ).mean()
        dynamics = -self.policy.log_prob(z0, z1, create_graph=True).mean() / self.latent_dim

        z = torch.cat((z0, z1))
        zc = z - z.mean(0)
        cov = zc.T @ zc / (z.shape[0] - 1)
        whitening = (cov - torch.eye(self.latent_dim, device=z.device)).pow(2).mean()
        whitening = whitening + z.mean(0).pow(2).mean()

        z_walk = self.rollout(z0.detach(), cycle_steps)[-1]
        cycle = (self.encode(self.decode(z_walk)) - z_walk).pow(2).mean()

        with torch.no_grad():
            sigma = tube_scale * (2.0 * self.policy.mobility(z0)).sqrt()
        tube = self._relative_msd(self.decode(z0 + sigma * torch.randn_like(z0)), x_now).mean()

        terms = {
            "reconstruction": reconstruction,
            "dynamics": dynamics,
            "whitening": whitening,
            "cycle": cycle,
            "tube": tube,
        }
        if labels_now is not None:
            target = (labels_now - self.label_mean) / self.label_scale
            terms["labels"] = (self.label_head(z0) - target).pow(2).mean()
        return terms


# --------------------------------------------------------------------------- #
# Data and training
# --------------------------------------------------------------------------- #


class TrajectoryEnsembles:
    """Pools trajectories (T_i, N, 3) from any number of ensembles into lag pairs.

    Frames of different trajectories/ensembles may be in different lab frames.
    Pairs (x_t, x_{t+lag}) never cross a trajectory boundary. ``labels`` holds one
    (frames,) or (frames, n_labels) array of per-frame descriptors per trajectory.
    """

    def __init__(
        self,
        trajectories: list[Tensor],
        lag: int,
        labels: list[Tensor] | None = None,
    ) -> None:
        if lag < 1:
            raise ValueError("lag must be a positive number of frames")
        if labels is not None and len(labels) != len(trajectories):
            raise ValueError("labels must match trajectories")
        frames, series, src, dst, offset = [], [], [], [], 0
        for k, traj in enumerate(trajectories):
            traj = torch.as_tensor(traj, dtype=torch.float32)
            if traj.ndim != 3 or traj.shape[-1] != 3:
                raise ValueError("each trajectory must have shape (frames, atoms, 3)")
            frames.append(traj)
            if labels is not None:
                column = torch.as_tensor(labels[k], dtype=torch.float32)
                column = column.reshape(traj.shape[0], -1)
                series.append(column)
            if traj.shape[0] > lag:
                start = torch.arange(traj.shape[0] - lag)
                src.append(offset + start)
                dst.append(offset + start + lag)
            offset += traj.shape[0]
        if not src:
            raise ValueError("no trajectory is longer than the lag")
        self.lag = lag
        self.coords = torch.cat(frames)
        self.labels = torch.cat(series) if labels is not None else None
        self.src, self.dst = torch.cat(src), torch.cat(dst)

    def __len__(self) -> int:
        return self.src.numel()

    def pairs(self, index: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor | None]:
        """Lag pairs (x_t, x_{t+lag}, labels_t); all pairs when ``index`` is None."""
        src = self.src if index is None else self.src[index]
        dst = self.dst if index is None else self.dst[index]
        labels = None if self.labels is None else self.labels[src]
        return self.coords[src], self.coords[dst], labels

    def sample(
        self, batch_size: int, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        index = torch.randint(len(self), (batch_size,), generator=generator)
        return self.pairs(index)


@dataclass
class TrainConfig:
    """Hyperparameters of :func:`fit`.

    The dynamics and cycle terms are ramped in over ``warmup_steps`` so the
    representation first becomes informative (reconstruction + whitening) and is
    then adapted to the rule.
    """

    steps: int = 4000
    batch_size: int = 512
    learning_rate: float = 2e-3
    final_lr_fraction: float = 0.05
    warmup_steps: int = 1000
    cycle_steps: int = 4
    grad_clip: float = 5.0
    weight_reconstruction: float = 10.0
    weight_dynamics: float = 1.0
    weight_whitening: float = 10.0
    weight_cycle: float = 1.0
    weight_labels: float = 1.0
    weight_tube: float = 10.0
    tube_scale: float = 1.0
    seed: int = 0
    log_every: int = 500


def fit(
    model: LatentMD,
    data: TrajectoryEnsembles,
    config: TrainConfig | None = None,
    log: Callable[[str], None] | None = None,
) -> list[dict[str, float]]:
    """Train encoder, decoder, policy and label head jointly on lag pairs.

    Returns the logged history (one dict of loss terms per ``log_every`` steps).
    """
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    model.initialize_statistics(data.coords, data.labels)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    decay = config.final_lr_fraction
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: decay + (1 - decay) * 0.5 * (1 + math.cos(math.pi * s / config.steps)),
    )
    history: list[dict[str, float]] = []
    model.train()
    for step in range(1, config.steps + 1):
        ramp = min(1.0, step / max(1, config.warmup_steps))
        x_now, x_next, labels = data.sample(config.batch_size, generator)
        terms = model.loss_terms(x_now, x_next, labels, config.cycle_steps, config.tube_scale)
        loss = (
            config.weight_reconstruction * terms["reconstruction"]
            + ramp * config.weight_dynamics * terms["dynamics"]
            + config.weight_whitening * terms["whitening"]
            + ramp * config.weight_cycle * terms["cycle"]
            + ramp * config.weight_tube * terms["tube"]
        )
        if "labels" in terms:
            loss = loss + config.weight_labels * terms["labels"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        schedule.step()
        if step % config.log_every == 0 or step == config.steps:
            record = {k: float(v.detach()) for k, v in terms.items()}
            record["step"] = step
            history.append(record)
            if log is not None:
                log("step %5d  " % step + "  ".join(f"{k}={v:.4f}" for k, v in record.items() if k != "step"))
    model.eval()
    sample = model.encode_frames(data.coords[:: max(1, len(data.coords) // 20000)])
    with torch.no_grad():
        model.free_energy_offset.copy_(model.policy.free_energy(sample).min())
    return history
