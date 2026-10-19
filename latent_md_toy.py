"""Synthetic MD ensembles used to exercise and validate ``latent_md.py``.

The system is a small two-domain "complex" (8 atoms, chiral) with three
metastable contact conformations (``open``, ``closed_a``, ``closed_b``). Each
conformation is an elastic-network basin; the basins are mixed with a smooth
minimum (multiple-basin Go-type model), so the barriers are finite and the
potential energy, forces and contact patterns are known analytically. Dynamics
are underdamped Langevin (BAOAB) in full Cartesian space; frames are stored at a
coarse stride and can be expressed in arbitrary lab frames, as raw MD output.

It is a stand-in for real MD only: it provides ground truth (energies, contact
labels, the exact spatial dynamics) against which latent-space evolution can be
assessed. No real simulation data are used or included.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor

STATE_NAMES = ("open", "closed_a", "closed_b")
_DTYPE = torch.float64


def rotation_z(angle: float) -> Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=_DTYPE)


def random_rotations(
    n: int, generator: torch.Generator | None = None, dtype: torch.dtype = _DTYPE
) -> Tensor:
    """Haar-random proper rotations, shape (n, 3, 3)."""
    q, r = torch.linalg.qr(torch.randn(n, 3, 3, generator=generator, dtype=dtype))
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
    det = torch.linalg.det(q)
    q[:, :, 0] = q[:, :, 0] * det.unsqueeze(-1)
    return q


def apply_lab_frame(
    traj: Tensor, mode: str, generator: torch.Generator | None = None
) -> Tensor:
    """Express a (T, N, 3) trajectory in a different lab frame.

    ``"raw"`` keeps the simulated frame, ``"fixed"`` applies one random rigid
    motion to the whole trajectory and ``"random"`` an independent random rigid
    motion to every frame (completely unaligned output).
    """
    if mode == "raw":
        return traj
    n = traj.shape[0] if mode == "random" else 1
    if mode not in ("fixed", "random"):
        raise ValueError(f"unknown lab-frame mode {mode!r}")
    rot = random_rotations(n, generator, traj.dtype)
    shift = 5.0 * torch.randn(n, 1, 3, generator=generator, dtype=traj.dtype)
    return traj @ rot.transpose(-1, -2) + shift


class TwoDomainComplex:
    """Three-basin two-domain complex with analytic energy and forces.

    Atoms 0-3 form domain A and atoms 4-7 domain B (an apex plus an equilateral
    base triangle each; atoms 1-3 / 5-7 face each other across the interface).
    ``closed_a`` pairs atom ``1+i`` with ``5+i``; ``closed_b`` is B rotated by
    120 degrees about the interface normal, which gives a different set of
    native contacts. ``open`` has B lifted off with no native contacts. The
    parameter set below was chosen (by simulation) so that the three
    conformations have comparable populations and relaxation times of about
    one hundred frames at the default stride. Energies are in units of k_B T.
    """

    n_atoms = 8
    n_domain = 4
    interface_a = (1, 2, 3)
    interface_b = (5, 6, 7)

    def __init__(
        self,
        k_intra: float = 100.0,
        k_inter: Sequence[float] = (2.5, 4.0, 4.0),
        state_offsets: Sequence[float] = (0.0, -5.4, -4.6),
        mixing_beta: float = 0.5,
        contact_cutoff: float = 1.6,
        side: float = 2.4,
        height_closed: float = 1.2,
        height_open: float = 2.4,
    ) -> None:
        self.k_intra = k_intra
        self.k_inter = torch.tensor(list(k_inter), dtype=_DTYPE)
        self.state_offsets = torch.tensor(list(state_offsets), dtype=_DTYPE)
        self.mixing_beta = mixing_beta
        self.contact_cutoff = contact_cutoff

        rho = side / math.sqrt(3.0)
        angles = torch.tensor([90.0, 210.0, 330.0], dtype=_DTYPE) * math.pi / 180.0
        base = torch.stack(
            (rho * torch.cos(angles), rho * torch.sin(angles), torch.zeros(3, dtype=_DTYPE)),
            dim=-1,
        )
        apex_a = torch.tensor([[0.30, 0.15, -1.00]], dtype=_DTYPE)
        apex_b = torch.tensor([[-0.25, 0.30, 1.00]], dtype=_DTYPE)
        self.domain_a = torch.cat((apex_a, base))
        self.domain_b = torch.cat((apex_b, base))

        poses = (
            self._place(rotation_z(0.7), height_open),
            self._place(rotation_z(0.0), height_closed),
            self._place(rotation_z(2.0 * math.pi / 3.0), height_closed),
        )
        self.references = torch.stack(poses)  # (3, 8, 3)

        self.intra_i, self.intra_j = self._intra_pairs()
        self.intra_rest = self._distances(self.references[1], self.intra_i, self.intra_j)
        ia = torch.arange(self.n_domain)
        ib = torch.arange(self.n_domain) + self.n_domain
        self.inter_rest = torch.stack(
            [torch.cdist(r[ia], r[ib]) for r in self.references]
        )  # (3, 4, 4)

        # Native contacts of each closed pose (indices into the 4x4 interface block).
        self.native = [
            (self.inter_rest[s] < contact_cutoff).nonzero(as_tuple=False) for s in (1, 2)
        ]

    def _place(self, rotation: Tensor, height: float) -> Tensor:
        b = self.domain_b @ rotation.T + torch.tensor([0.0, 0.0, height], dtype=_DTYPE)
        coords = torch.cat((self.domain_a, b))
        return coords - coords.mean(0)

    def _intra_pairs(self) -> tuple[Tensor, Tensor]:
        i, j = torch.triu_indices(self.n_domain, self.n_domain, offset=1)
        return torch.cat((i, i + self.n_domain)), torch.cat((j, j + self.n_domain))

    @staticmethod
    def _distances(x: Tensor, i: Tensor, j: Tensor) -> Tensor:
        return (x[..., i, :] - x[..., j, :]).norm(dim=-1)

    def reference(self, state: str | int) -> Tensor:
        index = STATE_NAMES.index(state) if isinstance(state, str) else state
        return self.references[index].clone()

    def _inter_terms(self, x: Tensor):
        a, b = x[..., : self.n_domain, :], x[..., self.n_domain :, :]
        diff = a.unsqueeze(-2) - b.unsqueeze(-3)  # (..., 4, 4, 3)
        dist = diff.norm(dim=-1).clamp_min(1e-9)
        dev = dist.unsqueeze(-3) - self.inter_rest  # (..., S, 4, 4)
        k = self.k_inter.view(-1, 1, 1)
        basin = 0.5 * (k * dev.pow(2)).sum((-1, -2)) + self.state_offsets
        logits = -self.mixing_beta * basin
        return diff, dist, dev, k, logits

    def energy(self, x: Tensor) -> Tensor:
        """Potential energy (k_B T) of coordinates ``x`` with shape (..., 8, 3)."""
        x = x.to(_DTYPE)
        dev = self._distances(x, self.intra_i, self.intra_j) - self.intra_rest
        intra = 0.5 * self.k_intra * dev.pow(2).sum(-1)
        logits = self._inter_terms(x)[-1]
        return intra - torch.logsumexp(logits, dim=-1) / self.mixing_beta

    def force(self, x: Tensor) -> Tensor:
        """Analytic force -dU/dx, shape (..., 8, 3)."""
        x = x.to(_DTYPE)
        force = torch.zeros_like(x)
        vec = x[..., self.intra_i, :] - x[..., self.intra_j, :]
        dist = vec.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        dev = dist - self.intra_rest.unsqueeze(-1)
        pair = -self.k_intra * dev * vec / dist
        force = force.index_add(-2, self.intra_i, pair).index_add(-2, self.intra_j, -pair)

        diff, dist, dev, k, logits = self._inter_terms(x)
        weights = torch.softmax(logits, dim=-1)  # (..., S)
        coeff = (weights.unsqueeze(-1).unsqueeze(-1) * k * dev).sum(-3)  # (..., 4, 4)
        pair = -coeff.unsqueeze(-1) * diff / dist.unsqueeze(-1)
        force_a = pair.sum(-2)
        force_b = -pair.sum(-3)
        return force + torch.cat((force_a, force_b), dim=-2)

    def contact_fractions(self, x: Tensor, steepness: float = 10.0, slack: float = 1.25) -> Tensor:
        """Soft fraction of native contacts of both closed poses, shape (..., 2).

        Uses the smooth contact function 1 / (1 + exp(steepness * (r - slack * r0))).
        """
        x = x.to(_DTYPE)
        a, b = x[..., : self.n_domain, :], x[..., self.n_domain :, :]
        dist = (a.unsqueeze(-2) - b.unsqueeze(-3)).norm(dim=-1)  # (..., 4, 4)
        out = []
        for pose, pairs in zip((1, 2), self.native):
            r = dist[..., pairs[:, 0], pairs[:, 1]]
            r0 = self.inter_rest[pose][pairs[:, 0], pairs[:, 1]]
            out.append(torch.sigmoid(-steepness * (r - slack * r0)).mean(-1))
        return torch.stack(out, dim=-1)

    def state_labels(self, x: Tensor, threshold: float = 0.5) -> Tensor:
        """Contact-conformation label: 0 open, 1 closed_a, 2 closed_b."""
        q = self.contact_fractions(x)
        best = q.argmax(-1) + 1
        return torch.where(q.max(-1).values > threshold, best, torch.zeros_like(best))

    @torch.no_grad()
    def simulate(
        self,
        start: str | int,
        n_replicas: int,
        n_frames: int,
        stride: int = 10,
        dt: float = 0.01,
        friction: float = 4.0,
        burn_in: int = 200,
        seed: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Underdamped Langevin (BAOAB, unit mass, k_B T = 1) in full space.

        Returns frames of shape (n_replicas, n_frames, 8, 3) and the potential
        energies (n_replicas, n_frames). Replicas start from the chosen state.
        """
        gen = torch.Generator().manual_seed(seed)
        x = self.reference(start).expand(n_replicas, -1, -1).clone()
        v = torch.randn(x.shape, generator=gen, dtype=_DTYPE)
        c1 = math.exp(-friction * dt)
        c2 = math.sqrt(1.0 - c1 * c1)
        f = self.force(x)
        frames = torch.empty(n_replicas, n_frames, self.n_atoms, 3, dtype=_DTYPE)
        energy = torch.empty(n_replicas, n_frames, dtype=_DTYPE)
        for step in range(burn_in + n_frames * stride):
            v = v + 0.5 * dt * f
            x = x + 0.5 * dt * v
            v = c1 * v + c2 * torch.randn(x.shape, generator=gen, dtype=_DTYPE)
            x = x + 0.5 * dt * v
            f = self.force(x)
            v = v + 0.5 * dt * f
            k, rem = divmod(step - burn_in + 1, stride)
            if step >= burn_in and rem == 0:
                frames[:, k - 1] = x
                energy[:, k - 1] = self.energy(x)
        return frames, energy


def generate_ensembles(
    system: TwoDomainComplex,
    specs: Sequence[tuple[str, str]],
    n_replicas: int = 12,
    n_frames: int = 3000,
    seed: int = 0,
    **simulate_kwargs,
) -> tuple[list[Tensor], list[Tensor], list[int]]:
    """Simulate one ensemble per ``(start_state, lab_frame_mode)`` spec.

    Returns one float32 trajectory (n_frames, 8, 3) and one energy series per
    replica, plus the ensemble index of each replica. Ensembles differ in their
    starting conformation, random seed and lab-frame convention, mimicking MD
    output that was not aligned to a common reference.
    """
    trajectories: list[Tensor] = []
    energies: list[Tensor] = []
    ensemble_ids: list[int] = []
    gen = torch.Generator().manual_seed(seed + 12345)
    for e, (start, frame_mode) in enumerate(specs):
        frames, energy = system.simulate(
            start, n_replicas, n_frames, seed=seed + 1000 * (e + 1), **simulate_kwargs
        )
        for r in range(n_replicas):
            trajectories.append(apply_lab_frame(frames[r], frame_mode, gen).float())
            energies.append(energy[r].float())
            ensemble_ids.append(e)
    return trajectories, energies, ensemble_ids
