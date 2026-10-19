"""Fast tests of the latent-space MD model (``pytest -q``; about a minute on a laptop CPU).

They check the properties the design relies on: exact geometry helpers, rigid-motion
invariance of the encoder, a deterministic total decoder, a transition policy that is a
proper Langevin kernel, the toy force field, lag pairs that respect trajectory
boundaries, and that the likelihood of the transition rule shapes the representation.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

import latent_md_real
from latent_md import (
    LatentLangevinPolicy,
    LatentMD,
    TrainConfig,
    TrajectoryEnsembles,
    center,
    coords_from_distances,
    coords_from_gram,
    distance_matrix,
    fit,
    gram_matrix,
    kabsch_align,
    pair_distances,
    rmsd,
)
from latent_md_toy import TwoDomainComplex, generate_ensembles, random_rotations

torch.manual_seed(0)


@pytest.fixture(scope="module")
def system() -> TwoDomainComplex:
    return TwoDomainComplex()


@pytest.fixture(scope="module")
def toy_data(system):
    """Two small ensembles with different lab-frame conventions."""
    trajectories, energies, _ = generate_ensembles(
        system, [("open", "fixed"), ("closed_b", "random")], n_replicas=2, n_frames=250, seed=1
    )
    return trajectories, energies


def rigid_motion(x: torch.Tensor, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    rot = random_rotations(1, gen, x.dtype)[0]
    return x @ rot.T + 3.0 * torch.randn(3, generator=gen, dtype=x.dtype)


def small_model(trajectories, latent_dim: int = 2, n_labels: int = 0) -> LatentMD:
    model = LatentMD(trajectories[0].shape[1], latent_dim=latent_dim, hidden_dim=32, policy_hidden_dim=16, n_labels=n_labels)
    model.initialize_statistics(torch.cat(trajectories))
    return model


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def test_classical_mds_inverts_gram_and_distance_matrices_exactly():
    x = torch.randn(5, 9, 3, dtype=torch.float64)
    gram = gram_matrix(x)
    assert torch.allclose(gram_matrix(coords_from_gram(gram)), gram, atol=1e-10)
    dist = distance_matrix(x)
    assert torch.allclose(distance_matrix(coords_from_distances(dist)), dist, atol=1e-8)
    # any symmetric matrix is mapped to a valid structure (nearest PSD rank-3 embedding)
    noisy = gram + 0.3 * torch.randn(5, 9, 9, dtype=torch.float64)
    structure = coords_from_gram(0.5 * (noisy + noisy.transpose(-1, -2)))
    assert structure.shape == (5, 9, 3) and torch.isfinite(structure).all()


def test_kabsch_recovers_rigid_motions_and_distinguishes_mirror_images():
    x = torch.randn(7, 3, dtype=torch.float64)
    y = rigid_motion(x)
    assert rmsd(y, x) < 1e-10
    assert torch.allclose(kabsch_align(y, x), x, atol=1e-10)
    mirror = x * torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
    assert torch.allclose(pair_distances(mirror), pair_distances(x))  # distances cannot see chirality
    assert rmsd(mirror, x) > 0.1  # the proper-rotation fit does


def test_pair_distances_are_differentiable_at_coincident_atoms():
    x = torch.zeros(1, 4, 3, requires_grad=True)
    pair_distances(x).sum().backward()
    assert torch.isfinite(x.grad).all()


# --------------------------------------------------------------------------- #
# Encoder / decoder
# --------------------------------------------------------------------------- #


def test_encoder_is_invariant_to_rigid_motions(toy_data):
    trajectories, _ = toy_data
    model = small_model(trajectories)
    x = trajectories[0][:20].double()
    z = model.encode(x.float())
    z_moved = model.encode(rigid_motion(x, seed=3).float())
    assert torch.allclose(z, z_moved, atol=1e-4)


def test_decoder_is_total_deterministic_and_centred(toy_data):
    trajectories, _ = toy_data
    model = small_model(trajectories, latent_dim=3)
    z = 5.0 * torch.randn(16, 3)  # far outside the data: still a valid structure
    x1, x2 = model.decode(z), model.decode(z)
    assert torch.equal(x1, x2)
    assert x1.shape == (16, trajectories[0].shape[1], 3) and torch.isfinite(x1).all()
    assert torch.allclose(x1.mean(-2), torch.zeros(16, 3), atol=1e-5)
    assert torch.allclose(x1, model.decoder.reference.expand_as(x1), atol=1e-6)  # zero-initialised head


def test_pullback_is_the_observable_of_the_decoded_structure(toy_data):
    trajectories, _ = toy_data
    model = small_model(trajectories)
    z = torch.randn(6, 2)
    radius = lambda x: center(x).norm(dim=-1).mean(-1)
    assert torch.allclose(model.pullback(z, radius), radius(model.decode(z)))


def test_label_head_predicts_labels_in_original_units(toy_data):
    trajectories, energies = toy_data
    model = LatentMD(8, latent_dim=2, hidden_dim=32, policy_hidden_dim=16, n_labels=1)
    labels = torch.cat(energies).unsqueeze(-1)
    model.initialize_statistics(torch.cat(trajectories), labels)
    out = model.labels(torch.randn(5, 2))
    assert out.shape == (5, 1)
    assert torch.allclose(model.labels(torch.zeros(1, 2)).flatten(), model.label_head(torch.zeros(1, 2)).flatten() * model.label_scale + model.label_mean)
    with pytest.raises(RuntimeError):
        small_model(trajectories).labels(torch.zeros(1, 2))


def test_initialize_statistics_is_chunked_and_subsampled(toy_data):
    trajectories, _ = toy_data
    x = torch.cat(trajectories)
    full, subset = LatentMD(8, latent_dim=2), LatentMD(8, latent_dim=2)
    full.initialize_statistics(x, max_frames=10**9)
    subset.initialize_statistics(x, max_frames=100)
    features = pair_distances(x)
    assert torch.allclose(full.encoder.feature_mean, features.mean(0), atol=1e-5)
    assert torch.allclose(full.encoder.feature_scale, features.std(0, unbiased=False), rtol=1e-3, atol=1e-4)
    assert torch.isfinite(subset.encoder.feature_scale).all()
    assert abs(float(subset.reference_msd / full.reference_msd) - 1.0) < 0.5
    assert torch.allclose(full.encode_frames(x, chunk_size=37), full.encode(x), atol=1e-6)


# --------------------------------------------------------------------------- #
# Transition policy
# --------------------------------------------------------------------------- #


def test_policy_is_a_gaussian_kernel_with_the_stated_mean_and_variance():
    policy = LatentLangevinPolicy(3, hidden_dim=16).double()
    z, z_next = torch.randn(10, 3, dtype=torch.float64), torch.randn(10, 3, dtype=torch.float64)
    drift, mobility = policy.drift(z)
    var = 2.0 * mobility
    manual = (-0.5 * ((z_next - z - drift).pow(2) / var + torch.log(2 * math.pi * var))).sum(-1)
    assert torch.allclose(policy.log_prob(z, z_next), manual, atol=1e-10)
    assert (mobility >= 1e-3).all() and (mobility <= 2.0).all()


def test_policy_drift_is_minus_mobility_times_gradient_plus_ito_correction():
    policy = LatentLangevinPolicy(3, hidden_dim=16, max_step=1e6).double()
    for parameter in policy.parameters():  # make the landscape and mobility non-trivial
        torch.nn.init.normal_(parameter, std=0.3)
    z = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
    (grad_f,) = torch.autograd.grad(policy.free_energy(z).sum(), z)
    mobility = policy.mobility(z)
    ito = torch.stack(
        [torch.autograd.grad(mobility[:, i].sum(), z, retain_graph=True)[0][:, i] for i in range(3)], dim=-1
    )
    drift, _ = policy.drift(z.detach())
    assert torch.allclose(drift, -mobility * grad_f + ito, atol=1e-9)


def test_drift_cap_limits_the_step_and_works_under_no_grad():
    policy = LatentLangevinPolicy(2, hidden_dim=8, max_step=0.5)
    z = 1e3 * torch.randn(4, 2)
    with torch.no_grad():
        drift, _ = policy.drift(z)
    assert (drift.abs() <= 0.5 + 1e-6).all()


def test_bias_potential_steers_rollouts():
    policy_model = LatentMD(8, latent_dim=2, hidden_dim=16, policy_hidden_dim=16)
    z0 = torch.zeros(512, 2)
    plain = policy_model.rollout(z0, 5, generator=torch.Generator().manual_seed(7))
    pushed = policy_model.rollout(z0, 5, bias=lambda z: 20.0 * z[..., 0], generator=torch.Generator().manual_seed(7))
    assert plain.shape == (6, 512, 2)
    assert (pushed[-1, :, 0] - plain[-1, :, 0]).mean() < -0.1  # a linear potential pulls towards -z_0


# --------------------------------------------------------------------------- #
# Synthetic MD
# --------------------------------------------------------------------------- #


def test_toy_force_is_minus_the_energy_gradient_and_energy_is_invariant(system):
    x = (system.reference("open") + 0.2 * torch.randn(6, 8, 3, dtype=torch.float64)).requires_grad_(True)
    (grad,) = torch.autograd.grad(system.energy(x).sum(), x)
    assert torch.allclose(system.force(x.detach()), -grad, atol=1e-8)
    moved = rigid_motion(x.detach(), seed=5)
    assert torch.allclose(system.energy(moved), system.energy(x.detach()), atol=1e-8)


def test_toy_ensembles_have_the_documented_shapes(toy_data):
    trajectories, energies = toy_data
    assert len(trajectories) == len(energies) == 4
    assert trajectories[0].shape == (250, 8, 3) and trajectories[0].dtype == torch.float32
    assert all(torch.isfinite(e).all() for e in energies)


# --------------------------------------------------------------------------- #
# Data pooling
# --------------------------------------------------------------------------- #


def test_lag_pairs_never_cross_trajectory_boundaries():
    lengths = [12, 5, 9]  # 5 is not longer than the lag below and contributes no pair
    trajectories, labels = [], []
    for k, n in enumerate(lengths):
        x = torch.zeros(n, 4, 3)
        x[:, 0, 0] = 1000.0 * k + torch.arange(n)  # marker: trajectory id and time
        trajectories.append(x)
        labels.append(torch.arange(n, dtype=torch.float32) + 1000.0 * k)
    data = TrajectoryEnsembles(trajectories, lag=5, labels=labels)
    assert len(data) == (12 - 5) + (9 - 5)
    x0, x1, lab = data.pairs()
    assert torch.equal(x1[:, 0, 0] - x0[:, 0, 0], torch.full((len(data),), 5.0))
    assert torch.equal(lab.squeeze(-1), x0[:, 0, 0])
    with pytest.raises(ValueError):
        TrajectoryEnsembles([torch.zeros(3, 4, 3)], lag=3)
    with pytest.raises(ValueError):
        TrajectoryEnsembles([torch.zeros(10, 4, 3)], lag=0)
    with pytest.raises(ValueError):
        TrajectoryEnsembles([torch.zeros(10, 4, 3)], lag=2, labels=[])


# --------------------------------------------------------------------------- #
# Joint learning
# --------------------------------------------------------------------------- #


def test_transition_likelihood_shapes_the_representation(toy_data):
    """The 'mutual adaptation': the rule's likelihood has a gradient into the encoder."""
    trajectories, _ = toy_data
    data = TrajectoryEnsembles(trajectories, lag=5)
    model = small_model(trajectories)
    x0, x1, _ = data.sample(64, torch.Generator().manual_seed(0))
    terms = model.loss_terms(x0, x1)
    assert set(terms) == {"reconstruction", "dynamics", "whitening", "cycle", "tube"}
    terms["dynamics"].backward()
    grad = {name: p.grad for name, p in model.named_parameters()}
    assert all(g is not None and g.abs().sum() > 0 for n, g in grad.items() if n.startswith("encoder"))
    assert grad["policy.energy_net.4.weight"].abs().sum() > 0  # landscape
    assert grad["policy.mobility_net.4.weight"].abs().sum() > 0  # mobility
    assert all(g is None for n, g in grad.items() if n.startswith("decoder"))  # decoder is not in the rule


def test_training_reduces_reconstruction_error_and_rollouts_stay_finite(toy_data):
    trajectories, energies = toy_data
    labels = [e.unsqueeze(-1) for e in energies]
    data = TrajectoryEnsembles(trajectories, lag=5, labels=labels)
    model = LatentMD(8, latent_dim=2, hidden_dim=32, policy_hidden_dim=16, n_labels=1)
    config = TrainConfig(steps=120, batch_size=128, warmup_steps=40, log_every=40)
    history = fit(model, data, config)
    assert [h["step"] for h in history] == [40, 80, 120]
    assert all(math.isfinite(v) for h in history for v in h.values())
    assert history[-1]["reconstruction"] < 0.8 * history[0]["reconstruction"]
    with torch.no_grad():
        z = model.encode(data.coords[:8])
        path = model.rollout(z, 20)
        structures = model.decode(path.reshape(-1, 2))
    assert torch.isfinite(path).all() and torch.isfinite(structures).all()
    assert not model.training


# --------------------------------------------------------------------------- #
# Real-data runner
# --------------------------------------------------------------------------- #


def test_split_holdout_by_files_and_by_tail():
    items = [torch.zeros(n, 2, 3) for n in (100, 100, 100, 8)]
    train, held, kept = latent_md_real.split_holdout(items, holdout_files=2, holdout_tail=0.2, lag=10)
    assert len(train) == 2 and kept == [2]  # the 8-frame trajectory is too short to hold out
    train, held, kept = latent_md_real.split_holdout(items, holdout_files=0, holdout_tail=0.25, lag=10)
    assert [t.shape[0] for t in train] == [75, 75, 75, 6]
    assert [h.shape[0] for h in held] == [25, 25, 25] and kept == [0, 1, 2]
    train, held, kept = latent_md_real.split_holdout(items, holdout_files=0, holdout_tail=0.0, lag=10)
    assert train is items and held == [] and kept == []
    with pytest.raises(SystemExit):
        latent_md_real.split_holdout(items, holdout_files=4, holdout_tail=0.2, lag=10)


def test_loading_arrays_applies_the_stride_and_checks_labels(tmp_path):
    coords = np.random.default_rng(0).normal(size=(25, 6, 3)).astype(np.float32)
    np.save(tmp_path / "a.npy", coords)
    np.savez(tmp_path / "a.npz", coords=coords)
    for name in ("a.npy", "a.npz"):
        x, n_raw = latent_md_real.load_trajectory(str(tmp_path / name), None, "name CA", stride=4)
        assert n_raw == 25 and torch.equal(x, torch.as_tensor(coords[::4]))
    np.save(tmp_path / "labels.npy", np.arange(25, dtype=np.float32))
    labels = latent_md_real.load_labels(str(tmp_path / "labels.npy"), 25, stride=4)
    assert labels.shape == (7, 1) and torch.equal(labels.squeeze(-1), torch.arange(0, 25, 4).float())
    with pytest.raises(SystemExit):
        latent_md_real.load_labels(str(tmp_path / "labels.npy"), 24, stride=4)
    np.save(tmp_path / "bad.npy", np.zeros((10, 6)))
    with pytest.raises(SystemExit):
        latent_md_real.load_trajectory(str(tmp_path / "bad.npy"), None, "name CA", stride=1)
    with pytest.raises(SystemExit):  # MDAnalysis-readable files need a topology
        latent_md_real.load_trajectory(str(tmp_path / "x.xtc"), None, "name CA", stride=1)


def test_loading_dcd_with_atom_selection_and_stride(tmp_path):
    mda = pytest.importorskip("MDAnalysis")
    from MDAnalysis.coordinates.DCD import DCDWriter

    n_ca, n_other, n_frames = 6, 3, 20
    rng = np.random.default_rng(0)
    frames = rng.normal(size=(n_frames, n_ca + n_other, 3)).astype(np.float32)
    universe = mda.Universe.empty(
        n_ca + n_other, n_residues=n_ca, atom_resindex=[min(i, n_ca - 1) for i in range(n_ca + n_other)], trajectory=True
    )
    universe.add_TopologyAttr("name", ["CA"] * n_ca + ["O"] * n_other)
    universe.add_TopologyAttr("resname", ["ALA"] * n_ca)
    universe.add_TopologyAttr("resid", list(range(1, n_ca + 1)))
    universe.atoms.positions = frames[0]
    universe.atoms.write(str(tmp_path / "top.pdb"))
    with DCDWriter(str(tmp_path / "run.dcd"), n_atoms=n_ca + n_other) as writer:
        for frame in frames:
            universe.atoms.positions = frame
            writer.write(universe.atoms)
    x, n_raw = latent_md_real.load_trajectory(str(tmp_path / "run.dcd"), str(tmp_path / "top.pdb"), "name CA", stride=3)
    assert n_raw == n_frames and x.shape == (7, n_ca, 3)
    assert torch.allclose(x, torch.as_tensor(frames[::3, :n_ca]), atol=1e-4)


def test_wrapped_molecule_warning(capsys):
    steady = torch.randn(1, 6, 3) + 0.01 * torch.randn(50, 6, 3)
    latent_md_real.warn_if_wrapped(steady, "steady")
    assert capsys.readouterr().out == ""
    jumped = steady.clone()
    jumped[25:, 0, 0] += 40.0  # one atom wraps around the box
    latent_md_real.warn_if_wrapped(jumped, "jumped")
    assert "wrapped" in capsys.readouterr().out


def test_distribution_and_autocorrelation_metrics():
    a = torch.randn(20000)
    assert latent_md_real.wasserstein_1d(a, a) == 0.0
    assert latent_md_real.wasserstein_1d(a, a + 2.0) == pytest.approx(2.0, abs=1e-4)
    phi, series = 0.9, []
    gen = torch.Generator().manual_seed(0)
    for _ in range(8):  # AR(1) processes: autocorrelation phi^lag
        x = torch.zeros(5000)
        noise = torch.randn(5000, generator=gen)
        for t in range(1, 5000):
            x[t] = phi * x[t - 1] + noise[t]
        series.append(x)
    measured = latent_md_real.autocorrelation(series, [1, 5, 10_000])
    assert measured[0] == pytest.approx(phi, abs=0.03) and measured[1] == pytest.approx(phi**5, abs=0.05)
    assert math.isnan(measured[2])


def test_runner_end_to_end_and_checkpoint_roundtrip(toy_data, tmp_path):
    trajectories, energies = toy_data
    traj_files, label_files = [], []
    for k, (x, e) in enumerate(zip(trajectories, energies)):
        np.save(tmp_path / f"t{k}.npy", x.numpy())
        np.save(tmp_path / f"e{k}.npy", e.numpy())
        traj_files.append(str(tmp_path / f"t{k}.npy"))
        label_files.append(str(tmp_path / f"e{k}.npy"))
    out = tmp_path / "out"
    report = latent_md_real.main(
        ["--traj", *traj_files, "--labels", *label_files, "--lag", "5", "--latent-dim", "2", "--hidden-dim", "32",
         "--steps", "60", "--holdout-tail", "0.3", "--rollout-chains", "16", "--rollout-steps", "20", "--out", str(out)]
    )
    for name in ("model.pt", "history.csv", "assessment.json", "latent_0_t0.npy", "latent_3_t3.npy"):
        assert (out / name).exists(), name
    assessment = json.loads((out / "assessment.json").read_text())
    assert assessment["comparison_data"] == "heldout" and "heldout" in assessment
    assert set(assessment["observables"]) == {"rmsd_to_mean", "radius_of_gyration", "label0"}
    assert report["train"]["relative_msd"] == pytest.approx(assessment["train"]["relative_msd"])

    checkpoint = torch.load(out / "model.pt", weights_only=True)
    model = LatentMD(**checkpoint["init_args"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    z_saved = torch.as_tensor(np.load(out / "latent_0_t0.npy"))
    assert torch.allclose(model.encode_frames(trajectories[0]), z_saved, atol=1e-5)


def test_runner_rejects_inconsistent_inputs(toy_data, tmp_path):
    trajectories, _ = toy_data
    np.save(tmp_path / "a.npy", trajectories[0].numpy())
    np.save(tmp_path / "b.npy", trajectories[1][:, :6].numpy())
    with pytest.raises(SystemExit):  # different atoms
        latent_md_real.main(["--traj", str(tmp_path / "a.npy"), str(tmp_path / "b.npy"), "--steps", "2", "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit):  # label count mismatch
        latent_md_real.main(["--traj", str(tmp_path / "a.npy"), "--labels", "x.npy", "y.npy", "--out", str(tmp_path / "o")])
