"""Train and assess the latent-space MD model on the synthetic two-domain complex.

Training ensembles start in ``open`` (random lab frame) and ``closed_a`` (fixed
frame). The held-out ensemble starts in ``closed_b`` in yet another lab frame and
is never used for fitting, so it probes transfer of the learned rule to a region
and frame the model has not seen. All reported numbers are computed here.

Usage:
    python latent_md_demo.py [--latent-dim 3] [--steps 3000] [--csv latent_md_summary.csv]
"""

from __future__ import annotations

import argparse
import csv
import time

import torch

from latent_md import LatentMD, TrainConfig, TrajectoryEnsembles, fit, rmsd
from latent_md_toy import STATE_NAMES, TwoDomainComplex, generate_ensembles

LAG = 10
N_CHAINS = 1500
N_ROLLOUT = 300
RELAX_TIMES = (0, 10, 20, 50, 100, 200, 400, 1000, 2000)


def frame_labels(system: TwoDomainComplex, trajectories, energies) -> list[torch.Tensor]:
    """Per-frame labels: [potential energy, one-hot contact state (open, a, b)]."""
    out = []
    for traj, energy in zip(trajectories, energies):
        state = system.state_labels(traj.double())
        onehot = torch.nn.functional.one_hot(state, len(STATE_NAMES)).float()
        out.append(torch.cat([energy.float().unsqueeze(-1), onehot], dim=-1))
    return out


@torch.no_grad()
def pair_metrics(model: LatentMD, data: TrajectoryEnsembles, n: int = 20000) -> dict[str, float]:
    """One-step quality on lag pairs: reconstruction, dynamics NLL, label R^2."""
    index = torch.randperm(len(data), generator=torch.Generator().manual_seed(0))[:n]
    x0, x1, lab0 = data.pairs(index)
    z0, z1 = model.encode(x0), model.encode(x1)
    msd = rmsd(model.decode(z0), x0).pow(2).mean()
    nll = -model.policy.log_prob(z0, z1).mean() / model.latent_dim
    r2 = 1 - (model.labels(z0) - lab0).pow(2).mean(0) / lab0.var(0)
    out = {
        "rmsd": float(msd.sqrt()),
        "relative_msd": float(msd / model.reference_msd),
        "nll_per_dim": float(nll),
        "r2_energy": float(r2[0]),
    }
    out.update({f"r2_{name}": float(r2[1 + k]) for k, name in enumerate(STATE_NAMES)})
    return out


@torch.no_grad()
def latent_vs_md(model: LatentMD, system: TwoDomainComplex, trajectories) -> dict[str, object]:
    """Compare populations, relaxation and structural validity of MD and latent rollouts."""
    traj = torch.stack(trajectories)  # (replicas, frames, atoms, 3)
    replicas, frames = traj.shape[:2]
    flat = traj.reshape(-1, *traj.shape[2:])
    state = system.state_labels(flat.double()).reshape(replicas, frames)
    md_pops = torch.stack([(state == s).double().mean() for s in range(len(STATE_NAMES))])
    z = model.encode(flat)
    encoded_pops = model.labels(z)[:, 1:].clamp(0, 1).mean(0)

    generator = torch.Generator().manual_seed(0)
    z0 = model.encode(traj[:, 0]).repeat_interleave(N_CHAINS // replicas, 0)
    path = model.rollout(z0, N_ROLLOUT, generator=generator)
    soft = model.labels(path.reshape(-1, model.latent_dim))[:, 1:].clamp(0, 1)
    soft = soft.reshape(N_ROLLOUT + 1, N_CHAINS, len(STATE_NAMES))
    rollout_pops = soft[100:].mean((0, 1))

    relax = []
    for t in RELAX_TIMES:
        if t % LAG or t // LAG > N_ROLLOUT or t >= frames:
            continue
        md_t = [(state[:, t] == s).double().mean().item() for s in range(len(STATE_NAMES))]
        relax.append((t, md_t, soft[t // LAG].mean(0).tolist()))

    def mean_energy(x: torch.Tensor) -> float:
        return float(system.energy(x.double()).mean())

    return {
        "md_pops": md_pops.tolist(),
        "encoded_pops": encoded_pops.tolist(),
        "rollout_pops": rollout_pops.tolist(),
        "relaxation": relax,
        "energy_raw_md": mean_energy(flat[::50]),
        "energy_decoded_md": mean_energy(model.decode(z[::50])),
        "energy_decoded_rollout": mean_energy(model.decode(path[10:].reshape(-1, model.latent_dim)[::50])),
        "max_abs_z_rollout": float(path.abs().max()),
        "max_abs_z_data": float(z.abs().max()),
    }


def main(argv: list[str] | None = None) -> dict[str, float]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--latent-dim", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--csv", default="latent_md_summary.csv")
    args = parser.parse_args(argv)

    system = TwoDomainComplex()
    train_specs = [("open", "random"), ("closed_a", "raw")]
    trajs, energies, _ = generate_ensembles(system, train_specs, n_replicas=12, n_frames=3000, seed=0)
    held, held_energies, _ = generate_ensembles(system, [("closed_b", "fixed")], n_replicas=12, n_frames=3000, seed=100)
    train = TrajectoryEnsembles(trajs, LAG, frame_labels(system, trajs, energies))
    test = TrajectoryEnsembles(held, LAG, frame_labels(system, held, held_energies))

    torch.manual_seed(0)
    model = LatentMD(n_atoms=system.n_atoms, latent_dim=args.latent_dim, n_labels=4)
    start = time.time()
    fit(model, train, TrainConfig(steps=args.steps), log=print)
    print(f"training time: {time.time() - start:.0f} s")

    metrics: dict[str, float] = {}
    for name, data in (("train", train), ("heldout", test)):
        for key, value in pair_metrics(model, data).items():
            metrics[f"{name}_{key}"] = value
    result = latent_vs_md(model, system, held)

    for name, data in (("train", train), ("heldout", test)):
        print(f"[{name}] " + "  ".join(f"{k[len(name) + 1:]}={v:.3f}" for k, v in metrics.items() if k.startswith(name)))
    print("populations (open, closed_a, closed_b) on the held-out ensemble")
    print("  MD frames:                ", [round(v, 3) for v in result["md_pops"]])
    print("  label head on encoded MD: ", [round(v, 3) for v in result["encoded_pops"]])
    print("  latent rollouts (t>=1000):", [round(v, 3) for v in result["rollout_pops"]])
    print("relaxation from the closed_b start (MD uses 12 replicas, latent 1500 chains)")
    for t, md_t, lat_t in result["relaxation"]:
        print(f"  t={t:5d}  MD {[round(v, 2) for v in md_t]}  latent {[round(v, 2) for v in lat_t]}")
    print(
        "true force-field energy: raw MD %.1f | decoded encoded-MD %.1f | decoded latent rollouts %.1f"
        % (result["energy_raw_md"], result["energy_decoded_md"], result["energy_decoded_rollout"])
    )
    for k, name in enumerate(STATE_NAMES):
        metrics[f"md_pop_{name}"] = result["md_pops"][k]
        metrics[f"encoded_pop_{name}"] = result["encoded_pops"][k]
        metrics[f"rollout_pop_{name}"] = result["rollout_pops"][k]
    for key in ("energy_raw_md", "energy_decoded_md", "energy_decoded_rollout", "max_abs_z_rollout", "max_abs_z_data"):
        metrics[key] = result[key]
    with open(args.csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows((k, f"{v:.4f}") for k, v in metrics.items())
    return metrics


if __name__ == "__main__":
    main()
