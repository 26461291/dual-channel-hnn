"""Fit and assess the latent-space MD model on real MD trajectories.

Every input file is one trajectory (a replica or member of an ensemble); frames of
different files may be in different lab frames, since the encoder only sees
rigid-motion-invariant pair distances. All files must contain the same atoms in the
same order. Coordinates keep their input length unit (MDAnalysis: angstrom).

Formats: ``.npy`` / ``.npz`` arrays of shape (frames, atoms, 3) (``.npz``: array
``coords`` or the first array), or anything MDAnalysis reads (needs ``--topology``;
``--selection`` picks the atoms, default ``name CA``). Optional per-frame labels
(potential energy, a collective variable, ...) come as ``.npy`` files of shape
(frames,) or (frames, k), one per trajectory.

Make the complex whole before running (no periodic-boundary jumps, e.g.
``gmx trjconv -pbc mol`` or ``cpptraj autoimage``); a heuristic warns when
single-frame atom jumps look like wrapping.

Because real data has no ground-truth states, the assessment compares MD with
latent rollouts through generic observables (RMSD to the mean structure, radius of
gyration, and any labels): distribution shift (1-D Wasserstein distance) and
autocorrelation at multiples of the lag.

Example:
    python latent_md_real.py --topology complex.pdb --traj r1.xtc r2.xtc r3.xtc \\
        --selection "name CA" --stride 5 --lag 10 --latent-dim 6 --out out/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from latent_md import LatentMD, TrainConfig, TrajectoryEnsembles, center, fit, rmsd

AUTOCORR_STEPS = (1, 2, 4, 8, 16)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_trajectory(path: str, topology: str | None, selection: str, stride: int) -> tuple[Tensor, int]:
    """(coordinates (frames, atoms, 3) float32 after striding, frame count before striding)."""
    suffix = Path(path).suffix.lower()
    if suffix in (".npy", ".npz"):
        data = np.load(path, mmap_mode="r") if suffix == ".npy" else np.load(path)
        if suffix == ".npz":
            data = data["coords"] if "coords" in data.files else data[data.files[0]]
        n_raw = int(data.shape[0])
        array = np.array(data[::stride], dtype=np.float32)
    else:
        if topology is None:
            raise SystemExit(f"--topology is required to read {path}")
        try:
            import MDAnalysis as mda
        except ImportError as error:
            raise SystemExit(
                f"reading {path} needs MDAnalysis (pip install MDAnalysis); "
                "alternatively export the coordinates to a .npy array (frames, atoms, 3)"
            ) from error
        universe = mda.Universe(topology, path)
        atoms = universe.select_atoms(selection)
        if len(atoms) == 0:
            raise SystemExit(f"selection {selection!r} matches no atoms")
        n_raw = len(universe.trajectory)
        array = np.stack([atoms.positions for _ in universe.trajectory[::stride]]).astype(np.float32)
    x = torch.from_numpy(array)
    if x.ndim != 3 or x.shape[-1] != 3:
        raise SystemExit(f"{path}: expected coordinates of shape (frames, atoms, 3), got {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise SystemExit(f"{path}: coordinates contain NaN or infinite values")
    return x, n_raw


def warn_if_wrapped(x: Tensor, name: str, factor: float = 30.0) -> None:
    """Heuristic: single-frame atom jumps far above the typical step suggest periodic wrapping."""
    if x.shape[0] < 3:
        return
    centred = center(x)
    step = (centred[1:] - centred[:-1]).norm(dim=-1)
    typical = float(step.median())
    if typical > 0 and float(step.max()) > factor * typical:
        print(
            f"WARNING {name}: an atom moves {float(step.max()):.3g} in one frame "
            f"(median {typical:.3g}); the molecule may be wrapped across the periodic box. "
            "Make it whole before running."
        )


def load_labels(path: str, n_raw_frames: int, stride: int) -> Tensor:
    """Per-frame labels (frames after striding, k) matching a trajectory of ``n_raw_frames`` frames."""
    array = np.load(path, mmap_mode="r")
    if array.shape[0] != n_raw_frames:
        raise SystemExit(f"{path}: {array.shape[0]} label rows but the trajectory has {n_raw_frames} frames")
    labels = torch.from_numpy(np.array(array[::stride], dtype=np.float32))
    if not torch.isfinite(labels).all():
        raise SystemExit(f"{path}: labels contain NaN or infinite values")
    return labels.reshape(labels.shape[0], -1)


def split_holdout(items: list[Tensor], holdout_files: int, holdout_tail: float, lag: int):
    """Train/held-out split of per-trajectory tensors (frames first).

    Holds out the last ``holdout_files`` trajectories, else the tail of every
    trajectory. Held-out pieces no more than ``lag`` frames long are dropped, since
    they contain no lag pair. Returns the split and the kept held-out indices, so
    that labels can be split identically.
    """
    if holdout_files:
        if holdout_files >= len(items):
            raise SystemExit("--holdout-files must be smaller than the number of trajectories")
        cut = len(items) - holdout_files
        kept = [k for k in range(cut, len(items)) if items[k].shape[0] > lag]
        return items[:cut], [items[k] for k in kept], kept
    if not holdout_tail:
        return items, [], []
    train, held, kept = [], [], []
    for k, item in enumerate(items):
        cut = int(round(item.shape[0] * (1.0 - holdout_tail)))
        train.append(item[:cut])
        if item.shape[0] - cut > lag:
            held.append(item[cut:])
            kept.append(k)
    return train, held, kept


# --------------------------------------------------------------------------- #
# Assessment helpers
# --------------------------------------------------------------------------- #


def wasserstein_1d(a: Tensor, b: Tensor, n_quantiles: int = 200) -> float:
    """1-D Wasserstein-1 distance between two samples via matched quantiles."""
    q = torch.linspace(0.0025, 0.9975, n_quantiles)

    def thin(v: Tensor) -> Tensor:  # torch.quantile rejects very large inputs
        v = v.float().flatten()
        return v[:: -(-v.numel() // 1_000_000)]

    return float((torch.quantile(thin(a), q) - torch.quantile(thin(b), q)).abs().mean())


def autocorrelation(series: list[Tensor], lags: list[int]) -> list[float]:
    """Pooled normalised autocorrelation of 1-D series at the given lags (nan if too short)."""
    flat = torch.cat(series)
    mean, var = flat.mean(), flat.var(unbiased=False).clamp_min(1e-30)
    out = []
    for lag in lags:
        products = [(s[:-lag] - mean) * (s[lag:] - mean) for s in series if len(s) > lag]
        out.append(float(torch.cat(products).mean() / var) if products else float("nan"))
    return out


def structure_observables(model: LatentMD, x: Tensor) -> dict[str, Tensor]:
    """Generic observables of structures: RMSD to the training mean structure and radius of gyration."""
    reference = model.decoder.reference
    return {
        "rmsd_to_mean": rmsd(x, reference.expand_as(x)),
        "radius_of_gyration": center(x).pow(2).sum(-1).mean(-1).sqrt(),
    }


@torch.no_grad()
def chunked(function, tensor: Tensor, size: int = 4096) -> dict[str, Tensor]:
    parts = [function(chunk) for chunk in tensor.split(size)]
    return {key: torch.cat([p[key] for p in parts]) for key in parts[0]}


@torch.no_grad()
def pair_quality(model: LatentMD, data: TrajectoryEnsembles, n_pairs: int, seed: int) -> dict[str, object]:
    """One-step quality on lag pairs: reconstruction, dynamics NLL per dimension, label R^2."""
    generator = torch.Generator().manual_seed(seed)
    index = torch.randperm(len(data), generator=generator)[:n_pairs]
    x0, x1, labels = data.pairs(index)
    z0, z1 = model.encode_frames(x0), model.encode_frames(x1)
    msd = torch.cat([rmsd(model.decode(c), x).pow(2) for c, x in zip(z0.split(2048), x0.split(2048))])
    mobility = model.policy.mobility(z0)
    floor = math.exp(model.policy.log_mobility_range[0])
    out: dict[str, object] = {
        "rmsd": float(msd.mean().sqrt()),
        "relative_msd": float(msd.mean() / model.reference_msd),
        "nll_per_dim": float(-model.policy.log_prob(z0, z1).mean() / model.latent_dim),
        "median_mobility": float(mobility.median()),
        "frac_mobility_at_floor": float((mobility < 1.5 * floor).float().mean()),
    }
    if labels is not None:
        r2 = 1 - (model.labels(z0) - labels).pow(2).mean(0) / labels.var(0).clamp_min(1e-12)
        out["label_r2"] = [float(v) for v in r2]
    return out


@torch.no_grad()
def compare_md_with_rollouts(
    model: LatentMD,
    trajectories: list[Tensor],
    labels: list[Tensor] | None,
    lag: int,
    n_chains: int,
    n_steps: int,
    seed: int,
) -> dict[str, object]:
    """MD versus latent rollouts through generic observables and (optional) labels.

    Rollouts start from random frames of ``trajectories`` and are decoded; they are
    compared with the raw MD frames and with the model's own round trip
    ``decode(encode(x))`` (which isolates what the latent dynamics adds).
    """
    generator = torch.Generator().manual_seed(seed)
    z_md = [model.encode_frames(t) for t in trajectories]
    z_all = torch.cat(z_md)
    start = torch.randint(len(z_all), (min(n_chains, len(z_all)),), generator=generator)
    path = model.rollout(z_all[start], n_steps, generator=generator)  # (steps + 1, chains, d)
    chains = path.shape[1]

    def of_latent(z: Tensor) -> dict[str, Tensor]:
        out = structure_observables(model, model.decode(z))
        if model.label_head is not None:
            for k, column in enumerate(model.labels(z).unbind(-1)):
                out[f"label{k}"] = column
        return out

    def of_structure(x: Tensor) -> dict[str, Tensor]:
        return structure_observables(model, x)

    raw = [chunked(of_structure, t) for t in trajectories]
    round_trip = [chunked(of_latent, z) for z in z_md]
    rolled = chunked(of_latent, path.reshape(-1, path.shape[-1]))
    if labels is not None:
        for k in range(labels[0].shape[1]):
            for series, column in zip(raw, labels):
                series[f"label{k}"] = column[:, k]

    lags_md = [k * lag for k in AUTOCORR_STEPS]
    lags_rollout = [k for k in AUTOCORR_STEPS if k < n_steps]
    table: dict[str, object] = {}
    for name in raw[0]:
        md_raw = [s[name] for s in raw]
        md_rt = [s[name] for s in round_trip]
        roll = rolled[name].reshape(n_steps + 1, chains)
        flat_rt, flat_roll = torch.cat(md_rt), roll[n_steps // 2 :].flatten()
        table[name] = {
            "md_raw_mean_std": [float(torch.cat(md_raw).mean()), float(torch.cat(md_raw).std())],
            "md_roundtrip_mean_std": [float(flat_rt.mean()), float(flat_rt.std())],
            "rollout_mean_std": [float(flat_roll.mean()), float(flat_roll.std())],
            "w1_rollout_vs_roundtrip_over_std": wasserstein_1d(flat_roll, flat_rt) / float(flat_rt.std().clamp_min(1e-12)),
            "w1_roundtrip_vs_raw_over_std": wasserstein_1d(flat_rt, torch.cat(md_raw)) / float(torch.cat(md_raw).std().clamp_min(1e-12)),
            "acf_steps": list(lags_rollout),
            "acf_md_raw": autocorrelation(md_raw, lags_md[: len(lags_rollout)]),
            "acf_md_roundtrip": autocorrelation(md_rt, lags_md[: len(lags_rollout)]),
            "acf_rollout": autocorrelation(list(roll.unbind(1)), lags_rollout),
        }
    return {
        "observables": table,
        "max_abs_z_data": float(z_all.abs().max()),
        "max_abs_z_rollout": float(path.abs().max()),
        "n_chains": chains,
        "n_steps": n_steps,
    }


def print_comparison(report: dict[str, object], lag: int) -> None:
    print(f"\nMD versus latent rollouts ({report['n_chains']} chains x {report['n_steps']} steps of {lag} frames)")
    print(f"max |z|: data {report['max_abs_z_data']:.2f}, rollouts {report['max_abs_z_rollout']:.2f}")
    print(
        "Read it as: rollout vs MD round trip = fidelity of the learned transition rule;\n"
        "            MD round trip vs MD raw  = what the latent space itself discards (the decoder returns\n"
        "            conditional-mean structures, so fast thermal noise is not reproduced)."
    )
    for name, row in report["observables"].items():
        fmt = lambda pair: f"{pair[0]:.4g} +- {pair[1]:.4g}"
        print(f"  {name}")
        print(f"    MD raw {fmt(row['md_raw_mean_std'])} | MD round trip {fmt(row['md_roundtrip_mean_std'])} | rollout {fmt(row['rollout_mean_std'])}")
        print(
            "    W1/std: rollout vs round trip %.3f, round trip vs raw %.3f"
            % (row["w1_rollout_vs_roundtrip_over_std"], row["w1_roundtrip_vs_raw_over_std"])
        )
        steps = ", ".join(f"{k * lag}f" for k in row["acf_steps"])
        for key, title in (("acf_md_raw", "MD raw"), ("acf_md_roundtrip", "MD round trip"), ("acf_rollout", "rollout")):
            print(f"    autocorrelation at {steps}: {title:13s} " + " ".join(f"{v:6.3f}" for v in row[key]))


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--traj", nargs="+", required=True, help="trajectory files, one per replica")
    parser.add_argument("--topology", help="topology file for MDAnalysis-readable trajectories")
    parser.add_argument("--selection", default="name CA", help="MDAnalysis atom selection (default: name CA)")
    parser.add_argument("--stride", type=int, default=1, help="keep every n-th frame")
    parser.add_argument("--labels", nargs="*", default=[], help="per-frame label .npy files, one per trajectory")
    parser.add_argument("--lag", type=int, default=10, help="lag in (strided) frames")
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--min-mobility", type=float, default=1e-3, help="resolution floor of the latent step noise")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--holdout-files", type=int, default=0, help="hold out the last K trajectories")
    parser.add_argument("--holdout-tail", type=float, default=0.2, help="else hold out this tail fraction of every trajectory")
    parser.add_argument("--rollout-chains", type=int, default=256)
    parser.add_argument("--rollout-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0: library default)")
    parser.add_argument("--out", default="latent_md_out")
    args = parser.parse_args(argv)
    if args.labels and len(args.labels) != len(args.traj):
        raise SystemExit("--labels needs exactly one file per trajectory")
    if not 0.0 <= args.holdout_tail < 1.0:
        raise SystemExit("--holdout-tail must be in [0, 1)")
    if args.threads:
        torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    names = [f"{k}_{Path(p).stem}" for k, p in enumerate(args.traj)]
    trajectories, labels = [], []
    for k, (name, path) in enumerate(zip(names, args.traj)):
        x, n_raw = load_trajectory(path, args.topology, args.selection, args.stride)
        if args.labels:
            labels.append(load_labels(args.labels[k], n_raw, args.stride))
        warn_if_wrapped(x, name)
        trajectories.append(x)
    n_atoms = trajectories[0].shape[1]
    if any(t.shape[1] != n_atoms for t in trajectories):
        raise SystemExit("all trajectories must contain the same atoms")
    if args.labels and len({lab.shape[1] for lab in labels}) != 1:
        raise SystemExit("all label files must have the same number of columns")
    if n_atoms > 200:
        print(f"WARNING: {n_atoms} atoms give {n_atoms * (n_atoms - 1) // 2} encoder inputs; training is "
              "CPU-bound and slow. Consider a coarser selection (e.g. CA atoms).")
    print(f"loaded {len(trajectories)} trajectories, {sum(t.shape[0] for t in trajectories)} frames, {n_atoms} atoms")

    train_x, held_x, kept = split_holdout(trajectories, args.holdout_files, args.holdout_tail, args.lag)
    train_l = held_l = None
    if args.labels:
        train_l, _, _ = split_holdout(labels, args.holdout_files, args.holdout_tail, args.lag)
        held_l = [
            labels[k][len(labels[k]) - len(x) :] for k, x in zip(kept, held_x)
        ]  # the held-out piece is always the tail
    wanted_holdout = bool(args.holdout_files or args.holdout_tail)
    if wanted_holdout and not held_x:
        raise SystemExit("no held-out trajectory is longer than --lag; lower --holdout-tail or --lag")
    if not any(t.shape[0] > args.lag for t in train_x):
        raise SystemExit("no training trajectory is longer than --lag")
    train = TrajectoryEnsembles(train_x, args.lag, train_l)
    held = TrajectoryEnsembles(held_x, args.lag, held_l) if held_x else None
    print(
        f"training on {train.coords.shape[0]} frames ({len(train)} lag pairs)"
        + (f"; held out {held.coords.shape[0]} frames" if held else "; no held-out data")
    )

    torch.manual_seed(args.seed)
    n_labels = train_l[0].shape[1] if train_l else 0
    init_args = dict(
        n_atoms=n_atoms,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_labels=n_labels,
        min_mobility=args.min_mobility,
    )
    model = LatentMD(**init_args)
    config = TrainConfig(
        steps=args.steps, batch_size=args.batch_size, warmup_steps=min(1000, args.steps // 3), seed=args.seed
    )
    start = time.time()
    history = fit(model, train, config, log=print)
    print(f"training time: {time.time() - start:.0f} s")
    torch.save({"init_args": init_args, "state_dict": model.state_dict(), "lag": args.lag}, out / "model.pt")
    with open(out / "history.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    for name, trajectory in zip(names, trajectories):
        np.save(out / f"latent_{name}.npy", model.encode_frames(trajectory).numpy())

    report: dict[str, object] = {"n_atoms": n_atoms, "lag": args.lag, "latent_dim": args.latent_dim}
    report["train"] = pair_quality(model, train, 20000, args.seed)
    if held is not None:
        report["heldout"] = pair_quality(model, held, 20000, args.seed)
    for key in ("train", "heldout"):
        if key in report:
            print(f"[{key}] " + "  ".join(f"{k}={v}" if isinstance(v, list) else f"{k}={v:.4g}" for k, v in report[key].items()))
    if report["train"]["frac_mobility_at_floor"] > 0.2:
        print(
            "WARNING: the latent step noise sits at its floor for much of the data, i.e. the lag is short "
            "compared with the latent motion the model can resolve. Increase --lag (or --stride) or lower --min-mobility."
        )
    assessed_on = "heldout" if held is not None else "train"
    report["comparison_data"] = assessed_on
    comparison = compare_md_with_rollouts(
        model,
        held_x if held is not None else train_x,
        held_l if held is not None else train_l,
        args.lag,
        args.rollout_chains,
        args.rollout_steps,
        args.seed,
    )
    report.update(comparison)
    print(f"(comparison on {assessed_on} trajectories)" + ("" if held is not None else " -- no held-out data, in-sample only"))
    print_comparison(report, args.lag)
    with open(out / "assessment.json", "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nwrote model.pt, history.csv, assessment.json and latent_*.npy to {out}/")
    return report


if __name__ == "__main__":
    main()
