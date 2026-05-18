"""
================================================================================
DEEPONET MULTI-SEED SENSOR-COUNT SWEEP
Team    : Turingz
File    : ml_models/DeepONet/sensor_sweep.py

Trains the DeepONet at several sensor counts m, each with multiple
random seeds, reports mean and std of extrapolation accuracy, and
recommends the smallest m whose mean is statistically indistinguishable
from the best-performing m (defined as mean within one standard
deviation of the best mean).

Self-contained: defines its own setup_device helper so it works even
if train.py is an older copy without the GPU helper.

Usage (run from project root)
-----------------------------
    python ml-models/DeepONet/sensor_sweep.py \
        --data data/burgers_1d_cole_hopf.pt \
        --iterations 4000 \
        --sensors 64 100 128 256 \
        --seeds 0 1 2 3 4 \
        --cooldown 15
================================================================================
"""

# ── Project-root + local-folder import setup ──────────────────────────────────
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
for _p in (_PROJECT_ROOT, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import gc
import json
import time

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np

try:
    import torch
except ImportError:
    torch = None

# ── Project ───────────────────────────────────────────────────────────────────
from deeponet_solver import DeepONetSolver
from train           import load_dataset


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE HELPER  (kept local; mirrors the one in train.py)
# ─────────────────────────────────────────────────────────────────────────────
def setup_device(requested: str = "auto") -> str:
    """Select GPU/CPU for DeepXDE+PyTorch and print a startup banner."""
    if torch is None:
        print("  device      : torch not available")
        return "cpu"

    if requested == "cpu":
        device = "cpu"
    elif requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False.")
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.cuda.set_device(0)
        torch.set_default_device("cuda")
        name = torch.cuda.get_device_name(0)
        cuda_v = torch.version.cuda
        mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  device      : GPU 0 - {name}  ({mem:.1f} GiB, CUDA {cuda_v})")
    else:
        torch.set_default_device("cpu")
        print(f"  device      : CPU  (torch.cuda.is_available() = {torch.cuda.is_available()})")
    return device


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Multi-seed sweep over DeepONet sensor counts.")
    p.add_argument("--data",       required=True)
    p.add_argument("--out",        default=str(_HERE / "sweep_results"),
                   help="Output directory (defaults to <script_dir>/sweep_results).")
    p.add_argument("--sensors",    type=int, nargs="+",
                   default=[64, 100, 128, 256])
    p.add_argument("--seeds",      type=int, nargs="+",
                   default=[0, 1, 2, 3, 4],
                   help="Random seeds; one run per (m, seed) combination.")
    p.add_argument("--iterations", type=int, default=4000,
                   help="Training iterations per run.")
    p.add_argument("--cooldown",   type=int, default=15,
                   help="Seconds to pause between runs (thermal management).")
    p.add_argument("--device",     type=str, default="auto",
                   choices=["auto", "cuda", "cpu"],
                   help="Computation device.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def summarise_m(m, m_runs, n_seeds):
    ext_vals  = np.array([r["extrapolation_rel_l2_mean"] for r in m_runs])
    ind_vals  = np.array([r["in_dist_rel_l2_mean"]       for r in m_runs])
    wall_vals = np.array([r["wall_time_s"]               for r in m_runs])

    def _std(v):
        return float(v.std(ddof=1)) if len(v) > 1 else 0.0

    return {
        "n_sensors":           m,
        "n_seeds":             n_seeds,
        "extrap_l2_mean":      float(ext_vals.mean()),
        "extrap_l2_std":       _std(ext_vals),
        "extrap_l2_per_seed":  ext_vals.tolist(),
        "in_dist_l2_mean":     float(ind_vals.mean()),
        "in_dist_l2_std":      _std(ind_vals),
        "in_dist_l2_per_seed": ind_vals.tolist(),
        "wall_s_mean":         float(wall_vals.mean()),
        "wall_s_total":        float(wall_vals.sum()),
    }


def write_partial(out_file, args, per_m_summary, all_runs, total_wall, complete):
    payload = {
        "complete":            complete,
        "iterations_per_run":  args.iterations,
        "sensors":             args.sensors,
        "seeds":               args.seeds,
        "n_runs_total":        len(args.sensors) * len(args.seeds),
        "n_runs_done":         len(all_runs),
        "wall_time_total_s":   total_wall,
        "per_m":               per_m_summary,
        "all_runs":            all_runs,
    }
    if complete and per_m_summary:
        means = np.array([s["extrap_l2_mean"] for s in per_m_summary])
        stds  = np.array([s["extrap_l2_std"]  for s in per_m_summary])
        best_idx  = int(np.argmin(means))
        best_m    = per_m_summary[best_idx]["n_sensors"]
        best_mean = float(means[best_idx])
        best_std  = float(stds[best_idx])
        threshold = best_mean + best_std
        sorted_by_m = sorted(per_m_summary, key=lambda x: x["n_sensors"])
        recommended = next(
            (s["n_sensors"] for s in sorted_by_m if s["extrap_l2_mean"] <= threshold),
            best_m,
        )
        payload["best_n_sensors"]         = best_m
        payload["best_extrap_l2_mean"]    = best_mean
        payload["best_extrap_l2_std"]     = best_std
        payload["recommended_n_sensors"]  = recommended
        payload["recommendation_rule"]    = "smallest m with mean extrap L2 <= best_mean + best_std"
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "sweep_results.json"

    n_runs = len(args.sensors) * len(args.seeds)
    print("=" * 72)
    print(f"  DeepONet MULTI-SEED sensor-count sweep")
    print(f"  sensors        : {args.sensors}")
    print(f"  seeds          : {args.seeds}   ({len(args.seeds)} per sensor count)")
    print(f"  iterations     : {args.iterations}    per run")
    print(f"  cooldown       : {args.cooldown}s between runs (thermal)")
    print(f"  total runs     : {n_runs}")
    print(f"  results JSON   : {out_file}")
    print("=" * 72)
    setup_device(args.device)

    print(f"\nLoading dataset: {args.data}")
    dataset = load_dataset(args.data)

    all_runs = []
    per_m_summary = []
    t0_global = time.perf_counter()

    write_partial(out_file, args, per_m_summary, all_runs, 0.0, complete=False)

    for mi, m in enumerate(args.sensors):
        print(f"\n{'=' * 72}")
        print(f"  Sensor count m = {m}   ({mi + 1}/{len(args.sensors)})")
        print(f"{'=' * 72}")

        m_runs = []
        for si, seed in enumerate(args.seeds):
            run_no = mi * len(args.seeds) + si + 1
            print(f"\n{'-' * 72}")
            print(f"  Run {run_no}/{n_runs}   m = {m}   seed = {seed}")
            print(f"{'-' * 72}")

            solver = DeepONetSolver(
                n_sensors  = m,
                iterations = args.iterations,
                seed       = seed,
            )
            info = solver.fit(dataset)
            info["seed"] = seed
            m_runs.append(info)
            all_runs.append(info)

            ind = info.get("in_dist_rel_l2_mean",       float("nan"))
            ext = info.get("extrapolation_rel_l2_mean", float("nan"))
            print(f"  -> wall {info['wall_time_s']:6.1f}s   "
                  f"in-dist L2 {ind:.4f}   extrap L2 {ext:.4f}")

            del solver
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if args.cooldown > 0 and run_no < n_runs:
                print(f"  cooling down for {args.cooldown}s ...")
                time.sleep(args.cooldown)

        s = summarise_m(m, m_runs, len(args.seeds))
        per_m_summary.append(s)
        print(f"\n  -- m = {m} summary across {len(args.seeds)} seeds --")
        print(f"     extrap L2  : {s['extrap_l2_mean']:.4f} +/- {s['extrap_l2_std']:.4f}")
        print(f"     in-dist L2 : {s['in_dist_l2_mean']:.4f} +/- {s['in_dist_l2_std']:.4f}")
        print(f"     wall (avg) : {s['wall_s_mean']:.1f}s   total {s['wall_s_total']:.1f}s")

        write_partial(out_file, args, per_m_summary, all_runs,
                      time.perf_counter() - t0_global, complete=False)

    total_wall = time.perf_counter() - t0_global

    means = np.array([s["extrap_l2_mean"] for s in per_m_summary])
    stds  = np.array([s["extrap_l2_std"]  for s in per_m_summary])
    best_idx  = int(np.argmin(means))
    best_m    = per_m_summary[best_idx]["n_sensors"]
    best_mean = float(means[best_idx])
    best_std  = float(stds[best_idx])
    threshold = best_mean + best_std

    sorted_by_m = sorted(per_m_summary, key=lambda x: x["n_sensors"])
    recommended = next(
        (s["n_sensors"] for s in sorted_by_m if s["extrap_l2_mean"] <= threshold),
        best_m,
    )

    write_partial(out_file, args, per_m_summary, all_runs, total_wall, complete=True)

    print("\n" + "=" * 72)
    print(f"  MULTI-SEED RESULTS   ({len(args.seeds)} seeds x {len(args.sensors)} sensor counts)")
    print("=" * 72)
    print(f"  {'m':>5}   {'wall avg':>9}   {'in-dist L2':>20}   {'extrap L2':>20}")
    print("  " + "-" * 65)
    for s in per_m_summary:
        m_  = s["n_sensors"]
        wt  = s["wall_s_mean"]
        ind = f"{s['in_dist_l2_mean']:.4f} +/- {s['in_dist_l2_std']:.4f}"
        ext = f"{s['extrap_l2_mean']:.4f} +/- {s['extrap_l2_std']:.4f}"
        mark = "  <--" if m_ == recommended else ""
        print(f"  {m_:>5}   {wt:>9.1f}   {ind:>20}   {ext:>20}{mark}")
    print("=" * 72)
    print(f"  Best mean extrap L2     : {best_mean:.4f} +/- {best_std:.4f}   (m = {best_m})")
    print(f"  Recommended sensor count: m = {recommended}")
    print(f"    rule: smallest m whose mean extrap L2 <= best_mean + best_std ({threshold:.4f})")
    print(f"  Total wall time         : {total_wall / 60:.1f} min over {n_runs} runs")
    print(f"  Results JSON            : {out_file}")
    print("=" * 72)


if __name__ == "__main__":
    main()