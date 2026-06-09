"""
================================================================================
ROBUSTNESS EVALUATION  —  Module 2 runner   (PINN, temporal extrapolation)
Team : Turingz  (Dharmapala R.D.)
File : module2_robustness/robustness_eval_pinn.py

WHY THIS IS A SEPARATE RUNNER FROM robustness_eval.py
-----------------------------------------------------
FNO and DeepONet are OPERATOR learners: they take the initial condition as an
input, so feeding them three out-of-FAMILY ICs (common/ood_spec.py) is a
meaningful generalisation test, and one model handles all three.

A PINN is a SINGLE-INSTANCE solver. Each trained BurgersPINN is fitted to ONE
specific initial condition by baking that IC into its loss; it does NOT read an
IC at inference time (BurgersPINN.predict literally ignores its `ic` argument and
warns if you pass a different one). So the fixed-OOD-IC design is structurally
meaningless for a PINN — you cannot ask it to generalise to sin(3 pi x) when it
was never given sin(3 pi x).

The robustness axis that IS meaningful for a single-instance solver is TEMPORAL
EXTRAPOLATION: the PINN is trained on t in [0, t_train_end] (=1.0) and we ask how
its solution degrades on the unseen window t in (t_train_end, T] (=(1, 2]). This
runner therefore, for each trained PINN sample:

    1. loads the trained checkpoint (no retraining),
    2. rolls it out on ITS OWN trained IC over the full grid t in [0, 2],
    3. compares against that sample's exact Cole-Hopf reference (stored in the
       checkpoint's problem.npz — same IC, same nu),
    4. records, at every time step:
         * relative-L2 error            (common/metrics.py,           shared)
         * spectral distance D(t)       (common/spectral_metrics.py,  shared)
    5. reports the train-window vs extrapolation-window errors, and the
       early-warning lead = how much earlier D(t) departs from its in-training
       baseline than relative L2 does, as the model crosses t_train_end.

The METRIC definitions are the same shared single-source-of-truth modules used by
the FNO/DeepONet runner, so the spectral numbers remain directly comparable; only
the experimental axis (time, not IC) differs, as the physics demands.

Outputs (under results/robustness/pinn/):
    pinn_robustness.csv              one row per PINN sample
    pinn_robustness_timeseries.json  per-time rel-L2 and D(t) curves + leads
    pinn_robustness_plot.png         rel-L2 and D(t) vs t, all samples overlaid

Run (needs deepxde; use the fyp-pde env):
    python module2_robustness/robustness_eval_pinn.py
    python module2_robustness/robustness_eval_pinn.py --weight-mode high_freq
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── path setup: project root for common.*, ml_models/pinn for its package ────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.metrics import _per_time_rel_l2  # noqa: E402  (shared rel-L2)
from common.spectral_metrics import (spectral_distance_per_t,  # noqa: E402
                                     first_crossing_time, early_warning_lead)

WEIGHT_MODES = ("low_freq", "uniform", "high_freq")
DEFAULT_PINN_ROOT = os.path.join(_PROJECT_ROOT, "results", "pinn")


# =============================================================================
#  Solver loading (the only PINN-specific part)
# =============================================================================
def load_pinn_solver(sample_dir: str):
    """Load a trained BurgersPINN from its checkpoint dir.

    Returns (solver, x, t, u_ref, ic, t_train_end, nu) — everything the
    temporal-extrapolation evaluation needs, all read back from the
    self-contained checkpoint (problem.npz + metadata.json).
    """
    from ml_models.pinn.model import BurgersPINN  # lazy: needs deepxde

    solver = BurgersPINN()
    solver.load(sample_dir)
    ds = solver.ds
    return (solver,
            np.asarray(ds.x, dtype=np.float64),
            np.asarray(ds.t, dtype=np.float64),
            np.asarray(ds.u_ref, dtype=np.float64),
            np.asarray(ds.ic_ref, dtype=np.float64),
            float(ds.t_train_end),
            float(ds.nu))


def discover_sample_dirs(root: str) -> List[str]:
    """Every results/pinn/sampleN dir that holds a finished checkpoint."""
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if (os.path.isdir(d)
                and os.path.isfile(os.path.join(d, "metadata.json"))
                and os.path.isfile(os.path.join(d, "problem.npz"))):
            out.append(d)
    return out


# =============================================================================
#  Per-sample evaluation
# =============================================================================
def evaluate_sample(sample_dir: str, weight_mode: str = "low_freq") -> Dict:
    """Roll out one trained PINN on its own IC over [0, 2] and score it."""
    (solver, x, t, u_ref, ic, t_train_end, nu) = load_pinn_solver(sample_dir)

    u_pred = np.asarray(solver.rollout(ic, x, t), dtype=np.float64)
    assert u_pred.shape == u_ref.shape, (u_pred.shape, u_ref.shape)

    rel_l2 = _per_time_rel_l2(u_pred, u_ref)                      # (nt,)
    spec_d = spectral_distance_per_t(u_pred, u_ref, weight_mode)  # (nt,)

    # Train-window vs extrapolation-window aggregates (the central PINN story).
    tr = t <= t_train_end + 1e-12
    ex = ~tr

    def _seg(arr, m):
        return (float(np.mean(arr[m])), float(np.max(arr[m]))) if m.any() else (float("nan"),) * 2

    rel_tr_mean, rel_tr_max = _seg(rel_l2, tr)
    rel_ex_mean, rel_ex_max = _seg(rel_l2, ex)

    # Early-warning lead under every weighting. The baseline window sits well
    # inside the trusted training region, so a crossing flags departure from the
    # in-training behaviour; the lead is how much sooner D(t) departs than rel-L2.
    base_win = (0.0, 0.5 * t_train_end)
    leads = {wm: early_warning_lead(rel_l2,
                                    spectral_distance_per_t(u_pred, u_ref, wm),
                                    t, baseline_window=base_win)
             for wm in WEIGHT_MODES}

    sample_name = os.path.basename(sample_dir.rstrip("/\\"))
    return {
        "sample": sample_name,
        "nu": nu,
        "weight_mode": weight_mode,
        "t_train_end": t_train_end,
        "rel_l2_train_mean": rel_tr_mean,
        "rel_l2_train_max": rel_tr_max,
        "rel_l2_extrap_mean": rel_ex_mean,
        "rel_l2_extrap_max": rel_ex_max,
        "final_rel_l2": float(rel_l2[-1]),
        "final_spec_dist": float(spec_d[-1]),
        "first_cross_rel_l2": leads[weight_mode]["first_cross_rel_l2"],
        "first_cross_spec_dist": leads[weight_mode]["first_cross_spec_dist"],
        "early_warning_lead": leads[weight_mode]["early_warning_lead"],
        "leads_by_weight": leads,
        "rel_l2_series": rel_l2.tolist(),
        "spec_dist_series": spec_d.tolist(),
        "t": t.tolist(),
    }


def run_robustness(sample_dirs: List[str], weight_mode: str = "low_freq") -> List[Dict]:
    results = []
    for d in sample_dirs:
        name = os.path.basename(d.rstrip("/\\"))
        print(f"  > {name} ...")
        r = evaluate_sample(d, weight_mode)
        print(f"      rel-L2  train(mean)={r['rel_l2_train_mean']:.3e}  "
              f"extrap(mean)={r['rel_l2_extrap_mean']:.3e}  "
              f"final={r['final_rel_l2']:.3e}   lead({weight_mode})={r['early_warning_lead']}")
        results.append(r)
    return results


# =============================================================================
#  Output writers
# =============================================================================
def write_csv(results: List[Dict], path: str) -> None:
    keys = ["sample", "nu", "weight_mode", "t_train_end",
            "rel_l2_train_mean", "rel_l2_train_max",
            "rel_l2_extrap_mean", "rel_l2_extrap_max",
            "final_rel_l2", "final_spec_dist",
            "first_cross_rel_l2", "first_cross_spec_dist", "early_warning_lead"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in keys})


def write_json(results: List[Dict], path: str, weight_mode: str) -> None:
    payload = {
        "model": "PINN",
        "module": "module2_robustness",
        "axis": "temporal_extrapolation",
        "weight_mode": weight_mode,
        "note": ("PINN is single-instance; robustness is measured as temporal "
                 "extrapolation past t_train_end on each sample's own IC, not by "
                 "feeding out-of-family ICs (which a PINN cannot read)."),
        "samples": results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def plot_results(results: List[Dict], path: str, weight_mode: str) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("PINN Robustness (temporal extrapolation) | Module 2 | Team Turingz "
                 f"| weight={weight_mode}", fontsize=12, fontweight="bold")

    t_end = results[0]["t_train_end"] if results else 1.0
    for r in results:
        t = np.asarray(r["t"])
        ax0.semilogy(t, np.maximum(np.asarray(r["rel_l2_series"]), 1e-12),
                     lw=1.2, label=r["sample"])
        ax1.plot(t, np.asarray(r["spec_dist_series"]), lw=1.2, label=r["sample"])

    for ax in (ax0, ax1):
        ax.axvline(t_end, color="k", ls="--", lw=1, label="train / extrapolation split")
        ax.set_xlabel("t"); ax.grid(alpha=0.3, which="both")
    ax0.set_title("relative L2 vs t"); ax0.set_ylabel("relative L2")
    ax1.set_title("spectral distance D(t) vs t")
    ax1.set_ylabel("weighted Fourier-amplitude distance")
    ax0.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
#  Main
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="PINN Module 2 robustness — temporal extrapolation on each "
                    "trained sample's own IC (single-instance solver).")
    ap.add_argument("--pinn-root", default=DEFAULT_PINN_ROOT,
                    help="Dir containing sampleN/ PINN checkpoint subdirs.")
    ap.add_argument("--weight-mode", default="low_freq", choices=WEIGHT_MODES,
                    help="Spectral-distance mode weighting for plots/CSV.")
    ap.add_argument("--out-dir",
                    default=os.path.join(_PROJECT_ROOT, "results", "robustness", "pinn"),
                    help="Output directory for CSV/JSON/plot.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "pinn_robustness.csv")
    json_path = os.path.join(args.out_dir, "pinn_robustness_timeseries.json")
    plot_path = os.path.join(args.out_dir, "pinn_robustness_plot.png")

    print("=" * 72)
    print("  PINN ROBUSTNESS EVAL  |  Module 2  |  Team Turingz")
    print("  axis: TEMPORAL EXTRAPOLATION (single-instance solver)")
    print("=" * 72)

    sample_dirs = discover_sample_dirs(args.pinn_root)
    if not sample_dirs:
        raise SystemExit(f"No finished PINN checkpoints under {args.pinn_root} "
                         "(need sampleN/ dirs with metadata.json + problem.npz).")
    print(f"\n  Found {len(sample_dirs)} PINN sample(s) under {args.pinn_root}")
    print(f"  Weighting    : {args.weight_mode}\n")

    results = run_robustness(sample_dirs, args.weight_mode)

    write_csv(results, csv_path)
    write_json(results, json_path, args.weight_mode)
    plot_results(results, plot_path, args.weight_mode)

    # Cross-sample summary (the headline degradation number).
    ex_means = [r["rel_l2_extrap_mean"] for r in results if np.isfinite(r["rel_l2_extrap_mean"])]
    tr_means = [r["rel_l2_train_mean"] for r in results if np.isfinite(r["rel_l2_train_mean"])]
    print("\n" + "-" * 72)
    if tr_means and ex_means:
        print(f"  across {len(results)} samples:  "
              f"rel-L2 train(mean)={np.mean(tr_means):.3e}   "
              f"extrap(mean)={np.mean(ex_means):.3e}   "
              f"(x{np.mean(ex_means)/max(np.mean(tr_means),1e-12):.1f} worse out-of-window)")
    print("-" * 72)
    print(f"  CSV   : {csv_path}")
    print(f"  JSON  : {json_path}")
    print(f"  Plot  : {plot_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
