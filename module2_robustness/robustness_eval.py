"""
================================================================================
ROBUSTNESS EVALUATION  —  Module 2 runner   (FNO)
Team : Turingz  (Dharmapala R.D.)
File : module2_robustness/robustness_eval.py

Implements the spectral-aware robustness mechanism from the interim report
(sections 4.3.2 / 5.6 / 6.6.2). For each of the three fixed out-of-FAMILY
initial conditions defined in common/ood_spec.py, this:

    1. loads the ALREADY-TRAINED FNO checkpoint (no retraining),
    2. rolls it out from t=0 to t=2 on the OOD initial condition,
    3. compares against the exact Cole-Hopf reference (regenerated with the
       OOD viscosity for OOD-3),
    4. records, at every time step:
         * relative-L2 error            (common/metrics.py, shared)
         * spectral distance D(t)       (common/spectral_metrics.py, shared)
    5. reports the final-time (t=2) numbers for the report table, and the
       early-warning lead time = how much earlier D(t) crosses its baseline
       threshold than relative L2 does.

The metric/spec/reference code all lives in common/, so the PINN and DeepONet
teammates reuse this exact runner: they only swap `load_fno_solver` for their
own loader that returns an object with a .rollout(ic, x, t) -> (nt, nx) method.

Outputs (under results/robustness/):
    fno_robustness.csv              one row per OOD case (table numbers)
    fno_robustness_timeseries.json  per-time rel-L2 and D(t) curves + leads
    fno_robustness_plot.png         2 x 3 panel: rel-L2 and D(t) vs t per case

Run from the project root (needs `neuralop` installed to load the FNO):
    python module2_robustness/robustness_eval.py
    python module2_robustness/robustness_eval.py --weight-mode high_freq
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Callable, Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── path setup: project root for common.*, ml_models/fno for fno_solver ──────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
_FNO_DIR = os.path.join(_PROJECT_ROOT, "ml_models", "fno")
for _p in (_PROJECT_ROOT, _FNO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.ood_spec import (OODCase, ood_cases, cole_hopf_reference,  # noqa: E402
                             native_grid, T_TRAIN_END)
from common.metrics import _per_time_rel_l2  # noqa: E402  (shared rel-L2)
from common.spectral_metrics import (spectral_distance_per_t,  # noqa: E402
                                     early_warning_lead)

WEIGHT_MODES = ("low_freq", "uniform", "high_freq")

# A rollout is any callable (ic, x_grid, t_grid) -> u(t, x) of shape (nt, nx).
RolloutFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


# =============================================================================
#  Solver loading (the ONLY FNO-specific part)
# =============================================================================
def load_fno_solver(checkpoint_path: str):
    """Load the trained FNO and return (solver, x_native).

    x_native is the solver's native spatial grid; the IC must be sampled on it.
    """
    from fno_solver import FNOSolver  # imported lazily so common.* tests need no neuralop

    solver = FNOSolver()
    solver.load(checkpoint_path)
    x_native = solver._x_grid.detach().cpu().numpy()
    return solver, x_native


# =============================================================================
#  Model-agnostic core
# =============================================================================
def evaluate_case(rollout_fn: RolloutFn, case: OODCase,
                  x: np.ndarray, t: np.ndarray,
                  weight_mode: str = "low_freq") -> Dict:
    """Roll out + score one OOD case. Solver-independent given rollout_fn."""
    ic = case.initial_condition(x)
    u_pred = np.asarray(rollout_fn(ic, x, t), dtype=np.float64)
    u_ref = cole_hopf_reference(case, x, t)
    assert u_pred.shape == u_ref.shape, (u_pred.shape, u_ref.shape)

    rel_l2 = _per_time_rel_l2(u_pred, u_ref)                      # (nt,)
    spec_d = spectral_distance_per_t(u_pred, u_ref, weight_mode)  # (nt,)

    # Early-warning lead under every weighting (resolves "which weight?" empirically).
    leads = {wm: early_warning_lead(rel_l2,
                                    spectral_distance_per_t(u_pred, u_ref, wm), t)
             for wm in WEIGHT_MODES}

    return {
        "case": case.short,
        "name": case.name,
        "description": case.description,
        "nu": case.nu,
        "weight_mode": weight_mode,
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


def run_robustness(rollout_fn: RolloutFn, x: np.ndarray, t: np.ndarray,
                   weight_mode: str = "low_freq") -> List[Dict]:
    """Evaluate every OOD case. Returns one result dict per case."""
    results = []
    for case in ood_cases():
        print(f"  > {case.name} (nu={case.nu:.4e}) ...")
        r = evaluate_case(rollout_fn, case, x, t, weight_mode)
        print(f"      final t={t[-1]:.0f}:  rel-L2 = {r['final_rel_l2']:.3e}   "
              f"D = {r['final_spec_dist']:.3e}   "
              f"lead({weight_mode}) = {r['early_warning_lead']}")
        results.append(r)
    return results


# =============================================================================
#  Output writers
# =============================================================================
def write_csv(results: List[Dict], path: str) -> None:
    keys = ["case", "name", "nu", "weight_mode",
            "final_rel_l2", "final_spec_dist",
            "first_cross_rel_l2", "first_cross_spec_dist", "early_warning_lead"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in keys})


def write_json(results: List[Dict], path: str, weight_mode: str) -> None:
    payload = {
        "model": "FNO",
        "module": "module2_robustness",
        "weight_mode": weight_mode,
        "t_train_end": T_TRAIN_END,
        "cases": results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def plot_results(results: List[Dict], path: str, weight_mode: str) -> None:
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 7))
    if n == 1:
        axes = np.asarray(axes).reshape(2, 1)

    fig.suptitle(f"FNO Robustness | Module 2 | Team Turingz | weight={weight_mode}",
                 fontsize=12, fontweight="bold")

    for col, r in enumerate(results):
        t = np.asarray(r["t"])
        rel = np.asarray(r["rel_l2_series"])
        spc = np.asarray(r["spec_dist_series"])

        ax0 = axes[0, col]
        ax0.semilogy(t, np.maximum(rel, 1e-12), color="C3")
        ax0.axvline(T_TRAIN_END, color="k", ls="--", lw=1, label="train / extrap split")
        c_l2 = r["first_cross_rel_l2"]
        if not np.isnan(c_l2):
            ax0.axvline(c_l2, color="C3", ls=":", lw=1.2, label=f"rel-L2 cross t={c_l2:.2f}")
        ax0.set_title(f"{r['case']}  |  relative L2 vs t")
        ax0.set_xlabel("t"); ax0.set_ylabel("relative L2")
        ax0.grid(alpha=0.3, which="both"); ax0.legend(fontsize=7)

        ax1 = axes[1, col]
        ax1.plot(t, spc, color="C0")
        ax1.axvline(T_TRAIN_END, color="k", ls="--", lw=1)
        c_sd = r["first_cross_spec_dist"]
        if not np.isnan(c_sd):
            ax1.axvline(c_sd, color="C0", ls=":", lw=1.2, label=f"D cross t={c_sd:.2f}")
        ax1.set_title(f"{r['case']}  |  spectral distance vs t")
        ax1.set_xlabel("t"); ax1.set_ylabel("weighted Fourier-amplitude distance")
        ax1.grid(alpha=0.3); ax1.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
#  Main
# =============================================================================
def main() -> None:
    default_ckpt = os.path.join(_FNO_DIR, "checkpoints", "fno_burgers.pt")

    ap = argparse.ArgumentParser(description="FNO Module 2 robustness evaluation.")
    ap.add_argument("--checkpoint", default=default_ckpt,
                    help="Trained FNO checkpoint (.pt file or checkpoint dir).")
    ap.add_argument("--weight-mode", default="low_freq", choices=WEIGHT_MODES,
                    help="Spectral-distance mode weighting for plots/CSV.")
    ap.add_argument("--out-dir",
                    default=os.path.join(_PROJECT_ROOT, "results", "robustness"),
                    help="Output directory for CSV/JSON/plot.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "fno_robustness.csv")
    json_path = os.path.join(args.out_dir, "fno_robustness_timeseries.json")
    plot_path = os.path.join(args.out_dir, "fno_robustness_plot.png")

    print("=" * 72)
    print("  FNO ROBUSTNESS EVAL  |  Module 2  |  Team Turingz")
    print("=" * 72)

    solver, x_native = load_fno_solver(args.checkpoint)
    print(f"\n  Loaded FNO   : {args.checkpoint}")

    x_grid, t = native_grid()
    # The FNO native grid must equal the Cole-Hopf reference grid, or the
    # pointwise comparison is meaningless. Guard it explicitly.
    if x_native.shape != x_grid.shape or not np.allclose(x_native, x_grid):
        raise ValueError(
            f"FNO native x grid (nx={x_native.shape[0]}) does not match the "
            f"Cole-Hopf grid (nx={x_grid.shape[0]}). The checkpoint was trained "
            "on a different grid than common/ood_spec.py defines."
        )
    print(f"  Grid         : nx={len(x_grid)}  nt={len(t)}  "
          f"t in [{t[0]:.2f}, {t[-1]:.2f}]  split @ t={T_TRAIN_END}")
    print(f"  Weighting    : {args.weight_mode}\n")

    results = run_robustness(solver.rollout, x_grid, t, args.weight_mode)

    write_csv(results, csv_path)
    write_json(results, json_path, args.weight_mode)
    plot_results(results, plot_path, args.weight_mode)

    print("\n" + "=" * 72)
    print(f"  CSV   : {csv_path}")
    print(f"  JSON  : {json_path}")
    print(f"  Plot  : {plot_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
