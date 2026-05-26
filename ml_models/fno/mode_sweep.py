"""
================================================================================
FNO FOURIER-MODE SWEEP
Team    : Turingz  (Dharmapala R.D.)
File    : ml_models/fno/mode_sweep.py

Runs a small or full sweep over truncated Fourier mode counts.

Default:
    quick smoke sweep, safe for debugging.

Full:
    python mode_sweep.py --full

Records:
- parameter count
- training wall time
- train/validation in-distribution relative L2
- validation extrapolation relative L2
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fno_solver import FNOConfig, FNOSolver
from train import evaluate, load_dataset, make_split, resolve_dataset_path


# =============================================================================
#  Paths
# =============================================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(_THIS_DIR, "sweep_results")
CSV_PATH = os.path.join(SWEEP_DIR, "fno_mode_sweep.csv")
PLOT_PATH = os.path.join(SWEEP_DIR, "fno_mode_sweep.png")


# =============================================================================
#  Sweep configuration
# =============================================================================
@dataclass
class SweepConfig:
    modes_to_test: tuple[int, ...] = (4, 8, 16)
    epochs: int = 30
    batch_size: int = 4
    hidden_channels: int = 32
    n_layers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    full: bool = False


def make_sweep_config(full: bool = False) -> SweepConfig:
    if full:
        return SweepConfig(
            modes_to_test=(4, 8, 12, 16, 20, 24, 32),
            epochs=300,
            batch_size=4,
            hidden_channels=32,
            n_layers=4,
            full=True,
        )

    return SweepConfig(
        modes_to_test=(4, 8, 16),
        epochs=30,
        batch_size=4,
        hidden_channels=32,
        n_layers=4,
        full=False,
    )


def build_cfg(sc: SweepConfig, n_modes: int) -> FNOConfig:
    cfg = FNOConfig()
    cfg.n_modes_t = n_modes
    cfg.n_modes_x = n_modes
    cfg.hidden_channels = sc.hidden_channels
    cfg.n_layers = sc.n_layers
    cfg.epochs = sc.epochs
    cfg.batch_size = sc.batch_size
    cfg.lr = sc.lr
    cfg.weight_decay = sc.weight_decay
    cfg.scheduler = "cosine"
    cfg.seed = sc.seed
    cfg.log_every = 50 if sc.full else 10
    cfg.verbose = False
    return cfg


# =============================================================================
#  Sweep
# =============================================================================
def run_sweep(sc: SweepConfig) -> list[dict]:
    data_path = resolve_dataset_path()
    dataset = load_dataset(data_path)

    U = dataset["u"]
    train_idx, val_idx = make_split(U.shape[0], n_val=1, seed=sc.seed)

    train_dataset = dict(dataset)
    train_dataset["train_idx"] = train_idx

    rows: list[dict] = []

    print("=" * 72)
    print("  FNO MODE SWEEP | Burgers 1D | Team Turingz")
    print("=" * 72)
    print(f"\n  Dataset       : {data_path}")
    print(f"  Sweep mode    : {'FULL' if sc.full else 'SMOKE'}")
    print(f"  Modes         : {sc.modes_to_test}")
    print(f"  Epochs/config : {sc.epochs}")
    print(f"  Train idx     : {train_idx}")
    print(f"  Val idx       : {val_idx}\n")

    for n_modes in sc.modes_to_test:
        cfg = build_cfg(sc, n_modes)
        solver = FNOSolver(cfg)

        print(f"  ▸ n_modes = {n_modes:2d} ...", flush=True)

        t0 = time.perf_counter()
        fit_info = solver.fit(train_dataset)
        train_wall_s = time.perf_counter() - t0

        eval_train = evaluate(solver, dataset, indices=train_idx, label="train")
        eval_val = evaluate(solver, dataset, indices=val_idx, label="val") if val_idx else {}

        eval_train.pop("per_ic_rel_l2_over_t", None)
        eval_val.pop("per_ic_rel_l2_over_t", None)

        row = {
            "n_modes": n_modes,
            "n_parameters": fit_info["n_parameters"],
            "train_wall_s": train_wall_s,
            "final_loss": fit_info["final_loss"],

            "train_in_dist_rel_l2_mean": eval_train["in_dist_rel_l2_mean"],
            "train_extrap_rel_l2_mean": eval_train["extrap_rel_l2_mean"],

            "val_in_dist_rel_l2_mean": eval_val.get("in_dist_rel_l2_mean", np.nan),
            "val_extrap_rel_l2_mean": eval_val.get("extrap_rel_l2_mean", np.nan),

            "train_in_dist_rel_l2_max": eval_train["in_dist_rel_l2_max"],
            "train_extrap_rel_l2_max": eval_train["extrap_rel_l2_max"],

            "val_in_dist_rel_l2_max": eval_val.get("in_dist_rel_l2_max", np.nan),
            "val_extrap_rel_l2_max": eval_val.get("extrap_rel_l2_max", np.nan),
        }

        rows.append(row)

        print(
            f"      params={row['n_parameters']:>8,} "
            f"train={train_wall_s:6.1f}s "
            f"val_in={row['val_in_dist_rel_l2_mean']:.3e} "
            f"val_extrap={row['val_extrap_rel_l2_mean']:.3e}"
        )

    return rows


# =============================================================================
#  Outputs
# =============================================================================
def save_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not rows:
        raise ValueError("No sweep rows to save.")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_sweep(rows: list[dict], path: str) -> None:
    modes = np.array([r["n_modes"] for r in rows])
    val_in = np.array([r["val_in_dist_rel_l2_mean"] for r in rows])
    val_extrap = np.array([r["val_extrap_rel_l2_mean"] for r in rows])
    params = np.array([r["n_parameters"] for r in rows])
    walls = np.array([r["train_wall_s"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    fig.suptitle("FNO mode-truncation sweep | Burgers 1D | Team Turingz",
                 fontsize=11, fontweight="bold")

    # 1) Accuracy vs modes
    axes[0].semilogy(modes, np.maximum(val_in, 1e-12), "o-", lw=2,
                     label="validation in-distribution")
    axes[0].semilogy(modes, np.maximum(val_extrap, 1e-12), "s--", lw=2,
                     label="validation extrapolation")
    axes[0].set_xlabel("truncated Fourier modes")
    axes[0].set_ylabel("relative L2")
    axes[0].set_title("Accuracy vs mode truncation")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=9)

    # 2) Parameter count vs modes
    axes[1].plot(modes, params / 1e3, "d-", lw=2)
    axes[1].set_xlabel("modes")
    axes[1].set_ylabel("parameters (×1e3)")
    axes[1].set_title("Model size vs modes")
    axes[1].grid(alpha=0.3)

    # 3) Cost vs extrapolation accuracy
    scatter = axes[2].scatter(walls, np.maximum(val_extrap, 1e-12), s=80, c=modes)
    for m, w, e in zip(modes, walls, val_extrap):
        axes[2].annotate(f"{m}", (w, max(e, 1e-12)), fontsize=9,
                          textcoords="offset points", xytext=(5, 5))
    axes[2].set_yscale("log")
    axes[2].set_xlabel("training wall time (s)")
    axes[2].set_ylabel("validation extrapolation relative L2")
    axes[2].set_title("Cost vs extrapolation accuracy")
    axes[2].grid(alpha=0.3, which="both")
    fig.colorbar(scatter, ax=axes[2], label="n_modes")

    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
#  Main
# =============================================================================
def main(full: bool = False) -> None:
    sc = make_sweep_config(full=full)
    rows = run_sweep(sc)

    save_csv(rows, CSV_PATH)
    plot_sweep(rows, PLOT_PATH)

    print("\n" + "=" * 72)
    print(f"  Sweep CSV  : {CSV_PATH}")
    print(f"  Sweep plot : {PLOT_PATH}")

    best_val_extrap = min(rows, key=lambda r: r["val_extrap_rel_l2_mean"])
    best_val_in = min(rows, key=lambda r: r["val_in_dist_rel_l2_mean"])

    print(f"\n  Best val in-dist : n_modes={best_val_in['n_modes']} "
          f"({best_val_in['val_in_dist_rel_l2_mean']:.3e})")
    print(f"  Best val extrap  : n_modes={best_val_extrap['n_modes']} "
          f"({best_val_extrap['val_extrap_rel_l2_mean']:.3e})")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FNO Fourier mode sweep.")
    parser.add_argument("--full", action="store_true",
                        help="Run full sweep instead of quick smoke sweep.")
    args = parser.parse_args()

    main(full=args.full)
