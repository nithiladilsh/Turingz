"""
================================================================================
FNO TRAINING SCRIPT
Team    : Turingz  (Dharmapala R.D.)
File    : ml_models/fno/train.py

Trains the Fourier Neural Operator on the Cole-Hopf reference dataset.

Training strategy:
    IC -> full training-time solution block, t <= t_train_end

Extrapolation strategy:
    block-wise autoregressive rollout beyond t_train_end

Outputs:
    ml_models/fno/checkpoints/fno_burgers.pt
    ml_models/fno/checkpoints/fno_train_log.json
    ml_models/fno/checkpoints/fno_training_diagnostics.png
================================================================================
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fno_solver import FNOConfig, FNOSolver


# =============================================================================
#  Paths
# =============================================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))

CHECKPOINT_DIR = os.path.join(_THIS_DIR, "checkpoints")
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "fno_burgers.pt")
LOG_PATH = os.path.join(CHECKPOINT_DIR, "fno_train_log.json")
PLOT_PATH = os.path.join(CHECKPOINT_DIR, "fno_training_diagnostics.png")


def resolve_dataset_path() -> str:
    """Find Cole-Hopf dataset from accepted project locations."""
    candidates = [
        # Recommended final location
        os.path.join(_PROJECT_ROOT, "data", "colehopf", "burgers_1d_cole_hopf.pt"),

        # Older/simple generator location
        os.path.join(_PROJECT_ROOT, "data", "burgers_1d_cole_hopf.pt"),

        # If accidentally saved under numerical solver folder
        os.path.join(_PROJECT_ROOT, "numerical_solvers", "colehopf", "burgers_1d_cole_hopf.pt"),
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "Cole-Hopf dataset not found.\n"
        "Expected one of:\n"
        + "\n".join(f"  - {p}" for p in candidates)
        + "\n\nRun numerical_solvers/colehopf/colehopf.py first, or move the dataset to:\n"
        f"  {candidates[0]}"
    )


DATA_PATH = resolve_dataset_path()


# =============================================================================
#  Configuration
# =============================================================================
def default_config(smoke: bool = False) -> FNOConfig:
    cfg = FNOConfig()

    cfg.n_modes_t = 16
    cfg.n_modes_x = 16
    cfg.hidden_channels = 32
    cfg.n_layers = 4

    if smoke:
        cfg.epochs = 20
        cfg.batch_size = 2
        cfg.log_every = 5
        cfg.verbose = True
    else:
        cfg.epochs = 500
        cfg.batch_size = 4
        cfg.log_every = 25
        cfg.verbose = True

    cfg.lr = 1e-3
    cfg.weight_decay = 1e-4
    cfg.scheduler = "cosine"
    cfg.seed = 0

    return cfg


# =============================================================================
#  Dataset helpers
# =============================================================================
def torch_load_any(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_dataset(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Cole-Hopf dataset not found at {path}. "
            "Run numerical_solvers/colehopf/colehopf.py first."
        )

    data = torch_load_any(path)

    out: dict = {}
    for k, v in data.items():
        out[k] = v.numpy() if isinstance(v, torch.Tensor) else v

    # Compatibility: some generators may save "U" instead of "u"
    if "u" not in out and "U" in out:
        out["u"] = out["U"]

    required = ["u", "x", "t", "t_train_end"]
    missing = [k for k in required if k not in out]
    if missing:
        raise KeyError(f"Dataset missing required keys: {missing}")

    return out


def make_split(N: int,
               n_val: int = 1,
               seed: int = 0) -> tuple[list[int], list[int]]:
    """Create trajectory-level train/validation split."""
    if N < 2:
        return list(range(N)), []

    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)

    n_val = min(max(n_val, 1), N - 1)
    val_idx = idx[:n_val].tolist()
    train_idx = idx[n_val:].tolist()

    return train_idx, val_idx


# =============================================================================
#  Metrics
# =============================================================================
def relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    num = np.linalg.norm(pred - ref)
    den = np.linalg.norm(ref) + 1e-12
    return float(num / den)


def per_time_relative_l2(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Row-wise relative L2 over time, shape (nt,)."""
    num = np.linalg.norm(pred - ref, axis=1)
    den = np.linalg.norm(ref, axis=1) + 1e-12
    return num / den


# =============================================================================
#  Evaluation
# =============================================================================
def evaluate(solver: FNOSolver,
             dataset: dict,
             indices: Optional[list[int]] = None,
             label: str = "all") -> dict:
    """Compute in-distribution and extrapolation relative-L2 error."""
    U = dataset["u"]
    x = dataset["x"]
    t = dataset["t"]
    t_train_end = float(dataset["t_train_end"])

    if indices is None:
        indices = list(range(U.shape[0]))

    train_mask = (t <= t_train_end + 1e-9)
    extra_mask = ~train_mask

    in_dist: list[float] = []
    extra: list[float] = []
    per_ic_curves: list[np.ndarray] = []

    for i in indices:
        ic = U[i, 0, :]
        pred = solver.rollout(ic, x, t)
        ref = U[i]

        in_dist.append(relative_l2(pred[train_mask], ref[train_mask]))

        if extra_mask.any():
            extra.append(relative_l2(pred[extra_mask], ref[extra_mask]))

        per_ic_curves.append(per_time_relative_l2(pred, ref))

    in_arr = np.asarray(in_dist, dtype=np.float64)
    extra_arr = np.asarray(extra, dtype=np.float64) if extra else np.asarray([np.nan])
    curves = np.stack(per_ic_curves, axis=0) if per_ic_curves else np.empty((0, len(t)))

    return {
        "label": label,
        "indices": indices,
        "n_cases": len(indices),
        "in_dist_rel_l2_mean": float(np.nanmean(in_arr)),
        "in_dist_rel_l2_max": float(np.nanmax(in_arr)),
        "extrap_rel_l2_mean": float(np.nanmean(extra_arr)),
        "extrap_rel_l2_max": float(np.nanmax(extra_arr)),
        "per_ic_in_dist": in_arr.tolist(),
        "per_ic_extrap": extra_arr.tolist(),
        "per_ic_rel_l2_over_t": curves,
    }


# =============================================================================
#  Diagnostics plot
# =============================================================================
def plot_diagnostics(history: dict,
                     dataset: dict,
                     solver: FNOSolver,
                     err_over_t: np.ndarray,
                     save_path: str) -> None:
    U = dataset["u"]
    x = dataset["x"]
    t = dataset["t"]
    t_train_end = float(dataset["t_train_end"])

    pred0 = solver.rollout(U[0, 0, :], x, t)
    ref0 = U[0]
    err0 = np.abs(pred0 - ref0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    fig.suptitle(f"FNO training diagnostics | {solver.name}",
                 fontsize=11, fontweight="bold")

    # 1) Loss curve
    axes[0].semilogy(history["epoch"], np.maximum(history["loss"], 1e-12))
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("relative L2 loss")
    axes[0].set_title("Training loss")
    axes[0].grid(alpha=0.3)

    # 2) Sample 0 absolute error heatmap
    im = axes[1].pcolormesh(x, t, err0, cmap="magma", shading="auto")
    axes[1].axhline(t_train_end, color="cyan", ls="--", lw=1,
                    label="train / extrap split")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    axes[1].set_title("Sample 0 | |u_FNO - u_ref|")
    fig.colorbar(im, ax=axes[1])
    axes[1].legend(fontsize=8, loc="lower right")

    # 3) Per-IC relative L2 over time
    for i in range(err_over_t.shape[0]):
        series = np.maximum(err_over_t[i], 1e-12)
        axes[2].semilogy(t, series, lw=1.0, alpha=0.7,
                         label=f"IC {i}" if i < 3 else None)

    axes[2].axvline(t_train_end, color="k", ls="--", lw=1)
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("relative L2")
    axes[2].set_title("Rollout error vs time")
    axes[2].grid(alpha=0.3, which="both")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
#  JSON helper
# =============================================================================
def clean_for_json(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: clean_for_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_for_json(v) for v in o]
    return o


# =============================================================================
#  Main
# =============================================================================
def main(smoke: bool = False) -> None:
    print("=" * 72)
    print("  FNO TRAINING | Burgers 1D | Team Turingz")
    print("=" * 72)

    cfg = default_config(smoke=smoke)
    # Unified seed: read the canonical SEED so every model matches.
    from common.canonical_split import SEED as _CANON_SEED
    cfg.seed = _CANON_SEED

    print(f"\n  Architecture: modes=({cfg.n_modes_t},{cfg.n_modes_x}) "
          f"width={cfg.hidden_channels} layers={cfg.n_layers}")
    print(f"  Optimiser  : Adam(lr={cfg.lr}, wd={cfg.weight_decay}) "
          f"{cfg.scheduler} schedule epochs={cfg.epochs}")
    print(f"  Device     : {cfg.device}")
    print(f"  Dataset    : {DATA_PATH}\n")

    dataset = load_dataset(DATA_PATH)
    U, x, t = dataset["u"], dataset["x"], dataset["t"]

    print(f"  Loaded dataset: U={U.shape} x={x.shape} t={t.shape} "
          f"t_train_end={dataset['t_train_end']}")

    # Canonical split — single source of truth shared with PINN and DeepONet.
    # Operators train on the canonical TRAIN_IDX and are tested on the held-out
    # (unseen) TEST_IDX. This replaces the previous random per-seed split.
    from common.canonical_split import operator_train_idx, operator_test_idx
    train_idx = operator_train_idx()
    test_idx = operator_test_idx()
    train_dataset = dict(dataset)
    train_dataset["train_idx"] = train_idx

    print(f"  Canonical split: train_idx={train_idx}  held-out test_idx(ood)={test_idx}\n")

    solver = FNOSolver(cfg)
    print(f"  Training {solver.name}...\n")
    fit_info = solver.fit(train_dataset)

    print("\n  ── training complete ──")
    print(f"    final loss : {fit_info['final_loss']:.4e}")
    print(f"    wall time  : {fit_info['wall_time_s']:.1f}s")
    print(f"    parameters : {fit_info['n_parameters']:,}")

    print("\n  Evaluating train / val / all trajectories...")
    t0 = time.perf_counter()

    eval_train = evaluate(solver, dataset, indices=train_idx, label="train(in_dist)")
    eval_test = evaluate(solver, dataset, indices=test_idx, label="test(ood)") if test_idx else {}
    eval_all = evaluate(solver, dataset, indices=None, label="all")

    eval_time = time.perf_counter() - t0

    err_over_t = eval_all.pop("per_ic_rel_l2_over_t")
    eval_train.pop("per_ic_rel_l2_over_t", None)
    eval_test.pop("per_ic_rel_l2_over_t", None)

    print(f"    train(in_dist) rel-L2 : mean={eval_train['in_dist_rel_l2_mean']:.4e} "
          f"max={eval_train['in_dist_rel_l2_max']:.4e}")
    if eval_test:
        print(f"    test(ood) in-dist rel-L2 : mean={eval_test['in_dist_rel_l2_mean']:.4e} "
              f"max={eval_test['in_dist_rel_l2_max']:.4e}")
        print(f"    test(ood) extrap  rel-L2 : mean={eval_test['extrap_rel_l2_mean']:.4e} "
              f"max={eval_test['extrap_rel_l2_max']:.4e}")

    print(f"    all   extrap rel-L2  : mean={eval_all['extrap_rel_l2_mean']:.4e} "
          f"max={eval_all['extrap_rel_l2_max']:.4e}")
    print(f"    rollout evaluation time: {eval_time:.1f}s")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    solver.save(MODEL_PATH)

    log = {
        "dataset_path": DATA_PATH,
        "config_version": "v1_fno_ic_to_block",
        "training_strategy": "IC_to_full_training_block",
        "extrapolation_strategy": "blockwise_autoregressive_rollout",
        "train_idx": train_idx,
        "test_idx": test_idx,
        "split_source": "common.canonical_split",
        "config": cfg.__dict__,
        "fit_info": {k: v for k, v in fit_info.items() if k != "history"},
        "evaluation": {
            "train_in_dist": eval_train,
            "test_ood": eval_test,
            "all": eval_all,
        },
        "history": fit_info["history"],
        "smoke_run": smoke,
    }

    with open(LOG_PATH, "w") as f:
        json.dump(clean_for_json(log), f, indent=2)

    plot_diagnostics(fit_info["history"], dataset, solver, err_over_t, PLOT_PATH)

    print("\n" + "=" * 72)
    print(f"  Saved checkpoint : {MODEL_PATH}")
   