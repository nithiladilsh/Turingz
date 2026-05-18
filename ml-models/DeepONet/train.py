"""
================================================================================
DEEPONET TRAINING DRIVER
Team    : Turingz
File    : ml_models/DeepONet/train.py

Loads the Cole-Hopf reference dataset (.pt), trains a DeepONetSolver,
saves the trained checkpoint and a JSON of training metrics.

Usage (run from project root)
-----------------------------
    python ml_models/DeepONet/train.py \\
        --data data/burgers_1d_cole_hopf.pt \\
        --n-sensors 128 \\
        --iterations 30000 \\
        --out ml_models/DeepONet/checkpoints/m128
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
import json

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch

# ── Project ───────────────────────────────────────────────────────────────────
from deeponet_solver import DeepONetSolver


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE SETUP
# ─────────────────────────────────────────────────────────────────────────────
def setup_device(requested: str = "auto") -> str:
    """Select GPU/CPU for DeepXDE+PyTorch and print a startup banner.

    DeepXDE with the PyTorch backend picks the device from PyTorch's default,
    so calling torch.set_default_device() and torch.cuda.set_device() here
    is enough — no further plumbing is needed inside the solver.
    """
    if requested == "cpu":
        device = "cpu"
    elif requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False.")
        device = "cuda"
    else:  # "auto"
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.cuda.set_device(0)
        torch.set_default_device("cuda")
        name = torch.cuda.get_device_name(0)
        cuda_v = torch.version.cuda
        mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  device      : GPU 0 — {name}  ({mem:.1f} GiB, CUDA {cuda_v})")
    else:
        torch.set_default_device("cpu")
        print(f"  device      : CPU  (torch.cuda.is_available() = {torch.cuda.is_available()})")
    return device


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(pt_path: str) -> dict:
    """Load the Cole-Hopf .pt file produced by the dataset generator.

    Returns a numpy-only dict so the solver does not depend on the
    torch tensor types.
    """
    d = torch.load(pt_path, weights_only=False, map_location="cpu")

    def _np(arr):
        return arr.detach().cpu().numpy() if torch.is_tensor(arr) else np.asarray(arr)

    return {
        "u"          : _np(d["u"]),
        "ICs"        : _np(d["ICs"]),
        "x"          : _np(d["x"]),
        "t"          : _np(d["t"]),
        "nu"         : float(d["nu"]),
        "L"          : float(d["L"]),
        "t_train_end": float(d["t_train_end"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train DeepONet on the Burgers reference.")
    p.add_argument("--data",       required=True, help="Path to burgers_1d_cole_hopf.pt")
    p.add_argument("--out",        required=True, help="Output directory for the checkpoint")
    p.add_argument("--n-sensors",  type=int,   default=128)
    p.add_argument("--latent-dim", type=int,   default=128)
    p.add_argument("--width",      type=int,   default=128, help="Hidden width for both branch and trunk")
    p.add_argument("--depth",      type=int,   default=3,   help="Hidden layers for both branch and trunk")
    p.add_argument("--activation", type=str,   default="relu", choices=["relu", "tanh", "gelu", "silu"])
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--iterations", type=int,   default=30_000)
    p.add_argument("--batch-size", type=int,   default=None,
                   help="Mini-batch size (over IC samples).  Default = use all ICs per step.")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--device",     type=str,   default="auto",
                   choices=["auto", "cuda", "cpu"],
                   help="Computation device.  'auto' uses GPU if available.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  DeepONet training — Burgers (Cole-Hopf reference)")
    print("=" * 72)

    # ── Device ────────────────────────────────────────────────────────────
    setup_device(args.device)

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"\nLoading dataset from: {args.data}")
    dataset = load_dataset(args.data)
    N, nt, nx = dataset["u"].shape
    print(f"  u           : {dataset['u'].shape}   (N, nt, nx)")
    print(f"  ICs         : {dataset['ICs'].shape}")
    print(f"  x           : [{dataset['x'].min():.4f}, {dataset['x'].max():.4f}]  nx={nx}")
    print(f"  t           : [{dataset['t'].min():.3f}, {dataset['t'].max():.3f}]  nt={nt}")
    print(f"  ν           : {dataset['nu']:.6f}")
    print(f"  t_train_end : {dataset['t_train_end']}")
    n_train_t = int((dataset["t"] <= dataset["t_train_end"]).sum())
    print(f"  → {n_train_t} training snapshots,  {nt - n_train_t} extrapolation snapshots")

    # ── Build solver ──────────────────────────────────────────────────────
    solver = DeepONetSolver(
        n_sensors    = args.n_sensors,
        latent_dim   = args.latent_dim,
        branch_width = args.width,
        trunk_width  = args.width,
        branch_depth = args.depth,
        trunk_depth  = args.depth,
        activation   = args.activation,
        lr           = args.lr,
        iterations   = args.iterations,
        batch_size   = args.batch_size,
        seed         = args.seed,
    )
    print(f"\nSolver: {solver.name}")
    print(f"  branch layers : [{args.n_sensors}] + [{args.width}]×{args.depth} + [{args.latent_dim}]")
    print(f"  trunk  layers : [2] + [{args.width}]×{args.depth} + [{args.latent_dim}]")
    print(f"  activation    : {args.activation}")
    print(f"  optimiser     : Adam(lr={args.lr})")
    print(f"  iterations    : {args.iterations}")

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}\nTraining ...\n{'─' * 72}")
    info = solver.fit(dataset)

    # ── Report ───────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}\nResults\n{'─' * 72}")
    print(f"  wall time             : {info['wall_time_s']:.1f}s")
    print(f"  final train loss      : {info['final_train_loss']:.3e}")
    print(f"  final val loss        : {info['final_val_loss']:.3e}")
    print(f"  in-dist  rel L2 mean  : {info['in_dist_rel_l2_mean']:.4f}")
    if "extrapolation_rel_l2_mean" in info:
        print(f"  extrap   rel L2 mean  : {info['extrapolation_rel_l2_mean']:.4f}")
        print(f"  extrap   per-IC       : "
              + ", ".join(f"{v:.3f}" for v in info["extrapolation_rel_l2_per_ic"]))

    # ── Save ──────────────────────────────────────────────────────────────
    solver.save(str(out_path))
    with open(out_path / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nCheckpoint saved to: {out_path}")
    print(f"  model.pt          weights")
    print(f"  meta.npz          sensor metadata")
    print(f"  config.json       hyper-parameters")
    print(f"  training_info.json  training metrics")
    print("=" * 72)


if __name__ == "__main__":
    main()