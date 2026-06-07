"""
================================================================================
DEEPONET SOLVER
Team    : Turingz   (Module 3 - 6.5 DeepONet Implementation)
File    : ml_models/DeepONet/deeponet_solver.py

Deep Operator Network for the 1D viscous Burgers equation, implemented
with DeepXDE on a PyTorch backend.

Architecture
------------
        u(x, t)  ≈  Σ_k  b_k(u0_sensors) · τ_k(x, t)  +  bias

    Branch network   :  u0 sampled at m fixed sensor locations  →  R^p
    Trunk  network   :  (x, t)                                  →  R^p
    Combination      :  inner product + scalar learned bias

Both subnetworks are fully-connected MLPs with multiple hidden layers
and a nonlinear activation function (ReLU by default).  The latent
embedding dimension p is shared so the branch and trunk outputs can be
combined through the inner-product head.

Training data
-------------
Tuples of (initial condition, x, t, u_truth) drawn from the reference
dataset on the training time window t ≤ t_train_end.  We use the
DeepXDE TripleCartesianProd container, which stores each IC once and
shares the trunk grid across ICs — this is mathematically identical to
flat (IC, x, t, u) tuples but avoids replicating the IC for every
query point.  Adam mini-batches over the IC axis at each iteration.

Loss
----
Relative L2 error between predicted and reference solution values.
================================================================================
"""

# ── Project-root import path setup ────────────────────────────────────────────
# Lets `from abstract_solver import AbstractSolver` resolve to the project-root
# file regardless of where the script is launched from.
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── DeepXDE backend selection (must precede the deepxde import) ───────────────
import os
os.environ.setdefault("DDE_BACKEND", "pytorch")

# ── Standard library ──────────────────────────────────────────────────────────
import json
import time
from typing import Dict, Any

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch

import deepxde as dde

# ── Project ───────────────────────────────────────────────────────────────────
from abstract_solver import AbstractSolver


# ─────────────────────────────────────────────────────────────────────────────
# DEEPONET SOLVER
# ─────────────────────────────────────────────────────────────────────────────
class DeepONetSolver(AbstractSolver):
    """DeepONet for 1D viscous Burgers, conforming to AbstractSolver.

    Parameters
    ----------
    n_sensors    : number of fixed sensor locations on x ∈ [-1, 1).
                   The branch input vector has length n_sensors.
                   See sensor_sweep.py for choosing this value.
    latent_dim   : shared output dimension of branch and trunk (the p
                   in the architecture diagram above).
    branch_width : hidden-layer width for the branch MLP.
    trunk_width  : hidden-layer width for the trunk MLP.
    branch_depth : number of hidden layers in the branch.
    trunk_depth  : number of hidden layers in the trunk.
    activation   : "relu" or "tanh".
    lr           : Adam learning rate.
    iterations   : number of Adam iterations.
    batch_size   : mini-batch size in IC samples (None = all ICs per step).
    val_fraction : fraction of trunk points held out for in-distribution
                   validation loss reporting (0 ≤ x < 1).
    seed         : random seed.
    """

    # ── Construction ────────────────────────────────────────────────────────
    def __init__(
        self,
        n_sensors:    int   = 128,
        latent_dim:   int   = 128,
        branch_width: int   = 128,
        trunk_width:  int   = 128,
        branch_depth: int   = 3,
        trunk_depth:  int   = 3,
        activation:   str   = "relu",
        lr:           float = 1e-3,
        iterations:   int   = 30_000,
        batch_size:   int   = None,
        val_fraction: float = 0.1,
        seed:         int   = 42,
    ):
        self.n_sensors    = int(n_sensors)
        self.latent_dim   = int(latent_dim)
        self.branch_width = int(branch_width)
        self.trunk_width  = int(trunk_width)
        self.branch_depth = int(branch_depth)
        self.trunk_depth  = int(trunk_depth)
        self.activation   = str(activation)
        self.lr           = float(lr)
        self.iterations   = int(iterations)
        self.batch_size   = batch_size
        self.val_fraction = float(val_fraction)
        self.seed         = int(seed)

        # Set after fit/load
        self._model       = None              # dde.Model
        self._sensor_idx  = None              # (m,)  indices into x
        self._sensor_x    = None              # (m,)  sensor positions
        self._x_full      = None              # (nx,) reference grid
        self._losshistory = None

    @property
    def name(self) -> str:
        return f"DeepONet(m={self.n_sensors})"

    # ── Architecture ────────────────────────────────────────────────────────
    def _build_network(self) -> dde.nn.NN:
        """Build branch/trunk MLPs and the inner-product head."""
        branch_layers = ([self.n_sensors] +
                         [self.branch_width] * self.branch_depth +
                         [self.latent_dim])
        trunk_layers  = ([2] +
                         [self.trunk_width] * self.trunk_depth +
                         [self.latent_dim])
        return dde.nn.DeepONetCartesianProd(
            layer_sizes_branch = branch_layers,
            layer_sizes_trunk  = trunk_layers,
            activation         = self.activation,
            kernel_initializer = "Glorot normal",
        )

    # ── Training ────────────────────────────────────────────────────────────
    def fit(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # ── Unpack (accept either numpy arrays or torch tensors) ──────────
        def _np(arr):
            return arr.numpy() if hasattr(arr, "numpy") else np.asarray(arr)

        U           = _np(dataset["u"])                  # (N, nt, nx)
        ICs         = _np(dataset["ICs"])                # (N, nx)
        x           = _np(dataset["x"])                  # (nx,)
        t           = _np(dataset["t"])                  # (nt,)
        t_train_end = float(dataset.get("t_train_end", 1.0))

        # ── Optional IC-level training subset (canonical split) ───────────
        # If dataset["train_idx"] is provided, train only on those ICs so the
        # held-out ICs stay genuinely unseen. Default (None) = train on all,
        # preserving the original behaviour.
        train_idx = dataset.get("train_idx", None)
        if train_idx is not None:
            train_idx = np.asarray(train_idx, dtype=int)
            if train_idx.size == 0:
                raise ValueError("dataset['train_idx'] is empty.")
            U   = U[train_idx]
            ICs = ICs[train_idx]

        N, nt, nx = U.shape
        if ICs.shape != (N, nx) or x.shape != (nx,) or t.shape != (nt,):
            raise ValueError(
                f"Inconsistent dataset shapes: "
                f"u={U.shape}, ICs={ICs.shape}, x={x.shape}, t={t.shape}"
            )

        # ── Sensor locations: uniform subsample of the reference grid ─────
        self._sensor_idx = np.linspace(0, nx - 1, self.n_sensors).astype(int)
        self._sensor_x   = x[self._sensor_idx]
        self._x_full     = x.copy()

        branch_input = ICs[:, self._sensor_idx].astype(np.float32)   # (N, m)

        # ── Trunk grid on the training time window ────────────────────────
        train_mask = t <= t_train_end
        t_train    = t[train_mask]                                   # (nt_tr,)
        U_train    = U[:, train_mask, :]                             # (N, nt_tr, nx)

        Xg, Tg     = np.meshgrid(x, t_train, indexing="xy")          # (nt_tr, nx)
        trunk_full = np.stack([Xg.ravel(), Tg.ravel()], axis=1).astype(np.float32)
        y_full     = U_train.reshape(N, -1).astype(np.float32)       # (N, nt_tr*nx)

        # ── Train / in-distribution validation split over trunk points ────
        rng        = np.random.default_rng(self.seed)
        perm       = rng.permutation(trunk_full.shape[0])
        n_val      = max(1, int(self.val_fraction * trunk_full.shape[0]))
        val_idx    = perm[:n_val]
        train_idx  = perm[n_val:]

        trunk_tr   = trunk_full[train_idx]
        trunk_val  = trunk_full[val_idx]
        y_tr       = y_full[:, train_idx]
        y_val      = y_full[:, val_idx]

        # ── Build DeepXDE data + model ────────────────────────────────────
        data = dde.data.TripleCartesianProd(
            X_train = (branch_input, trunk_tr),
            y_train = y_tr,
            X_test  = (branch_input, trunk_val),
            y_test  = y_val,
        )
        net   = self._build_network()
        model = dde.Model(data, net)
        model.compile("adam", lr=self.lr, metrics=["mean l2 relative error"])

        # ── Adam training ────────────────────────────────────────────────
        t0 = time.perf_counter()
        losshistory, _ = model.train(
            iterations    = self.iterations,
            batch_size    = self.batch_size,
            display_every = max(1, self.iterations // 30),
        )
        wall_time = time.perf_counter() - t0

        self._model       = model
        self._losshistory = losshistory

        # ── Report ──────────────────────────────────────────────────────────
        info: Dict[str, Any] = {
            "solver"          : self.name,
            "n_sensors"       : self.n_sensors,
            "latent_dim"      : self.latent_dim,
            "iterations"      : self.iterations,
            "wall_time_s"     : wall_time,
            "final_train_loss": float(np.array(losshistory.loss_train[-1]).sum()),
            "final_val_loss"  : float(np.array(losshistory.loss_test[-1]).sum()),
        }

        # In-distribution per-IC relative L2 (on full training grid)
        pred_train = model.predict((branch_input, trunk_full))
        err_tr     = pred_train - y_full
        rel_l2_in  = np.linalg.norm(err_tr, axis=1) / (np.linalg.norm(y_full, axis=1) + 1e-12)
        info["in_dist_rel_l2_per_ic"] = rel_l2_in.tolist()
        info["in_dist_rel_l2_mean"]   = float(np.mean(rel_l2_in))

        # Extrapolation evaluation (t > t_train_end) — the central study
        ext_mask = t > t_train_end
        if ext_mask.any():
            t_ext   = t[ext_mask]
            Xe, Te  = np.meshgrid(x, t_ext, indexing="xy")
            trunk_e = np.stack([Xe.ravel(), Te.ravel()], axis=1).astype(np.float32)
            y_ext   = U[:, ext_mask, :].reshape(N, -1).astype(np.float32)

            pred_e  = model.predict((branch_input, trunk_e))
            err_e   = pred_e - y_ext
            rel_l2_ext = np.linalg.norm(err_e, axis=1) / (np.linalg.norm(y_ext, axis=1) + 1e-12)
            info["extrapolation_rel_l2_per_ic"] = rel_l2_ext.tolist()
            info["extrapolation_rel_l2_mean"]   = float(np.mean(rel_l2_ext))

        return info

    # ── Inference ───────────────────────────────────────────────────────────
    def predict(self, ic: np.ndarray, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained.  Call fit() or load() first.")
        ic    = np.asarray(ic,  dtype=np.float32).ravel()
        x_arr = np.asarray(x,   dtype=np.float32).ravel()
        t_arr = np.asarray(t,   dtype=np.float32).ravel()
        if x_arr.shape != t_arr.shape:
            raise ValueError("predict(): x and t must have the same length.")

        branch_in = ic[self._sensor_idx][None, :]            # (1, m)
        trunk_in  = np.stack([x_arr, t_arr], axis=1)         # (M, 2)
        pred      = self._model.predict((branch_in, trunk_in))   # (1, M)
        return pred.ravel()

    def rollout(self, ic: np.ndarray, x_grid: np.ndarray,
                t_grid: np.ndarray) -> np.ndarray:
        """Override the default rollout for a single batched forward pass."""
        if self._model is None:
            raise RuntimeError("Model not trained.  Call fit() or load() first.")
        ic        = np.asarray(ic, dtype=np.float32).ravel()
        branch_in = ic[self._sensor_idx][None, :]                       # (1, m)

        Xg, Tg    = np.meshgrid(x_grid, t_grid, indexing="xy")          # (nt, nx)
        trunk_in  = np.stack([Xg.ravel(), Tg.ravel()], axis=1).astype(np.float32)
        pred      = self._model.predict((branch_in, trunk_in))          # (1, nt*nx)
        return pred.reshape(len(t_grid), len(x_grid))

    # ── Persistence ─────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """Save state_dict + sensor metadata + hyper-parameters."""
        if self._model is None:
            raise RuntimeError("Nothing to save: model not trained.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        # Underlying torch network weights
        torch.save(self._model.net.state_dict(), out / "model.pt")

        # Sensor metadata
        np.savez(
            out / "meta.npz",
            sensor_idx = self._sensor_idx,
            sensor_x   = self._sensor_x,
            x_full     = self._x_full,
        )

        # Hyper-parameters
        with open(out / "config.json", "w") as f:
            json.dump(self._config_dict(), f, indent=2)

    def load(self, path: str) -> None:
        inp = Path(path)
        with open(inp / "config.json") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            setattr(self, k, v)

        meta = np.load(inp / "meta.npz")
        self._sensor_idx = meta["sensor_idx"]
        self._sensor_x   = meta["sensor_x"]
        self._x_full     = meta["x_full"]

        # Rebuild the network and wire it through dde.Model for predict().
        # The Model needs a data object, but it is only used to look up
        # auxiliary metrics, so a one-row dummy is fine.
        net   = self._build_network()
        dummy_branch = np.zeros((1, self.n_sensors), dtype=np.float32)
        dummy_trunk  = np.zeros((1, 2),               dtype=np.float32)
        dummy_y      = np.zeros((1, 1),               dtype=np.float32)
        data  = dde.data.TripleCartesianProd(
            X_train = (dummy_branch, dummy_trunk),
            y_train = dummy_y,
            X_test  = (dummy_branch, dummy_trunk),
            y_test  = dummy_y,
        )
        model = dde.Model(data, net)
        model.compile("adam", lr=self.lr)
        net.load_state_dict(torch.load(inp / "model.pt", weights_only=False))
        self._model = model

    def _config_dict(self) -> Dict[str, Any]:
        return {
            "n_sensors":    self.n_sensors,
            "latent_dim":   self.latent_dim,
            "branch_width": self.branch_width,
            "trunk_width":  self.trunk_width,
            "branch_depth": self.branch_depth,
            "trunk_depth":  self.trunk_depth,
            "activation":   self.activation,
            "lr":           self.lr,
            "iterations":   self.iterations,
            "batch_size":   self.batch_size,
            "val_fraction": self.val_fraction,
            "seed":         self.seed,
        }