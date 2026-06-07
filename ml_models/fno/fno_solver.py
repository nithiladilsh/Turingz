"""
================================================================================
FOURIER NEURAL OPERATOR SOLVER
Team    : Turingz  (Dharmapala R.D. - Module 2 / FNO implementation)
File    : ml_models/fno/fno_solver.py

Final FNO version:
- Trains FNO as an IC -> full training-window solution-block operator.
- Input channels: initial condition tiled over time, x-coordinate, relative t-coordinate.
- Output: u(x,t) for t <= t_train_end.
- Long-time extrapolation: block-wise rollout. The last predicted slice of a block
  becomes the initial condition for the next block.
- Conforms to the project-wide AbstractSolver interface.

Expected dataset format:
dataset["u"]            : (N, nt, nx)
dataset["x"]            : (nx,)
dataset["t"]            : (nt,)
dataset["t_train_end"]  : float
dataset["nu"]           : optional float
dataset["train_idx"]    : optional list/array of training trajectory indices
================================================================================
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ── project-root import of the shared interface ──────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from abstract_solver import AbstractSolver  # noqa: E402


# ── neuraloperator compatibility ─────────────────────────────────────────────
try:
    from neuralop.models import FNO
except Exception:  # pragma: no cover
    from neuralop.models import TFNO as FNO

try:
    from neuralop.losses import LpLoss
except Exception:  # pragma: no cover
    from neuralop import LpLoss


# =============================================================================
#  Configuration
# =============================================================================
@dataclass
class FNOConfig:
    """FNO hyperparameters and training settings."""

    # Architecture
    n_modes_t: int = 16
    n_modes_x: int = 16
    hidden_channels: int = 32
    n_layers: int = 4
    in_channels: int = 3       # (IC_tiled, x_coord, t_rel_coord)
    out_channels: int = 1      # u(x,t)
    lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 2

    # Optimisation
    epochs: int = 500
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # "cosine" or "step"
    step_gamma: float = 0.5
    step_size: int = 100

    # Misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    log_every: int = 25
    verbose: bool = True


# =============================================================================
#  Helper functions
# =============================================================================
def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _wrap_x(x: np.ndarray,
            x_min: float = -1.0,
            x_max: float = 1.0) -> np.ndarray:
    """Wrap x into [x_min, x_max) for periodic interpolation."""
    x = np.asarray(x, dtype=np.float64)
    L = x_max - x_min
    return ((x - x_min) % L) + x_min


def _build_input_tensor(ic_batch: torch.Tensor,
                        x_grid: torch.Tensor,
                        t_rel: torch.Tensor) -> torch.Tensor:
    """Build FNO input tensor.

    Parameters
    ----------
    ic_batch : (B, nx)
        Initial condition/current state for each sample in the batch.
    x_grid : (nx,)
        Native spatial grid.
    t_rel : (nt_block,)
        Relative time grid within one prediction block.

    Returns
    -------
    Tensor of shape (B, 3, nt_block, nx)
    channel 0: IC tiled over time
    channel 1: x coordinate
    channel 2: relative t coordinate
    """
    B = ic_batch.shape[0]
    nx = x_grid.shape[0]
    nt = t_rel.shape[0]

    ic_ch = ic_batch[:, None, None, :].expand(B, 1, nt, nx)
    x_ch = x_grid[None, None, None, :].expand(B, 1, nt, nx)
    t_ch = t_rel[None, None, :, None].expand(B, 1, nt, nx)

    return torch.cat([ic_ch, x_ch, t_ch], dim=1).contiguous()


def _torch_load(path: str, map_location: Any):
    """torch.load wrapper compatible with older and newer PyTorch."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # older PyTorch does not support weights_only
        return torch.load(path, map_location=map_location)


# =============================================================================
#  FNOSolver
# =============================================================================
class FNOSolver(AbstractSolver):
    """FNO wrapper conforming to AbstractSolver.

    Learns:
        G: u(x,0) -> u(x,t),  t in [0, t_train_end]

    Extrapolation:
        block-wise rollout beyond the training window.
    """

    def __init__(self, cfg: Optional[FNOConfig] = None):
        self.cfg = cfg or FNOConfig()
        _set_seed(self.cfg.seed)

        self.model: Optional[nn.Module] = None
        self.device = torch.device(self.cfg.device)

        # Populated by fit/load
        self._x_grid: Optional[torch.Tensor] = None
        self._t_train_rel: Optional[torch.Tensor] = None
        self._t_train_end: Optional[float] = None
        self._u_mean: float = 0.0
        self._u_std: float = 1.0
        self._nu: Optional[float] = None

    @property
    def name(self) -> str:
        return (
            f"FNO(modes=({self.cfg.n_modes_t},{self.cfg.n_modes_x}),"
            f"w={self.cfg.hidden_channels},L={self.cfg.n_layers})"
        )

    def _build_model(self) -> nn.Module:
        """Build neuraloperator FNO, handling small API differences."""
        kwargs = dict(
            n_modes=(self.cfg.n_modes_t, self.cfg.n_modes_x),
            in_channels=self.cfg.in_channels,
            out_channels=self.cfg.out_channels,
            hidden_channels=self.cfg.hidden_channels,
            n_layers=self.cfg.n_layers,
            lifting_channel_ratio=self.cfg.lifting_channel_ratio,
            projection_channel_ratio=self.cfg.projection_channel_ratio,
            non_linearity=torch.nn.functional.gelu,
        )

        try:
            model = FNO(**kwargs)
        except TypeError:
            kwargs.pop("lifting_channel_ratio", None)
            kwargs.pop("projection_channel_ratio", None)
            try:
                model = FNO(**kwargs)
            except TypeError:
                kwargs.pop("non_linearity", None)
                model = FNO(**kwargs)

        return model.to(self.device)

    # -------------------------------------------------------------------------
    # AbstractSolver.fit
    # -------------------------------------------------------------------------
    def fit(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Train FNO on selected in-distribution trajectories.

        If dataset["train_idx"] exists, only those trajectory indices are used.
        Otherwise, all trajectories are used.
        """
        cfg = self.cfg

        U = torch.as_tensor(dataset["u"], dtype=torch.float32)
        x = torch.as_tensor(dataset["x"], dtype=torch.float32)
        t = torch.as_tensor(dataset["t"], dtype=torch.float32)

        # Basic dataset checks
        assert U.ndim == 3, f"U must have shape (N, nt, nx); got {tuple(U.shape)}"
        assert x.ndim == 1, f"x must be 1D; got {tuple(x.shape)}"
        assert t.ndim == 1, f"t must be 1D; got {tuple(t.shape)}"
        assert U.shape[1] == len(t), "Time dimension mismatch: U.shape[1] != len(t)"
        assert U.shape[2] == len(x), "Space dimension mismatch: U.shape[2] != len(x)"

        # Optional train split
        if "train_idx" in dataset and dataset["train_idx"] is not None:
            train_idx = np.asarray(dataset["train_idx"], dtype=int)
            if len(train_idx) == 0:
                raise ValueError("dataset['train_idx'] is empty.")
            U = U[train_idx]

        t_train_end = float(dataset["t_train_end"])
        nu = float(dataset.get("nu", np.nan))

        # Slice training-time block
        train_mask = (t <= t_train_end + 1e-9)
        t_train = t[train_mask]
        U_train = U[:, train_mask, :]

        if t_train.shape[0] < 2:
            raise ValueError("Training time block must contain at least 2 snapshots.")

        ICs = U_train[:, 0, :]  # (N_train, nx)

        # Store grid/stats
        self._x_grid = x.to(self.device)
        self._t_train_rel = t_train.to(self.device)
        self._t_train_end = t_train_end
        self._nu = nu

        self._u_mean = float(U_train.mean())
        self._u_std = float(U_train.std()) + 1e-12

        # Targets normalized, inputs kept physical scale
        Y = U_train.unsqueeze(1)  # (N_train, 1, nt_train, nx)
        Y_norm = (Y - self._u_mean) / self._u_std

        ds = TensorDataset(ICs, Y_norm)
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

        self.model = self._build_model()
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        if cfg.scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
        elif cfg.scheduler == "step":
            sched = torch.optim.lr_scheduler.StepLR(
                opt, step_size=cfg.step_size, gamma=cfg.step_gamma
            )
        else:
            raise ValueError(f"Unknown scheduler: {cfg.scheduler}")

        loss_fn = LpLoss(d=2, p=2)

        history = {"epoch": [], "loss": [], "lr": []}
        t0_total = time.perf_counter()

        for epoch in range(cfg.epochs):
            self.model.train()
            running = 0.0

            for ic_batch, y_batch in loader:
                ic_batch = ic_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                xin = _build_input_tensor(ic_batch, self._x_grid, self._t_train_rel)
                pred = self.model(xin)

                loss = loss_fn(pred, y_batch)

                opt.zero_grad()
                loss.backward()
                opt.step()

                running += loss.item() * ic_batch.shape[0]

            sched.step()
            running /= len(ds)

            history["epoch"].append(epoch)
            history["loss"].append(running)
            history["lr"].append(opt.param_groups[0]["lr"])

            if cfg.verbose and (epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1):
                print(
                    f"  epoch {epoch:4d}/{cfg.epochs}   "
                    f"loss={running:.4e}   lr={opt.param_groups[0]['lr']:.2e}"
                )

        wall_time = time.perf_counter() - t0_total

        per_ic_rel_l2 = self._per_ic_train_error(ICs, U_train)
        n_params = sum(p.numel() for p in self.model.parameters())

        return {
            "wall_time_s": wall_time,
            "final_loss": history["loss"][-1],
            "history": history,
            "n_parameters": n_params,
            "per_ic_train_rel_l2": per_ic_rel_l2,
            "device": str(self.device),
            "n_modes_t": cfg.n_modes_t,
            "n_modes_x": cfg.n_modes_x,
            "hidden_channels": cfg.hidden_channels,
            "n_layers": cfg.n_layers,
            "training_strategy": "IC_to_full_training_block",
            "extrapolation_strategy": "blockwise_autoregressive_rollout",
        }

    # -------------------------------------------------------------------------
    # AbstractSolver.predict
    # -------------------------------------------------------------------------
    def predict(self, ic: np.ndarray, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Predict u at scattered query points (x,t)."""
        if self.model is None:
            raise RuntimeError("FNOSolver.predict called before fit/load.")

        x_q = np.asarray(x, dtype=np.float64).ravel()
        t_q = np.asarray(t, dtype=np.float64).ravel()

        if x_q.shape[0] != t_q.shape[0]:
            raise ValueError("predict: x and t must have the same length.")

        x_native_np = self._x_grid.detach().cpu().numpy()
        dx = x_native_np[1] - x_native_np[0]
        x_min = float(x_native_np[0])
        x_max = float(x_native_np[-1] + dx)
        x_q = _wrap_x(x_q, x_min, x_max)

        t_native = self._build_native_time_grid(float(t_q.max()))
        u_grid = self.rollout(ic, x_native_np, t_native)

        return self._bilinear_interp(u_grid, t_native, x_native_np, t_q, x_q)

    # -------------------------------------------------------------------------
    # Full-grid rollout
    # -------------------------------------------------------------------------
    def rollout(self, ic: np.ndarray, x_grid: np.ndarray,
                t_grid: np.ndarray) -> np.ndarray:
        """Predict u on a requested full (t,x) grid.

        The initial condition must be sampled on the native training grid.
        """
        if self.model is None:
            raise RuntimeError("FNOSolver.rollout called before fit/load.")

        self.model.eval()

        x_native = self._x_grid
        nx_native = x_native.shape[0]
        t_rel = self._t_train_rel
        T_blk = float(t_rel[-1].item())

        ic_arr = np.asarray(ic, dtype=np.float32).reshape(-1)
        if ic_arr.shape[0] != nx_native:
            raise ValueError(
                f"IC length {ic_arr.shape[0]} does not match native nx={nx_native}. "
                "Resample the IC to the dataset grid before calling rollout."
            )

        t_grid = np.asarray(t_grid, dtype=np.float64).reshape(-1)
        x_grid = np.asarray(x_grid, dtype=np.float64).reshape(-1)

        t_max = float(np.max(t_grid))
        n_blocks = int(np.ceil(t_max / T_blk)) if t_max > 0 else 1
        n_blocks = max(n_blocks, 1)

        ic_t = torch.as_tensor(ic_arr, dtype=torch.float32, device=self.device).reshape(1, nx_native)

        absolute_times: list[np.ndarray] = []
        slabs: list[np.ndarray] = []

        with torch.no_grad():
            for b in range(n_blocks):
                xin = _build_input_tensor(ic_t, x_native, t_rel)
                pred = self.model(xin)
                u_blk = pred[0, 0] * self._u_std + self._u_mean
                u_np = u_blk.detach().cpu().numpy()

                t_abs = t_rel.detach().cpu().numpy() + b * T_blk

                if b == 0:
                    absolute_times.append(t_abs)
                    slabs.append(u_np)
                else:
                    # Drop duplicate start slice
                    absolute_times.append(t_abs[1:])
                    slabs.append(u_np[1:])

                # Next block input is the last predicted physical state
                ic_t = u_blk[-1:, :].detach()

        t_full = np.concatenate(absolute_times)
        u_full = np.concatenate(slabs, axis=0)

        x_native_np = self._x_grid.detach().cpu().numpy()
        return self._regrid(u_full, t_full, x_native_np, t_grid, x_grid)

    def num_parameters(self) -> int:
        """Trainable parameter count of the FNO network."""
        if self.model is None:
            return 0
        return int(sum(p.numel() for p in self.model.parameters()))

    # -------------------------------------------------------------------------
    # Save/load
    # -------------------------------------------------------------------------
    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted FNOSolver.")

        from common.persistence import write_manifest

        # Uniform directory-with-manifest convention. Backward compatible:
        # if `path` ends in .pt it is the weights file and the manifest is
        # written alongside it; otherwise `path` is a directory holding
        # fno_model.pt. Either way a manifest.json is produced so the model
        # can be reloaded through common.persistence.load_any.
        if path.endswith(".pt"):
            out_dir = os.path.dirname(path) or "."
            ckpt_file = path
        else:
            out_dir = path
            ckpt_file = os.path.join(out_dir, "fno_model.pt")
        os.makedirs(out_dir, exist_ok=True)

        torch.save({
            "state_dict": self.model.state_dict(),
            "cfg": asdict(self.cfg),
            "x_grid": self._x_grid.detach().cpu(),
            "t_train_rel": self._t_train_rel.detach().cpu(),
            "t_train_end": self._t_train_end,
            "u_mean": self._u_mean,
            "u_std": self._u_std,
            "nu": self._nu,
            "training_strategy": "IC_to_full_training_block",
            "extrapolation_strategy": "blockwise_autoregressive_rollout",
        }, ckpt_file)

        write_manifest(out_dir, "fno", os.path.basename(ckpt_file),
                       name=self.name, framework="pytorch-neuralop")

    def load(self, path: str) -> None:
        from common.persistence import read_manifest

        # Accept a directory (preferred) or a legacy .pt file.
        if os.path.isdir(path):
            man = read_manifest(path)
            fname = man["checkpoint"] if man else "fno_model.pt"
            ckpt_file = os.path.join(path, fname)
        else:
            ckpt_file = path

        ckpt = _torch_load(ckpt_file, map_location=self.device)

        for k, v in ckpt["cfg"].items():
            setattr(self.cfg, k, v)

        self._x_grid = ckpt["x_grid"].to(self.device)
        self._t_train_rel = ckpt["t_train_rel"].to(self.device)
        self._t_train_end = ckpt["t_train_end"]
        self._u_mean = float(ckpt["u_mean"])
        self._u_std = float(ckpt["u_std"])
        self._nu = ckpt.get("nu", None)

        self.model = self._build_model()
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------
    def _per_ic_train_error(self, ICs: torch.Tensor,
                            U_train: torch.Tensor) -> np.ndarray:
        self.model.eval()
        errs = []

        with torch.no_grad():
            for i in range(ICs.shape[0]):
                ic_t = ICs[i:i + 1].to(self.device)
                xin = _build_input_tensor(ic_t, self._x_grid, self._t_train_rel)
                pred = self.model(xin)[0, 0] * self._u_std + self._u_mean
                ref = U_train[i].to(self.device)

                num = torch.linalg.norm(pred - ref)
                den = torch.linalg.norm(ref) + 1e-12
                errs.append((num / den).item())

        return np.asarray(errs, dtype=np.float64)

    def _build_native_time_grid(self, t_max: float) -> np.ndarray:
        t_rel = self._t_train_rel.detach().cpu().numpy()
        T_blk = float(t_rel[-1])

        if t_max <= T_blk + 1e-9:
            return t_rel.copy()

        n_blocks = int(np.ceil(t_max / T_blk))
        chunks = [t_rel]

        for b in range(1, n_blocks):
            chunks.append(t_rel[1:] + b * T_blk)

        return np.concatenate(chunks)

    @staticmethod
    def _regrid(u: np.ndarray,
                t_src: np.ndarray,
                x_src: np.ndarray,
                t_dst: np.ndarray,
                x_dst: np.ndarray) -> np.ndarray:
        """Bilinear regrid from source (t,x) grid to destination (t,x) grid.
        Periodic in x.
        """
        t_dst = np.asarray(t_dst, dtype=np.float64)
        x_dst = np.asarray(x_dst, dtype=np.float64)

        dx = x_src[1] - x_src[0]
        x_min = float(x_src[0])
        x_max = float(x_src[-1] + dx)
        x_dst = _wrap_x(x_dst, x_min, x_max)

        out = np.empty((len(t_dst), len(x_dst)), dtype=np.float64)

        x_ext = np.concatenate([x_src, [x_src[-1] + dx]])

        for i, tq in enumerate(t_dst):
            j = int(np.clip(np.searchsorted(t_src, tq), 1, len(t_src) - 1))
            t0, t1 = t_src[j - 1], t_src[j]
            wt = 0.0 if t1 == t0 else (tq - t0) / (t1 - t0)
            wt = float(np.clip(wt, 0.0, 1.0))

            row0 = u[j - 1]
            row1 = u[j]
            row = (1.0 - wt) * row0 + wt * row1

            row_ext = np.concatenate([row, [row[0]]])
            out[i] = np.interp(x_dst, x_ext, row_ext)

        return out

    @staticmethod
    def _bilinear_interp(u: np.ndarray,
                         t_src: np.ndarray,
                         x_src: np.ndarray,
                         t_q: np.ndarray,
                         x_q: np.ndarray) -> np.ndarray:
        """Scattered bilinear interpolation. Periodic in x."""
        t_q = np.asarray(t_q, dtype=np.float64)
        x_q = np.asarray(x_q, dtype=np.float64)

        dx = x_src[1] - x_src[0]
        x_min = float(x_src[0])
        x_max = float(x_src[-1] + dx)
        x_q = _wrap_x(x_q, x_min, x_max)

        x_ext = np.concatenate([x_src, [x_src[-1] + dx]])
        u_ext = np.concatenate([u, u[:, :1]], axis=1)

        out = np.empty(len(t_q), dtype=np.float64)

        for k in range(len(t_q)):
            tq = t_q[k]
            j = int(np.clip(np.searchsorted(t_src, tq), 1, len(t_src) - 1))
            t0, t1 = t_src[j - 1], t_src[j]
            wt = 0.0 if t1 == t0 else (tq - t0) / (t1 - t0)
            wt = float(np.clip(wt, 0.0, 1.0))

            r0 = np.interp(x_q[k], x_ext, u_ext[j - 1])
            r1 = np.interp(x_q[k], x_ext, u_ext[j])

            out[k] = (1.0 - wt) * r0 + wt * r1

        return out
