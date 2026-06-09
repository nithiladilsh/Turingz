import os
os.environ.setdefault("DDE_BACKEND", "pytorch")

import sys
import json
import time
import warnings
from typing import Optional, Dict, Any, Union

import numpy as np
import torch
import deepxde as dde

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from abstract_solver import AbstractSolver 

from .config import PINNConfig
from .dataset import ColeHopfDataset


class BurgersPINN(AbstractSolver):
    def __init__(self, cfg: Optional[PINNConfig] = None):
        self.cfg = cfg or PINNConfig()
        self.ds: Optional[ColeHopfDataset] = None
        self.nu: Optional[float] = None
        self.model: Optional[Any] = None
        self.geomtime = None
        self.data = None
        self._bc_kinds = []                 
        self._loss_weights = None
        self._losshistory = None
        self._save_path = None
        self._trained_ic = None 
        self._ic_warned = False

    @property
    def name(self) -> str:
        hidden = list(self.cfg.hidden)
        depth, width = len(hidden), (hidden[0] if hidden else 0)
        per = "hardP" if self.cfg.hard_periodic else "softP"
        anc = ",anchors" if self.cfg.use_data_anchors else ""
        return f"PINN({depth}x{width},{per}{anc},sample={self.cfg.sample})"

    def _feature_transform(self, x):
        xs, t = x[:, 0:1], x[:, 1:2]
        feats = [t]
        for k in range(1, self.cfg.n_harmonics + 1):
            feats.append(torch.sin(k * np.pi * xs))
            feats.append(torch.cos(k * np.pi * xs))
        return torch.cat(feats, dim=1)

    def _pde(self, x, u):
        assert self.nu is not None
        u_t = dde.grad.jacobian(u, x, i=0, j=1)
        u_x = dde.grad.jacobian(u, x, i=0, j=0)
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        assert u_xx is not None
        return u_t + u * u_x - self.nu * u_xx

    def _build(self):
        if self.ds is None:
            raise RuntimeError("_build called before a dataset was attached.")
        dde.config.set_random_seed(self.cfg.seed)  # type: ignore[attr-defined]
        if self.cfg.float64:
            dde.config.set_default_float("float64")  # type: ignore[attr-defined]

        t_end = self.cfg.t_train_end or self.ds.t_train_end
        geom = dde.geometry.Interval(self.ds.x_start, self.ds.x_end)
        timedomain = dde.geometry.TimeDomain(0.0, t_end)
        self.geomtime = dde.geometry.GeometryXTime(geom, timedomain)

        ic = dde.icbc.IC(self.geomtime, self.ds.ic_func(),
                         lambda _, on_initial: on_initial)
        bcs: list[Any] = [ic]
        self._bc_kinds = ["ic"]

        if not self.cfg.hard_periodic:
            bcs.append(dde.icbc.PeriodicBC(self.geomtime, 0,
                       lambda _, on_boundary: on_boundary, derivative_order=0))
            bcs.append(dde.icbc.PeriodicBC(self.geomtime, 0,
                       lambda _, on_boundary: on_boundary, derivative_order=1))
            self._bc_kinds += ["bc", "bc"]

        if self.cfg.use_data_anchors:
            rng = np.random.default_rng(self.cfg.seed)
            ax, au = self.ds.anchor_points(self.cfg.n_anchors, rng)
            bcs.append(dde.icbc.PointSetBC(ax, au, component=0))
            self._bc_kinds += ["data"]

        self.data = dde.data.TimePDE(
            self.geomtime, self._pde, bcs,
            num_domain=self.cfg.num_domain,
            num_boundary=(0 if self.cfg.hard_periodic else self.cfg.num_boundary),
            num_initial=self.cfg.num_initial,
        )

        in_dim = (2 * self.cfg.n_harmonics + 1) if self.cfg.hard_periodic else 2
        net = dde.nn.FNN(  # type: ignore[attr-defined]
            [in_dim] + list(self.cfg.hidden) + [1],
            self.cfg.activation, "Glorot normal")
        if self.cfg.hard_periodic:
            net.apply_feature_transform(lambda x: self._feature_transform(x))

        self.model = dde.Model(self.data, net)
        self._loss_weights = self._build_loss_weights()
        self._trained_ic = np.asarray(self.ds.ic_ref, dtype=np.float64)

    def _build_loss_weights(self):
        w = {"ic": self.cfg.w_ic, "bc": self.cfg.w_bc, "data": self.cfg.w_data}
        return [self.cfg.w_pde] + [w[k] for k in self._bc_kinds]

    def _run_training(self, out_dir: Optional[str] = None):
        assert self.model is not None
        self.model.compile("adam", lr=self.cfg.lr, loss_weights=self._loss_weights)
        callbacks = []
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            callbacks.append(dde.callbacks.ModelCheckpoint(
                os.path.join(out_dir, "ckpt"), save_better_only=True,
                period=self.cfg.display_every))
        history, _ = self.model.train(
            iterations=self.cfg.adam_iters, callbacks=callbacks,
            display_every=self.cfg.display_every)
        if self.cfg.lbfgs:
            self.model.compile("L-BFGS", loss_weights=self._loss_weights)
            history, _ = self.model.train(display_every=self.cfg.display_every)
        self._losshistory = history
        return history
    
    def fit(self, dataset: Union[Dict[str, Any], ColeHopfDataset],
            out_dir: Optional[str] = None) -> Dict[str, Any]:
        t0 = time.time()
        if isinstance(dataset, ColeHopfDataset):
            self.ds = dataset
            self.cfg.sample = dataset.sample
        else:
            self.ds = ColeHopfDataset.from_blob(dataset, sample=self.cfg.sample)

        self.nu = self.ds.nu
        self._build()
        history = self._run_training(out_dir)
        if out_dir:
            self.save(out_dir)

        info = {
            "name": self.name,
            "sample": int(self.cfg.sample),
            "nu": float(self.nu),
            "wall_time_s": time.time() - t0,
            "n_parameters": self.num_parameters(),
            "final_loss_train": [float(v) for v in history.loss_train[-1]],
            "final_loss_test": [float(v) for v in history.loss_test[-1]],
            "loss_order": ["pde"] + self._bc_kinds,
        }
        return info

    def _check_ic(self, ic: Optional[np.ndarray]):
        if ic is None or self._trained_ic is None or self._ic_warned:
            return
        ic = np.asarray(ic, dtype=np.float64).ravel()
        if ic.shape == self._trained_ic.shape and not np.allclose(
                ic, self._trained_ic, atol=1e-4):
            warnings.warn(
                f"{self.name}: predict() called with an IC that differs from the "
                "trained one. A PINN is single-instance; this prediction reflects "
                "the IC it was fitted on, not the IC passed in.",
                RuntimeWarning, stacklevel=2)
            self._ic_warned = True

    def predict(self, ic: np.ndarray, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("BurgersPINN.predict called before fit/load.")
        self._check_ic(ic)
        x_q = np.asarray(x, dtype=np.float64).ravel()
        t_q = np.asarray(t, dtype=np.float64).ravel()
        if x_q.shape[0] != t_q.shape[0]:
            raise ValueError("predict: x and t must have the same length.")
        X = np.column_stack([x_q, t_q])         
        return np.asarray(self.model.predict(X)).ravel()

    def rollout(self, ic: np.ndarray, x_grid: np.ndarray,
                t_grid: np.ndarray) -> np.ndarray:
        """Predict u on a full (t x x) grid. Returns (nt, nx), row index = t."""
        if self.model is None:
            raise RuntimeError("BurgersPINN.rollout called before fit/load.")
        self._check_ic(ic)
        x_grid = np.asarray(x_grid, dtype=np.float64).ravel()
        t_grid = np.asarray(t_grid, dtype=np.float64).ravel()
        Xg, Tg = np.meshgrid(x_grid, t_grid, indexing="xy")   # (nt, nx)
        X = np.column_stack([Xg.ravel(), Tg.ravel()])
        u = np.asarray(self.model.predict(X)).reshape(len(t_grid), len(x_grid))
        return u

    def supported_samples(self, candidate_indices):
        """A PINN is single-instance: it only predicts the IC it was trained
        on (cfg.sample), so it is graded only on that sample."""
        s = self.cfg.sample
        return [s] if s in list(candidate_indices) else []

    def predict_grid(self, train_only: bool = False):
        if self.model is None:
            raise RuntimeError("BurgersPINN.predict_grid called before fit/load.")
        if self.ds is None:
            raise RuntimeError("predict_grid needs an attached dataset (fit/load first).")
        X, t, x = self.ds.eval_grid(train_only=train_only)
        u = np.asarray(self.model.predict(X)).reshape(len(t), len(x))
        return u, t, x

    def num_parameters(self) -> int:
        if self.model is None:
            return 0
        return int(sum(p.numel() for p in self.model.net.parameters()))

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted BurgersPINN.")
        assert self.nu is not None and self.ds is not None
        os.makedirs(path, exist_ok=True)
        self._save_path = self.model.save(os.path.join(path, "pinn_model"))

        np.savez(
            os.path.join(path, "problem.npz"),
            x=self.ds.x, t=self.ds.t,
            ic=self.ds.ic_ref, u_ref=self.ds.u_ref,
            nu=self.nu, L=self.ds.L,
            x_start=self.ds.x_start, x_end=self.ds.x_end,
            T=self.ds.T, t_train_end=self.ds.t_train_end,
            sample=self.ds.sample, nx=self.ds.nx, nt=self.ds.nt,
        )

        meta = {
            "config": self.cfg.to_dict(),
            "checkpoint": os.path.basename(self._save_path),
            "domain": {"x_start": self.ds.x_start, "x_end": self.ds.x_end,
                       "L": self.ds.L,
                       "t_train_end": self.cfg.t_train_end or self.ds.t_train_end,
                       "T": self.ds.T},
            "physics": {"nu": self.nu},
            "sample": self.ds.sample,
            "loss_order": ["pde"] + self._bc_kinds,
            "loss_weights": self._loss_weights,
            "backend": "pytorch",
            "deepxde": dde.__version__,
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        from common.persistence import write_manifest
        write_manifest(path, "pinn", os.path.basename(self._save_path),
                       name=self.name, framework="deepxde-pytorch")

    def load(self, path: str) -> None:
        with open(os.path.join(path, "metadata.json")) as f:
            meta = json.load(f)
        self.cfg = PINNConfig.from_dict(meta["config"])

        prob = np.load(os.path.join(path, "problem.npz"))
        u_ref = prob["u_ref"]
        ic = prob["ic"]
        blob = {
            "x": prob["x"], "t": prob["t"],
            "u": u_ref[None, ...],             
            "ICs": ic[None, ...],         
            "nu": float(prob["nu"]), "L": float(prob["L"]),
            "x_start": float(prob["x_start"]), "x_end": float(prob["x_end"]),
            "T": float(prob["T"]), "t_train_end": float(prob["t_train_end"]),
            "nx": int(prob["nx"]), "nt": int(prob["nt"]),
        }
        self.ds = ColeHopfDataset.from_blob(blob, sample=0)
        self.nu = self.ds.nu
        self._build()
        assert self.model is not None
        self.model.compile("adam", lr=self.cfg.lr, loss_weights=self._loss_weights)
        ckpt_path = os.path.join(path, meta["checkpoint"])
        try:
            self.model.restore(ckpt_path, verbose=1)
        except Exception as _restore_err:
            blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = (blob["model_state_dict"]
                     if isinstance(blob, dict) and "model_state_dict" in blob
                     else blob)
            self.model.net.load_state_dict(state)
            print(f"  [load] optimizer restore skipped "
                  f"({type(_restore_err).__name__}); loaded network weights "
                  f"only (correct for inference/cost profiling).")
        self._save_path = ckpt_path
