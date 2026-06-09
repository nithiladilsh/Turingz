import os
from typing import Callable, Optional, Tuple

import numpy as np
import torch


class ColeHopfDataset:
    def __init__(self, pt_path: str, sample: int = 0):
        if not os.path.isfile(pt_path):
            raise FileNotFoundError(pt_path)
        blob = torch.load(pt_path, map_location="cpu", weights_only=False)
        self._init_from_blob(blob, sample)

    @classmethod
    def from_blob(cls, blob: dict, sample: int = 0) -> "ColeHopfDataset":
        obj = cls.__new__(cls)
        obj._init_from_blob(blob, sample)
        return obj

    @staticmethod
    def _np(v) -> np.ndarray:
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        elif hasattr(v, "numpy"):
            v = v.numpy()
        return np.asarray(v, dtype=np.float64)

    def _init_from_blob(self, blob: dict, sample: int) -> None:
        self.sample = int(sample)
        self.x = self._np(blob["x"])                  
        self.t = self._np(blob["t"])                    
        self.u = self._np(blob["u"])                       
        self.ICs = self._np(blob["ICs"])           

        self.nu = float(blob["nu"])
        self.L = float(blob["L"])
        self.x_start = float(blob["x_start"])
        self.x_end = float(blob["x_end"])
        self.T = float(blob["T"])
        self.t_train_end = float(blob["t_train_end"])
        self.nx = int(blob["nx"]) if "nx" in blob else int(self.x.shape[0])
        self.nt = int(blob["nt"]) if "nt" in blob else int(self.t.shape[0])

        if not (0 <= self.sample < self.u.shape[0]):
            raise IndexError(f"sample {self.sample} out of range [0,{self.u.shape[0]})")

    @property
    def u_ref(self) -> np.ndarray:
        return self.u[self.sample]                        

    @property
    def ic_ref(self) -> np.ndarray:
        return self.ICs[self.sample]                   

    def train_time_mask(self) -> np.ndarray:
        return self.t <= self.t_train_end + 1e-12

    def extrap_time_mask(self) -> np.ndarray:
        return self.t > self.t_train_end + 1e-12

    def ic_func(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return IC(x) as a callable. Uses the exact analytic sine when the
        stored IC is sin(pi x) (the focal sample), otherwise a periodic linear
        interpolation of the gridded IC. Detection by value (not sample index)
        so a standalone-reloaded single-sample dataset rebuilds the right IC."""
        ic = self.ic_ref
        if np.allclose(ic, np.sin(np.pi * self.x), atol=1e-8):
            return lambda X: np.sin(np.pi * X[:, 0:1])
        xp = np.append(self.x, self.x[0] + self.L)     
        up = np.append(ic, ic[0])

        def f(X: np.ndarray) -> np.ndarray:
            xq = np.mod(X[:, 0:1] - self.x_start, self.L) + self.x_start
            return np.interp(xq.ravel(), xp, up)[:, None]
        return f

    def anchor_points(self, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        mask = self.train_time_mask()
        tt, uu = self.t[mask], self.u_ref[mask]               
        Tg, Xg = np.meshgrid(tt, self.x, indexing="ij")
        pts = np.column_stack([Xg.ravel(), Tg.ravel()])      
        vals = uu.ravel()[:, None]
        k = min(n, pts.shape[0])
        idx = rng.choice(pts.shape[0], size=k, replace=False)
        return pts[idx], vals[idx]

    def eval_grid(self, train_only: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = self.t[self.train_time_mask()] if train_only else self.t
        Tg, Xg = np.meshgrid(t, self.x, indexing="ij")
        X = np.column_stack([Xg.ravel(), Tg.ravel()])  
        return X, t, self.x

    def ref_on(self, t_mask: Optional[np.ndarray] = None) -> np.ndarray:
        return self.u_ref if t_mask is None else self.u_ref[t_mask]
