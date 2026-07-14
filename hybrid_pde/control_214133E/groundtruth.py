from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Reference:
    u: np.ndarray
    ICs: np.ndarray
    x: np.ndarray
    t: np.ndarray
    t_train_end: float


def load_reference(path=None) -> Reference:
    import torch
    from . import config
    p = config.COLEHOPF_PT if path is None else path
    d = torch.load(p, weights_only=False, map_location="cpu")
    def to_np(a):
        return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
    return Reference(to_np(d["u"]), to_np(d["ICs"]), to_np(d["x"]),
                     to_np(d["t"]), float(d["t_train_end"]))


def relative_l2(pred, ref, time_mask=None):
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if time_mask is not None:
        pred, ref = pred[time_mask], ref[time_mask]
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-12))
