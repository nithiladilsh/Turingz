from __future__ import annotations
import numpy as np


def hit_rate(errors, target):
    e = np.asarray(errors, dtype=float)
    return float(np.mean(e <= target)) if e.size else float("nan")


def mean_std(values):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1) if v.size > 1 else 0.0)
