from typing import Dict

import numpy as np

SHOCK_T_LO = 0.25
SHOCK_T_HI = 0.75


def _per_time_rel_l2(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    num = np.linalg.norm(pred - ref, axis=1)
    den = np.linalg.norm(ref, axis=1) + 1e-12
    return num / den


def _per_time_linf(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.max(np.abs(pred - ref), axis=1)


def _agg(v: np.ndarray) -> Dict[str, float]:
    if v.size == 0:
        return {"mean": float("nan"), "max": float("nan")}
    return {"mean": float(np.mean(v)), "max": float(np.max(v))}


def compute_metrics(pred: np.ndarray, ref: np.ndarray, t: np.ndarray,
                    t_train_end: float = 1.0) -> Dict:
    """pred, ref: (nt, nx) on the same grid. Returns global + windowed errors,
    with windows matching the Cole-Hopf cross-verification report."""
    assert pred.shape == ref.shape, (pred.shape, ref.shape)
    rel = _per_time_rel_l2(pred, ref)
    linf = _per_time_linf(pred, ref)

    pre = t < SHOCK_T_LO
    shock = (t >= SHOCK_T_LO) & (t <= SHOCK_T_HI)
    post = (t > SHOCK_T_HI) & (t <= t_train_end + 1e-12)
    extrap = t > t_train_end + 1e-12

    out = {
        "relative_l2": {**_agg(rel), "final": float(rel[-1])},
        "linf": {**_agg(linf), "final": float(linf[-1])},
        "windowed_rel_l2": {
            "pre_shock": _agg(rel[pre]),
            "shock": _agg(rel[shock]),
            "post_shock": _agg(rel[post]),
            "extrap": _agg(rel[extrap]),
        },
        "windowed_linf": {
            "pre_shock": _agg(linf[pre]),
            "shock": _agg(linf[shock]),
            "post_shock": _agg(linf[post]),
            "extrap": _agg(linf[extrap]),
        },
        "per_time_rel_l2": rel.tolist(),
        "t": t.tolist(),
        "windows": {"shock_t_lo": SHOCK_T_LO, "shock_t_hi": SHOCK_T_HI,
                    "t_train_end": t_train_end},
    }
    return out
