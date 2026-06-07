"""
================================================================================
SHARED ACCURACY METRICS  —  single source of truth
Team : Turingz   File : common/metrics.py

One implementation of the accuracy metrics, used identically for PINN, FNO and
DeepONet. Each model used to compute error in its own script with its own
conventions; this module replaces all of them so the cross-model comparison is
apples-to-apples.

Everything operates on a predicted field and a reference field of the SAME
shape (nt, nx), with the row index being time. The time windows match the
Cole-Hopf cross-verification report (results/cross_verification.json):
    pre_shock  : t < 0.25
    shock      : 0.25 <= t <= 0.75
    post_shock : 0.75 < t <= t_train_end
    extrap     : t > t_train_end
================================================================================
"""

from typing import Dict
import numpy as np

# Window edges — identical to the cross-verification report.
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
    """Accuracy of one predicted field vs its reference.

    pred, ref : (nt, nx) arrays on the same grid.
    t         : (nt,) time coordinates.
    Returns global and windowed relative-L2 and L-infinity, plus the
    per-time relative-L2 curve.
    """
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    assert pred.shape == ref.shape, (pred.shape, ref.shape)
    assert pred.shape[0] == t.shape[0], (pred.shape, t.shape)

    rel = _per_time_rel_l2(pred, ref)
    linf = _per_time_linf(pred, ref)

    pre = t < SHOCK_T_LO
    shock = (t >= SHOCK_T_LO) & (t <= SHOCK_T_HI)
    post = (t > SHOCK_T_HI) & (t <= t_train_end + 1e-12)
    extrap = t > t_train_end + 1e-12

    return {
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
                    "t_train_end": float(t_train_end)},
    }


def aggregate_over_samples(per_sample: Dict[int, Dict]) -> Dict:
    """Summarize per-sample metrics across many ICs, split by regime.

    per_sample : {sample_index: metrics_dict_from_compute_metrics}.
                 Each metrics dict may carry a 'regime' key ('in_dist'/'ood').
    Returns mean/max of the headline numbers over 'all', 'in_dist' and 'ood'.
    """
    def collect(getter, items):
        vals = [getter(m) for m in items]
        if not vals:
            return {"mean": float("nan"), "max": float("nan")}
        return {"mean": float(np.mean(vals)), "max": float(np.max(vals))}

    items_all = list(per_sample.values())

    def subset(regime):
        return [m for m in per_sample.values() if m.get("regime") == regime]

    out: Dict[str, Dict] = {}
    for tag, items in [("all", items_all),
                       ("in_dist", subset("in_dist")),
                       ("ood", subset("ood"))]:
        if not items:
            continue
        out[tag] = {
            "n_samples": len(items),
            "global_rel_l2": collect(lambda m: m["relative_l2"]["mean"], items),
            "shock_rel_l2": collect(lambda m: m["windowed_rel_l2"]["shock"]["mean"], items),
            "extrap_rel_l2": collect(lambda m: m["windowed_rel_l2"]["extrap"]["mean"], items),
            "final_rel_l2": collect(lambda m: m["relative_l2"]["final"], items),
        }
    return out
