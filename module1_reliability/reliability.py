import os
import numpy as np
from numpy.typing import ArrayLike

from common import reliability_signals as R

THRESHOLD = 0.10

def roc_auc(scores: ArrayLike, labels: ArrayLike) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    ok = np.isfinite(scores)
    scores, labels = scores[ok], labels[ok]
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")   
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    auc = (ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return float(auc)

def load_reference_any(pt_path: str) -> dict:
    try:
        from common.evaluation import load_reference
        return load_reference(pt_path)
    except Exception:
        csv = os.path.splitext(pt_path)[0] + ".csv"
        nx, nt = 512, 200                      
        arr = np.loadtxt(csv, delimiter=",", skiprows=1, usecols=2)
        N = arr.size // (nt * nx)
        u = arr.reshape(N, nt, nx)
        x = np.linspace(-1.0, 1.0, nx, endpoint=False)
        t = np.concatenate([[0.0], np.linspace(0.01, 2.0, nt - 1)])
        return {"u": u, "x": x, "t": t, "nu": 1.0 / (100.0 * np.pi), "t_train_end": 1.0}


def analyze_model(eval_dict: dict, reference: dict, threshold: float = THRESHOLD) -> dict:
    x = reference["x"]; t = reference["t"]; nu = reference["nu"]; u = reference["u"]
    t_end = float(eval_dict.get("t_train_end", reference.get("t_train_end", 1.0)))
    extrap = t > t_end

    per_sample = {}
    pooled = {"raw": [], "excess": [], "periodicity": [], "label": []}

    for s, m in eval_dict["samples"].items():
        s = int(s)
        err = np.asarray(m["per_time_rel_l2"], dtype=np.float64)         
        sig = m.get("reliability_signals", {})
        raw = np.asarray(sig.get("residual_rms", {}).get("curve", [np.nan] * len(t)))
        peri = np.asarray(sig.get("periodicity_gap", {}).get("curve", [np.nan] * len(t)))
        floor = R.residual_curve(u[s], x, t, nu)                         
        excess = raw - floor                                             

        label = (err > threshold).astype(int)                           
        horizon = float(t[np.argmax(label)]) if label.any() else float(t[-1])

        per_sample[s] = {
            "regime": m.get("regime"),
            "true_reliable_horizon": horizon,      
            "extrap_fail_fraction": float(label[extrap].mean()),
        }
        pooled["raw"] += raw[extrap].tolist()
        pooled["excess"] += excess[extrap].tolist()
        pooled["periodicity"] += peri[extrap].tolist()
        pooled["label"] += label[extrap].tolist()

    lab = np.asarray(pooled["label"], dtype=np.int64)
    horizons = [v["true_reliable_horizon"] for v in per_sample.values()]
    report = {
        "solver": eval_dict["solver"],
        "threshold": threshold,
        "t_train_end": t_end,
        "n_extrap_points": int(len(lab)),
        "extrap_fail_fraction": float(lab.mean()) if len(lab) else float("nan"),
        "mean_reliable_horizon": float(np.mean(horizons)) if horizons else float("nan"),
        "detector_auc": {
            "raw_residual": roc_auc(pooled["raw"], lab),
            "excess_residual": roc_auc(pooled["excess"], lab),
            "periodicity_gap": roc_auc(pooled["periodicity"], lab),
        },
        "per_sample": per_sample,
        "_pool": pooled,
    }
    return report
