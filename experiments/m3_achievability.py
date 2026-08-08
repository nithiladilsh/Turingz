from __future__ import annotations
import os, sys, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hybrid_pde.control_214133E import config
from hybrid_pde.control_214133E.groundtruth import load_reference
from hybrid_pde.control_214133E.integrate import (
    load_ml_solver, load_numerical_solver, load_coupling)
from hybrid_pde.control_214133E.controller import AdaptiveController, thresholds_for_target
from hybrid_pde.control_214133E.runtime import HybridRuntime
from hybrid_pde.control_214133E.trigger import TrustMonitorAdapter
from hybrid_pde.control_214133E.robustness import hit_rate, mean_std
from hybrid_pde.trust.coarse_reference import CoarseReferenceMonitor


def main():
    R = load_reference()
    x, t = R.x, R.t
    ml = load_ml_solver()
    num = load_numerical_solver()
    coupling = load_coupling()
    idx = config.TEST_IC_INDICES
    problems = [(R.ICs[i], R.u[i]) for i in idx]
    ml.rollout(R.ICs[idx[0]], x, t)  # warm up
    targets = config.DEFAULT_ACCURACY_TARGETS

    rows = []
    met_feats, missed_feats = [], []
    for tg in targets:
        lo, hi = thresholds_for_target(tg)
        feats, mets, errs = [], [], []
        for ic, ref in problems:
            mon = TrustMonitorAdapter(CoarseReferenceMonitor(ic, x, n=256))
            res = HybridRuntime(ml, num, mon, coupling, AdaptiveController(lo, 1.1)).run(ic, x, t, tg, reference=ref)
            tc = np.asarray(res.trust_curve, float)
            frac_below = float(np.mean(tc < lo))
            met = bool(res.cost.achieved_error <= tg)
            feats.append(frac_below); mets.append(met); errs.append(res.cost.achieved_error)
            (met_feats if met else missed_feats).append(frac_below)
        rows.append({
            "target": float(tg), "theta_lo": float(lo),
            "mean_error": mean_std(errs)[0], "hit_rate": hit_rate(errs, tg),
            "mean_frac_below": float(np.mean(feats)),
            "n_met": int(sum(mets)), "n_missed": int(len(mets) - sum(mets)),
        })

    mf = np.asarray(met_feats); xf = np.asarray(missed_feats)
    allf = np.concatenate([mf, xf]) if mf.size and xf.size else np.array([])
    best = {"threshold": None, "accuracy": None}
    if allf.size:
        cand = np.unique(allf)
        y = np.array([1] * len(mf) + [0] * len(xf))
        f = np.concatenate([mf, xf])
        acc, thr = -1.0, None
        for c in cand:
            pred = (f <= c).astype(int)  # low frac_below -> predict "met"
            a = float(np.mean(pred == y))
            if a > acc:
                acc, thr = a, float(c)
        best = {"threshold": thr, "accuracy": acc}

    out = {
        "feature": "fraction of steps trust < theta_lo (reference-free)",
        "per_target": rows,
        "met_feature_mean": float(mf.mean()) if mf.size else None,
        "missed_feature_mean": float(xf.mean()) if xf.size else None,
        "separator": best,
    }
    out_dir = os.path.join(config.RESULTS_DIR, "m3", "achievability")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "m3_achievability.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1, default=float)

    print("ACHIEVABILITY DIAGNOSTIC  (reference-free feature vs target met)")
    print("%-8s %8s %8s %14s  %s" % ("target", "error", "hit", "frac_below", "met/missed"))
    for r in rows:
        print("%-8.3g %7.2f%% %7.0f%% %13.3f  %d/%d" % (
            r["target"], 100 * r["mean_error"], 100 * r["hit_rate"],
            r["mean_frac_below"], r["n_met"], r["n_missed"]))
    print("\nmet-case feature mean   = %s" % out["met_feature_mean"])
    print("missed-case feature mean = %s" % out["missed_feature_mean"])
    print("best reference-free separator: frac_below <= %s  -> accuracy %s" % (best["threshold"], best["accuracy"]))
    print("\nsaved", path)
    return out


if __name__ == "__main__":
    main()
