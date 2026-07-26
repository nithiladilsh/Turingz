from __future__ import annotations
import os, sys, json, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hybrid_pde.control_214133E import config
from hybrid_pde.control_214133E.groundtruth import load_reference
from hybrid_pde.control_214133E.integrate import (
    load_ml_solver, load_numerical_solver, load_coupling, per_step_costs)
from hybrid_pde.control_214133E.controller import AdaptiveController, thresholds_for_target
from hybrid_pde.control_214133E.runtime import HybridRuntime
from hybrid_pde.control_214133E.trigger import TrustMonitorAdapter
from hybrid_pde.control_214133E.robustness import mean_std
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
    num.rollout(R.ICs[idx[0]], x, t)
    ml_c, num_c = per_step_costs(ml, num, R.ICs[idx[0]], x, t)
    targets = config.DEFAULT_ACCURACY_TARGETS

    pred_all, meas_all = [], []
    rows = []
    for tg in targets:
        lo, _ = thresholds_for_target(tg)
        preds, meas = [], []
        for ic, ref in problems:
            mon = TrustMonitorAdapter(CoarseReferenceMonitor(ic, x, n=256))
            rt = HybridRuntime(ml, num, mon, coupling, AdaptiveController(lo, 1.1))
            a = time.perf_counter()
            res = rt.run(ic, x, t, tg, reference=ref)
            wall = time.perf_counter() - a
            predicted = res.cost.ml_steps * ml_c + res.cost.correction_steps * num_c
            preds.append(predicted); meas.append(wall)
            pred_all.append(predicted); meas_all.append(wall)
        rows.append({"target": float(tg),
                     "predicted_cost_s": mean_std(preds)[0],
                     "measured_cost_s": mean_std(meas)[0]})

    p = np.asarray(pred_all); m = np.asarray(meas_all)
    r = float(np.corrcoef(p, m)[0, 1]) if p.size > 1 else float("nan")
    mape = float(np.mean(np.abs(p - m) / (m + 1e-12)))
    slope = float(np.polyfit(p, m, 1)[0]) if p.size > 1 else float("nan")

    out = {"ml_step_s": ml_c, "num_step_s": num_c,
           "per_target": rows,
           "pearson_r": r, "mape": mape, "slope_measured_vs_predicted": slope,
           "n_points": int(p.size)}
    out_dir = os.path.join(config.RESULTS_DIR, "m3", "cost_model")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "m3_cost_model.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1, default=float)

    print("COST-MODEL VALIDATION  (linear model vs measured wall-clock)")
    print("%-8s %16s %16s" % ("target", "predicted (s)", "measured (s)"))
    for row in rows:
        print("%-8.3g %15.3f %16.3f" % (row["target"], row["predicted_cost_s"], row["measured_cost_s"]))
    print("\nPearson r = %.4f   MAPE = %.1f%%   slope = %.3f   (n=%d)" % (r, 100 * mape, slope, p.size))
    print("saved", path)
    return out


if __name__ == "__main__":
    main()
