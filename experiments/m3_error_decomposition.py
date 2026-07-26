from __future__ import annotations
import os, sys, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hybrid_pde.control_214133E import config
from hybrid_pde.control_214133E.groundtruth import load_reference, relative_l2
from hybrid_pde.control_214133E.integrate import (
    load_ml_solver, load_numerical_solver, load_coupling)
from hybrid_pde.control_214133E.controller import AdaptiveController, thresholds_for_target
from hybrid_pde.control_214133E.runtime import HybridRuntime
from hybrid_pde.control_214133E.trigger import TrustMonitorAdapter
from hybrid_pde.control_214133E.robustness import hit_rate, mean_std
from hybrid_pde.trust.coarse_reference import CoarseReferenceMonitor


class OracleDetector:
    def __init__(self, ml_step_error, target):
        self.e = np.asarray(ml_step_error, float)
        self.target = float(target)
        self.i = 0
        self.tripped = False

    def __call__(self, state, t):
        if self.e[min(self.i, len(self.e) - 1)] > self.target:
            self.tripped = True
        self.i += 1
        return (0.0, True) if self.tripped else (1.0, False)

    def reset(self):
        self.i = 0
        self.tripped = False


def ml_step_errors(pred, ref):
    return [relative_l2(pred[n], ref[n]) for n in range(len(ref))]


def run_pair(problems, ml_preds, ml, num, coupling, x, t, targets):
    rows = []
    for tg in targets:
        lo, _ = thresholds_for_target(tg)
        real_errs, real_sw, orc_errs, orc_sw = [], [], [], []
        for (ic, ref), pred in zip(problems, ml_preds):
            mon = TrustMonitorAdapter(CoarseReferenceMonitor(ic, x, n=256))
            r = HybridRuntime(ml, num, mon, coupling, AdaptiveController(lo, 1.1)).run(ic, x, t, tg, reference=ref)
            real_errs.append(r.cost.achieved_error)
            real_sw.append(r.switch_times[0] if r.switch_times else float(t[-1]))

            orc = OracleDetector(ml_step_errors(pred, ref), tg)
            o = HybridRuntime(ml, num, orc, coupling, AdaptiveController(lo, 1.1)).run(ic, x, t, tg, reference=ref)
            orc_errs.append(o.cost.achieved_error)
            orc_sw.append(o.switch_times[0] if o.switch_times else float(t[-1]))
        rows.append({
            "target": float(tg),
            "real_error": mean_std(real_errs)[0], "real_hit": hit_rate(real_errs, tg),
            "real_switch_t": float(np.mean(real_sw)),
            "oracle_error": mean_std(orc_errs)[0], "oracle_hit": hit_rate(orc_errs, tg),
            "oracle_switch_t": float(np.mean(orc_sw)),
            "detection_lag_t": float(np.mean(real_sw) - np.mean(orc_sw)),
        })
    return rows


def main():
    R = load_reference()
    x, t = R.x, R.t
    ml = load_ml_solver()
    num = load_numerical_solver()
    coupling = load_coupling()
    idx = config.TEST_IC_INDICES
    problems = [(R.ICs[i], R.u[i]) for i in idx]
    ml.rollout(R.ICs[idx[0]], x, t)  # warm up
    ml_preds = [np.asarray(ml.rollout(ic, x, t), float) for ic, _ in problems]
    targets = config.DEFAULT_ACCURACY_TARGETS

    rows = run_pair(problems, ml_preds, ml, num, coupling, x, t, targets)

    out = {"grid": {"nx": int(len(x)), "nt": int(len(t))},
           "reliable_horizon": float(getattr(R, "t_train_end", float("nan"))),
           "decomposition": rows}
    out_dir = os.path.join(config.RESULTS_DIR, "m3", "error_decomposition")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "m3_error_decomposition.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1, default=float)

    print("ERROR DECOMPOSITION  (real monitor vs oracle detector, same correction)")
    print("%-8s %10s %8s %10s %8s %10s" % ("target", "real_err", "hit", "oracle_err", "hit", "lag_t"))
    for r in rows:
        print("%-8.3g %9.2f%% %7.0f%% %9.2f%% %7.0f%% %10.3f" % (
            r["target"], 100 * r["real_error"], 100 * r["real_hit"],
            100 * r["oracle_error"], 100 * r["oracle_hit"], r["detection_lag_t"]))
    print("\nsaved", path)
    return out


if __name__ == "__main__":
    main()
