from __future__ import annotations
import json, time
import numpy as np
from . import config
from .groundtruth import load_reference, relative_l2
from .trigger import TrustMonitorAdapter
from .runtime import HybridRuntime
from .controller import AdaptiveController, FixedIntervalController, thresholds_for_target
from .integrate import load_ml_solver_by_name, load_numerical_solver, load_coupling
from hybrid_pde.trust.coarse_reference import CoarseReferenceMonitor


def _pattern(trust_curve, ctrl_factory, nt):
    ctrl = ctrl_factory()
    ctrl.configure(0.05)
    ctrl.reset()
    pat = []
    for i in range(nt):
        pat.append(bool(ctrl.decide(float(trust_curve[i]), False, 0.0, i).correct))
    return pat


def _stats(pat):
    n = len(pat)
    corr = sum(pat)
    first = pat.index(True) if True in pat else None
    runs, c = [], 0
    for p in pat:
        if p:
            c += 1
        elif c:
            runs.append(c); c = 0
    if c:
        runs.append(c)
    after = pat[first:] if first is not None else []
    return {"corrections": corr, "steps": n, "first_switch": first,
            "switch_events": len(runs), "runs": runs,
            "post_switch_frac": (sum(after) / len(after)) if after else 0.0,
            "latched": bool(first is not None and sum(after) == len(after))}


HARDCODED_THETA = 0.4


def main(targets=None, model="FNO", out=None):
    R = load_reference()
    ml = load_ml_solver_by_name(model, R.x)
    num = load_numerical_solver()
    idx = config.TEST_IC_INDICES
    x, t = R.x, R.t
    nt = len(t)
    ml.rollout(R.ICs[idx[0]], x, t)
    num.rollout(R.ICs[idx[0]], x, t)

    tg = targets if targets is not None else config.DEFAULT_ACCURACY_TARGETS
    POLICIES = {
        "latch":    lambda lo, hi: AdaptiveController(lo, 1.1),
        "deadband": lambda lo, hi: AdaptiveController(lo, hi),
        "naive":    lambda lo, hi: AdaptiveController(lo, lo),
        "hardcoded": lambda lo, hi: AdaptiveController(HARDCODED_THETA, 1.1),
    }

    rows = []
    for target in tg:
        lo, hi = thresholds_for_target(target)
        rec = {"target": float(target), "theta_lo": lo, "theta_hi": hi, "policies": {}}
        for name, fac in POLICIES.items():
            errs, costs, corrs, pats = [], [], [], []
            for i in idx:
                ic, ref = R.ICs[i], R.u[i]
                trust = TrustMonitorAdapter(CoarseReferenceMonitor(ic, x, n=256))
                rt = HybridRuntime(ml, num, trust, load_coupling(), fac(lo, hi))
                a = time.perf_counter()
                res = rt.run(ic, x, t, target, reference=ref)
                costs.append(time.perf_counter() - a)
                errs.append(res.cost.achieved_error)
                corrs.append(res.cost.correction_steps)
                pats.append({"switch_events": len(res.switch_times),
                             "first_switch_t": res.switch_times[0] if res.switch_times else None})
            rec["policies"][name] = {
                "mean_error": float(np.mean(errs)), "std_error": float(np.std(errs)),
                "cost_s": float(np.mean(costs)),
                "corrections": float(np.mean(corrs)), "steps": nt,
                "corr_frac": float(np.mean(corrs)) / nt,
                "switch_events": float(np.mean([p["switch_events"] for p in pats])),
                "hit_rate": float(np.mean([e <= target for e in errs])),
                "latched": bool(np.mean([p["switch_events"] for p in pats]) <= 1.0),
            }

        base = rec["policies"]["latch"]
        k = max(1, int(round(nt / max(base["corrections"], 1))))
        errs, costs, corrs = [], [], []
        for i in idx:
            ic, ref = R.ICs[i], R.u[i]
            trust = TrustMonitorAdapter(CoarseReferenceMonitor(ic, x, n=256))
            rt = HybridRuntime(ml, num, trust, load_coupling(), FixedIntervalController(k))
            a = time.perf_counter()
            res = rt.run(ic, x, t, target, reference=ref)
            costs.append(time.perf_counter() - a)
            errs.append(res.cost.achieved_error)
            corrs.append(res.cost.correction_steps)
        rec["policies"]["fixed_matched"] = {
            "k": k, "mean_error": float(np.mean(errs)), "std_error": float(np.std(errs)),
            "cost_s": float(np.mean(costs)), "corrections": float(np.mean(corrs)),
            "steps": nt, "corr_frac": float(np.mean(corrs)) / nt,
            "hit_rate": float(np.mean([e <= target for e in errs])),
        }
        rows.append(rec)

    spread = {}
    for name in ["latch", "deadband", "naive", "hardcoded"]:
        e = [r_["policies"][name]["mean_error"] for r_ in rows]
        spread[name] = {"min": min(e), "max": max(e), "range": max(e) - min(e),
                        "responds_to_target": bool((max(e) - min(e)) > 0.005)}

    result = {"model": str(model), "steps": nt, "n_ics": len(idx),
              "hardcoded_theta": HARDCODED_THETA, "target_response": spread, "rows": rows}
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1)
    return result


def report(result):
    print("model=%s  steps=%d  ICs=%d" % (result["model"], result["steps"], result["n_ics"]))
    print()
    hdr = "%-8s %-14s %9s %9s %8s %8s %7s %7s" % (
        "target", "policy", "error", "cost_s", "corr", "corr%", "switch", "hit")
    print(hdr); print("-" * len(hdr))
    for r in result["rows"]:
        for name, p in r["policies"].items():
            print("%-8.3g %-14s %9.4f %9.3f %8.1f %7.0f%% %7s %6.0f%%" % (
                r["target"], name, p["mean_error"], p["cost_s"], p["corrections"],
                100 * p["corr_frac"], p.get("switch_events", "-"), 100 * p["hit_rate"]))
        print()
    sp = result.get("target_response", {})
    if sp:
        print("does the policy respond when you ask for a different accuracy?")
        for name, v in sp.items():
            print("  %-10s error spans %.4f -> %.4f  (range %.4f)  %s" % (
                name, v["min"], v["max"], v["range"],
                "RESPONDS" if v["responds_to_target"] else "FLAT - ignores the target"))
        print()
    print("latched = the controller never releases once it starts correcting")
    print("switch_events = how many separate times it engaged (1 means a single one-way handover)")


if __name__ == "__main__":
    import os
    d = os.path.join(config.RESULTS_DIR, "m3", "step11_switching_ablation")
    os.makedirs(d, exist_ok=True)
    res = main(out=os.path.join(d, "switching_ablation.json"))
    report(res)
    print("\nsaved to", os.path.join(d, "switching_ablation.json"))
