from __future__ import annotations
import json, os
import numpy as np
import core
from pipeline import _cc_setup, cole_hopf_from, _NumSolver, _per_step_rates
from hybrid_pde.coupling_214050V.m2_coupling import M2Coupling

TARGETS = [0.3, 0.2, 0.1, 0.05, 0.02, 0.01]


def run_one(model, target, pinn_index=0, ic=None):
    ic0, pred, true, mon, ctrl, lo, hi = _cc_setup(model, ic, pinn_index, target)
    X, T = core.X, core.T
    nt = len(T)
    ml_per, num_per = _per_step_rates(nt)
    cp, num = M2Coupling(), _NumSolver()
    st = np.asarray(ic0, float)
    prev = float(T[0])
    corr = 0
    cost = 0.0
    for n in range(nt):
        o = mon.update(st, float(T[n]))
        d = ctrl.decide(float(o["trust"]), not bool(o["ok"]), float(T[n]), n)
        if d.correct:
            st = np.asarray(cp.correct(st, X, prev, float(T[n]), num), float)
            corr += 1
            cost += num_per
        else:
            st = np.asarray(pred[n], float)
            cost += ml_per
        prev = float(T[n])
    err = float(np.linalg.norm(st - true[-1]) / (np.linalg.norm(true[-1]) + 1e-12))
    return {"error": err, "cost_s": cost, "corrections": corr, "steps": nt, "hit": bool(err <= target)}


def main(out=None):
    n_ics = int(len(core.PINN_ICS))
    rows = []
    for target in TARGETS:
        runs = [run_one("PINN", target, pinn_index=i) for i in range(n_ics)]
        rows.append({
            "target": target,
            "mean_error": float(np.mean([r["error"] for r in runs])),
            "std_error": float(np.std([r["error"] for r in runs])),
            "cost_s": float(np.mean([r["cost_s"] for r in runs])),
            "corrections": float(np.mean([r["corrections"] for r in runs])),
            "corr_frac": float(np.mean([r["corrections"] for r in runs])) / runs[0]["steps"],
            "hit_rate": float(np.mean([r["hit"] for r in runs])),
        })
    e = [r["mean_error"] for r in rows]
    res = {"model": "PINN", "n_ics": n_ics, "basis": "pre-trained checkpoints, training cost EXCLUDED",
           "rows": rows,
           "responds_to_target": bool(max(e) - min(e) > 0.005),
           "error_range": [min(e), max(e)],
           "best_hit_rate": max(r["hit_rate"] for r in rows)}
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
    return res


def report(res):
    print("PINN through the controller  -  %d pre-trained ICs, TRAINING COST EXCLUDED" % res["n_ics"])
    print("(this is the most favourable possible treatment: the 2114 s retrain is a gift)")
    print("=" * 70)
    print("%-10s %10s %10s %8s %8s" % ("target", "error", "cost_s", "corr%", "hit"))
    print("-" * 70)
    for r in res["rows"]:
        print("%-10.3g %9.2f%% %10.3f %7.0f%% %7.0f%%" % (
            r["target"], 100 * r["mean_error"], r["cost_s"], 100 * r["corr_frac"], 100 * r["hit_rate"]))
    print("-" * 70)
    print("error range %.2f%% - %.2f%%   best hit-rate %.0f%%   %s" % (
        100 * res["error_range"][0], 100 * res["error_range"][1], 100 * res["best_hit_rate"],
        "RESPONDS to target" if res["responds_to_target"] else "FLAT - knob has no effect"))


if __name__ == "__main__":
    p = os.path.join(core.ROOT, "results", "m3", "step12_pinn_regime", "pinn_controller.json")
    res = main(out=p)
    report(res)
    print("\nsaved to", p)
