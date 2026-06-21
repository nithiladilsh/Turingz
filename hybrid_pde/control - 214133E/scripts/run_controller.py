#!/usr/bin/env python3
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MDIR = os.path.dirname(HERE); ROOT = os.path.abspath(os.path.join(MDIR, "..", ".."))
sys.path.insert(0, MDIR); sys.path.insert(0, ROOT)
from m3_cost import (groundtruth as G, trigger as TR, controller as CT,
                     surrogate as S, config as C)

UNIT_NUM = {"fdm": 0.795, "spectral": 18.94}


def run_ic(ic, u_ml, u_true, targets, solver="spectral"):
    fixed = CT.FixedController(UNIT_NUM, solver=solver, switch_time=C.T_TRAIN_END)
    oracle = CT.OracleController(UNIT_NUM, solver=solver)
    rows = []
    for eps in targets:
        jf, sf, _ = fixed.plan(eps)
        rf = CT.evaluate_policy(ic, u_ml, u_true, jf, sf, UNIT_NUM[sf], float("nan"))
        jo, so, _ = oracle.plan(eps, u_ml, u_true)
        ro = CT.evaluate_policy(ic, u_ml, u_true, jo, so, UNIT_NUM[so], float("nan"))
        rows.append({"eps": eps,
                     "fixed": {"j": jf, "cost": rf["n_num_steps"], "err": rf["err_extrap"],
                               "hit": rf["err_extrap"] <= eps},
                     "adaptive": {"j": jo, "cost": ro["n_num_steps"], "err": ro["err_extrap"],
                                  "hit": ro["err_extrap"] <= eps}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="deeponet-cached")
    ap.add_argument("--ics", type=int, nargs="*", default=None)
    args = ap.parse_args()
    ICs = G.make_ics()
    if args.kind == "deeponet-cached":
        sur = S.CachedSurrogate(C.DEEPONET_FIELD); idxs = sur.available_ics()
    else:
        sur = S.get_surrogate(args.kind, ROOT); idxs = args.ics
    targets = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.12]

    all_rows = {}
    for idx in idxs:
        u_ml = sur.predict_rollout(idx); u_true = G.colehopf_solve(ICs[idx])
        all_rows[idx] = run_ic(ICs[idx], u_ml, u_true, targets)

    r0 = all_rows[idxs[0]]
    print(f"surrogate={sur.name}  IC={idxs[0]}")
    print(" eps   | fixed cost/err/hit      | adaptive cost/err/hit   | cost saved")
    for row in r0:
        f, a = row["fixed"], row["adaptive"]
        saved = f["cost"] - a["cost"]
        print(f" {row['eps']:.2f} | {f['cost']:3d} / {f['err']:.3f} / {str(f['hit'])[:1]}"
              f"        | {a['cost']:3d} / {a['err']:.3f} / {str(a['hit'])[:1]}"
              f"       | {saved:+d} steps")

    out = {"surrogate": sur.name, "unit_num_ms": UNIT_NUM,
           "targets": targets, "per_ic": all_rows}
    jp = os.path.join(C.M3_RESULTS, f"controller_{args.kind}.json")
    json.dump(out, open(jp, "w"), indent=2); print("wrote", jp)
    _plot(r0, args.kind)


def _plot(rows, tag):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    eps = [r["eps"] for r in rows]
    fc = [r["fixed"]["cost"] for r in rows]
    ac = [r["adaptive"]["cost"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(eps, fc, "-s", color="#888", label="fixed (switch @ t=1)")
    ax.plot(eps, ac, "-o", color="#bf616a", label="adaptive (trust-gated)")
    ax.set_xlabel("requested accuracy target (extrap rel-L2)")
    ax.set_ylabel("numerical steps spent")
    ax.invert_xaxis()
    ax.set_title(f"Adaptive vs fixed scheduling - cost to hit target [{tag}]")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    pp = os.path.join(C.M3_RESULTS, f"controller_{tag}.png")
    fig.savefig(pp, dpi=140); print("wrote", pp)


if __name__ == "__main__":
    main()
