#!/usr/bin/env python3
import os, sys, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
MDIR = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(MDIR, "..", ".."))
sys.path.insert(0, MDIR)
sys.path.insert(0, ROOT)
from m3_cost import (groundtruth as G, profiler as P, surrogate as S,
                     accuracy_cost as AC, config as C)


def get_surrogate_and_ics(kind, ics):
    if kind == "deeponet-cached":
        sur = S.CachedSurrogate(C.DEEPONET_FIELD)
        return sur, sur.available_ics()
    sur = S.get_surrogate(kind, ROOT)
    return sur, ics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="deeponet-cached")
    ap.add_argument("--ics", type=int, nargs="*", default=None)
    ap.add_argument("--ref-solver", default="spectral")
    args = ap.parse_args()

    ICs = G.make_ics()
    sur, ic_indices = get_surrogate_and_ics(args.kind, args.ics)
    print(f"surrogate={sur.name}  ICs={ic_indices}")

    units = P.profile_numerical_units(G, ICs[ic_indices[0]],
                                      step_repeats=9, full_repeats=2)
    unit_num = {u["op"].split(".")[0]: u["median_ms"]
                for u in units if u["op"].endswith(".step")}
    print("measured numerical step costs (ms):", {k: round(v, 3) for k, v in unit_num.items()})
    ml_step = P.profile_ml_step(sur)
    unit_ml = ml_step["median_ms"] if ml_step else float("nan")
    print("measured ML step cost (ms):", round(unit_ml, 4) if ml_step else "n/a (cached)")

    per_ic = []
    for idx in ic_indices:
        ic = ICs[idx]
        u_ml = sur.predict_rollout(idx)
        u_true = G.colehopf_solve(ic)
        cloud = AC.policy_cloud(ic, u_ml, u_true, unit_num, unit_ml=unit_ml)
        base = AC.baselines(ic, u_ml, u_true, unit_num, unit_ml=unit_ml,
                            ref_solver=args.ref_solver)
        front = AC.pareto_front(cloud)
        dom = AC.dominates_baselines(front, base)
        per_ic.append({"ic": int(idx), "cloud": cloud, "front": front,
                       "baselines": base, "dominance": dom})
        print(f" IC {idx}: ML extrap={base['pure_ml']['err_extrap']:.3f} | "
              f"speedup@beat-ml-2x={dom['speedup_to_beat_ml_2x']} | "
              f"floor={dom['hybrid_floor_acc']:.3f}")

    tag = args.kind
    out = {"surrogate": sur.name, "unit_num_ms": unit_num, "unit_ml_ms": unit_ml,
           "unit_cost_profile": units, "per_ic": per_ic}
    jpath = os.path.join(C.M3_RESULTS, f"frontier_{tag}.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", jpath)
    _plot(per_ic, tag)


def _plot(per_ic, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = per_ic[0]
    fig, ax = plt.subplots(figsize=(7, 5))
    cloud = d["cloud"]
    for solver, color in [("fdm", "#d08770"), ("spectral", "#5e81ac")]:
        xs = [r["numerical_step_equivalents"] for r in cloud if r["solver"] == solver]
        ys = [r["err_extrap"] for r in cloud if r["solver"] == solver]
        ax.scatter(xs, ys, s=18, alpha=0.4, color=color, label=f"{solver} policies")
    fx = [r["numerical_step_equivalents"] for r in d["front"]]
    fy = [r["err_extrap"] for r in d["front"]]
    ax.plot(fx, fy, "-o", color="#bf616a", lw=2, ms=5, label="hybrid Pareto frontier")
    b = d["baselines"]
    ax.scatter([b["pure_ml"]["numerical_step_equivalents"]], [b["pure_ml"]["err_extrap"]],
               marker="*", s=260, color="#2e7d32", zorder=5, label="pure ML")
    ax.scatter([b["pure_numerical"]["numerical_step_equivalents"]], [b["pure_numerical"]["err_extrap"]],
               marker="D", s=90, color="#000000", zorder=5, label="pure numerical")
    ax.set_yscale("log")
    ax.set_xlabel("cost  (numerical steps spent)")
    ax.set_ylabel("extrapolation relative L2 error (log)")
    ax.set_title(f"Cost vs accuracy - hybrid frontier  [{tag}, IC {d['ic']}]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    ppath = os.path.join(C.M3_RESULTS, f"frontier_{tag}.png")
    fig.savefig(ppath, dpi=140)
    print("wrote", ppath)


if __name__ == "__main__":
    main()
