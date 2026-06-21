#!/usr/bin/env python3
import os, sys, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
MDIR = os.path.dirname(HERE); ROOT = os.path.abspath(os.path.join(MDIR, "..", ".."))
sys.path.insert(0, MDIR); sys.path.insert(0, ROOT)
from m3_cost import groundtruth as G, surrogate as S, runtime as RT, config as C

UNIT_NUM = {"fdm": 0.795, "spectral": 18.94}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="deeponet-cached")
    ap.add_argument("--ic", type=int, default=900)
    args = ap.parse_args()
    ICs = G.make_ics()
    if args.kind == "deeponet-cached":
        sur = S.CachedSurrogate(C.DEEPONET_FIELD); args.ic = sur.available_ics()[0]
    else:
        sur = S.get_surrogate(args.kind, ROOT)

    rt = RT.HybridRuntime(sur, UNIT_NUM, solver="spectral", trust_mode="oracle")
    u_true = G.colehopf_solve(ICs[args.ic])

    print(f"=== HybridRuntime live demo  (surrogate={sur.name}, IC {args.ic}) ===")
    print(" knob(target) | switch t | corrector | num_steps | wall_ms | achieved | met")
    reports = []
    for eps in [0.50, 0.35, 0.25, 0.20, 0.15]:
        rep = rt.run(args.ic, eps, u_true=u_true)
        reports.append(rep)
        print(f"   {eps:.2f}       |  {rep['switch_time']:.3f}  | {rep['corrector']:8s} "
              f"|   {rep['numerical_steps_spent']:3d}     | {rep['wall_ms']:6.0f} "
              f"|  {rep['achieved_extrap_error']:.3f}  | {rep['target_met']}")

    rep, field_u = rt.run(args.ic, 0.20, u_true=u_true, return_field=True)
    json.dump({"surrogate": sur.name, "ic": int(args.ic), "reports": reports},
              open(os.path.join(C.M3_RESULTS, f"demo_{args.kind}.json"), "w"), indent=2)
    _panel(ICs[args.ic], sur.predict_rollout(args.ic), field_u, u_true, rep, args.kind)


def _panel(ic, u_ml, u_hyb, u_true, rep, tag):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tfinal = -1
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(G.X, u_true[tfinal], "k-", lw=2, label="truth (Cole-Hopf)")
    ax.plot(G.X, u_ml[tfinal], color="#2e7d32", ls="--", label="pure ML")
    ax.plot(G.X, u_hyb[tfinal], color="#bf616a", lw=1.8, label="hybrid (M3)")
    ax.set_title(f"Solution at final time t=2.0  (target={rep['accuracy_target']})")
    ax.set_xlabel("x"); ax.set_ylabel("u"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax2 = axes[1]
    err_ml = G.rel_l2(u_ml, u_true); err_hy = G.rel_l2(u_hyb, u_true)
    ax2.plot(G.T_GRID, err_ml, color="#2e7d32", ls="--", label="pure ML error")
    ax2.plot(G.T_GRID, err_hy, color="#bf616a", lw=1.8, label="hybrid error")
    ax2.axvline(rep["switch_time"], color="#5e81ac", ls=":", label="switch (trust drop)")
    ax2.axvline(C.T_TRAIN_END, color="#999", ls="-", alpha=0.5, label="t_train_end")
    ax2.set_yscale("log"); ax2.set_xlabel("t"); ax2.set_ylabel("relative L2 error (log)")
    ax2.set_title(f"num_steps={rep['numerical_steps_spent']} | achieved={rep['achieved_extrap_error']:.3f}")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, which="both")
    fig.suptitle(f"M3 cost-aware hybrid runtime - one knob, measured cost & accuracy [{tag}]")
    fig.tight_layout()
    pp = os.path.join(C.M3_RESULTS, f"demo_{tag}.png")
    fig.savefig(pp, dpi=140); print("wrote", pp)


if __name__ == "__main__":
    main()
