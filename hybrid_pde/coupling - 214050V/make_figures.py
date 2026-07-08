"""
Figure generation for the ML->numerical handoff study (Module 2, Coupling).
Recomputes hybrid/upper-bound/pure trajectories from results/eval/predictions.npz
using the restart wrapper, then renders the five core figures.

Stages:  python make_figures.py compute   # heavy: restarts, caches arrays
         python make_figures.py plot      # fast: renders PNGs from cache
         python make_figures.py all       # both
Outputs: results/module2/figures/*.png
"""
import os, sys, json
import numpy as np
from restart_spectral import solve_from, nearest_index, TGRID

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRED = os.path.join(ROOT, "results", "eval", "predictions.npz")
FIGDIR = os.path.join(ROOT, "results", "module2", "figures")
CACHE = os.path.join(FIGDIR, "_figure_data.npz")
SWITCH_TIMES = [1.0, 1.2, 1.4, 1.6, 1.8]
FIG1_TS = 1.0
EPS = 1e-12


def rl2(U, Uref):            # rel-L2 along last axis (space)
    return np.sqrt(((U - Uref) ** 2).sum(-1)) / (np.sqrt((Uref ** 2).sum(-1)) + EPS)


def spectral_distance(a, b, alpha=1.0):   # shape (Fourier-amplitude) distance
    A = np.abs(np.fft.rfft(a, axis=-1)); B = np.abs(np.fft.rfft(b, axis=-1))
    k = np.arange(A.shape[-1]); w = (1.0 + k) ** alpha
    return np.sqrt((w * (A - B) ** 2).sum(-1)) / (np.sqrt((w * B ** 2).sum(-1)) + EPS)


def integ(curve, times):     # time-averaged error
    return np.trapezoid(curve, times) / (times[-1] - times[0] + EPS)


def compute():
    os.makedirs(FIGDIR, exist_ok=True)
    d = np.load(PRED)
    t = d["t"]; Utrue = d["u_true_eval"].astype(np.float64); Ufno = d["FNO_eval"].astype(np.float64)
    n_ic, nt, nx = Utrue.shape
    assert np.allclose(t, TGRID)

    # per-(t_s, IC) tail metrics
    es   = np.zeros((len(SWITCH_TIMES), n_ic))
    sd_es = np.zeros((len(SWITCH_TIMES), n_ic))
    fno  = np.zeros_like(es); hyb = np.zeros_like(es); ub = np.zeros_like(es)
    idx  = []
    numfrac = []
    fig1 = {}
    for s, ts in enumerate(SWITCH_TIMES):
        i = nearest_index(ts); idx.append(i); numfrac.append((nt - i) / nt)
        tail_t = t[i:]; true_tail = Utrue[:, i:]
        UB = solve_from(Utrue[:, i], i)      # (n_ic, tail, nx)
        H  = solve_from(Ufno[:,  i], i)
        for j in range(n_ic):
            es[s, j]  = np.linalg.norm(Ufno[j, i] - Utrue[j, i]) / (np.linalg.norm(Utrue[j, i]) + EPS)
            sd_es[s, j] = spectral_distance(Ufno[j, i], Utrue[j, i])
            fno[s, j] = integ(rl2(Ufno[j, i:], true_tail[j]), tail_t)
            hyb[s, j] = integ(rl2(H[j],        true_tail[j]), tail_t)
            ub[s, j]  = integ(rl2(UB[j],       true_tail[j]), tail_t)
        if abs(ts - FIG1_TS) < 1e-9:
            fig1["i"] = i; fig1["H"] = H            # keep hybrid tail for the time-series figure

    # Fig 1 full-time curves at FIG1_TS: FNO, hybrid, numerical-alone
    NUM = solve_from(Utrue[:, 0], 0)                # numerical from the true IC (numerical-alone)
    fno_curve = rl2(Ufno, Utrue)                    # (n_ic, nt)
    num_curve = rl2(NUM,  Utrue)
    i1 = fig1["i"]; H1 = fig1["H"]
    hyb_curve = fno_curve.copy()
    for j in range(n_ic):
        hyb_curve[j, i1:] = rl2(H1[j], Utrue[j, i1:])

    # also write a clean summary table (supersedes the old sweep json)
    b = (fno - hyb) / (fno + EPS)
    rows = [dict(t_s=float(SWITCH_TIMES[s]), idx=int(idx[s]),
                 numerical_fraction=float(numfrac[s]),
                 e_s=float(es[s].mean()), fno_tail=float(fno[s].mean()),
                 hybrid_tail=float(hyb[s].mean()), upper_bound_tail=float(ub[s].mean()),
                 benefit=float(b[s].mean()), ics_improved=int((hyb[s] < fno[s]).sum()))
            for s in range(len(SWITCH_TIMES))]
    # a-priori viability rule (frozen BEFORE looking at results):
    #   viable if mean benefit >= 10% AND mean absolute hybrid tail error < E_MAX
    E_MAX = 0.10
    bmean = b.mean(1); hmean = hyb.mean(1)
    viable = [(bmean[s] >= 0.10 and hmean[s] < E_MAX) for s in range(len(SWITCH_TIMES))]
    # boundary: interpolate t_s where hybrid tail error crosses E_MAX
    boundary = None
    for s in range(len(SWITCH_TIMES) - 1):
        if hmean[s] < E_MAX <= hmean[s+1]:
            f = (E_MAX - hmean[s]) / (hmean[s+1] - hmean[s])
            boundary = float(SWITCH_TIMES[s] + f * (SWITCH_TIMES[s+1] - SWITCH_TIMES[s]))
    for s in range(len(rows)):
        rows[s]["sd_at_handoff"] = float(sd_es[s].mean())
        rows[s]["viable"] = bool(viable[s])
    print("\nViability rule (frozen a priori): benefit >= 10%% AND hybrid tail < %.2f" % E_MAX)
    for s in range(len(SWITCH_TIMES)):
        print("   t_s=%.1f  benefit=%.2f  hybrid_tail=%.3f  ->  %s" %
              (SWITCH_TIMES[s], bmean[s], hmean[s], "VIABLE" if viable[s] else "not viable"))
    print("   viability boundary (hybrid tail hits %.0f%%): t_s ~ %s   [FNO reliable horizon = 1.457]"
          % (E_MAX*100, ("%.2f" % boundary) if boundary else "n/a"))
    summary = {"status": "PRELIMINARY -- not final thesis results",
               "viability_rule": {"benefit_min": 0.10, "abs_error_max": E_MAX,
                                  "boundary_t_s": boundary, "fno_reliable_horizon": 1.457},
               "n_ic": int(n_ic), "handoff": "raw FNO state (no filtering)",
               "reference": "Cole-Hopf (u_true_eval)",
               "continuation_solver": "restart wrapper (team-scheme port; equivalence to be verified)",
               "switch_time_source": "externally supplied",
               "metric": "time-integrated relative L2 over [t_s, T]",
               "results": rows}
    json.dump(summary, open(os.path.join(FIGDIR, "handoff_sweep_results.json"), "w"), indent=2)

    np.savez(CACHE, t=t, ts=np.array(SWITCH_TIMES), idx=np.array(idx),
             numfrac=np.array(numfrac), es=es, sd_es=sd_es, fno=fno, hyb=hyb, ub=ub,
             fno_curve=fno_curve, hyb_curve=hyb_curve, num_curve=num_curve,
             i1=i1, fig1_ts=FIG1_TS, n_ic=n_ic)
    print("cached ->", CACHE)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = np.load(CACHE)
    t = c["t"]; ts = c["ts"]; es = c["es"]; fno = c["fno"]; hyb = c["hyb"]; ub = c["ub"]
    numfrac = c["numfrac"]; n_ic = int(c["n_ic"]); i1 = int(c["i1"]); fig1_ts = float(c["fig1_ts"])
    sd_es = c["sd_es"]
    C = dict(fno="#d1495b", hyb="#2e7d32", num="#1f6feb", ub="#8a8d91")

    # ---- Figure 1: error over time (mean +/- std across ICs) ----
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for key, arr, lab in [("fno", c["fno_curve"], "FNO alone"),
                          ("num", c["num_curve"], "Numerical alone"),
                          ("hyb", c["hyb_curve"], "Hybrid (handoff)")]:
        m = arr.mean(0); sd = arr.std(0)
        ax.plot(t, m, color=C[key], lw=2, label=lab)
        ax.fill_between(t, m - sd, m + sd, color=C[key], alpha=0.15)
    ax.axvline(fig1_ts, ls="--", color="k", lw=1); ax.axvline(1.0, ls=":", color="grey", lw=0.8)
    ax.text(fig1_ts + 0.02, ax.get_ylim()[1]*0.9, "handoff", fontsize=9)
    ax.set(xlabel="time t", ylabel="relative L2 error",
           title=f"Error over time (mean ± std, {n_ic} held-out ICs), handoff at t={fig1_ts}")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig1_error_over_time.png"), dpi=140); plt.close(fig)

    # ---- Figure 2: switch time vs integrated benefit ----
    b = (fno - hyb) / (fno + EPS)                    # per-IC benefit
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.errorbar(ts, b.mean(1), yerr=b.std(1), fmt="o-", color=C["hyb"], capsize=4, lw=2)
    ax.axhline(0.10, ls="--", color="grey"); ax.text(ts[0], 0.11, "viability threshold (10%)", fontsize=9)
    ax.set(xlabel="handoff time t_s", ylabel="mean benefit  (1 − hybrid/FNO)",
           title="Handoff benefit vs when we switch")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig2_switch_time_vs_benefit.png"), dpi=140); plt.close(fig)

    # ---- Figure 3: handoff-state error vs benefit (one point per IC per t_s) ----
    fig, ax = plt.subplots(figsize=(7, 4.3))
    sc = ax.scatter(es.ravel(), b.ravel(), c=np.repeat(ts, es.shape[1]),
                    cmap="viridis", s=40, edgecolor="k", linewidth=0.3)
    ax.axhline(0.10, ls="--", color="grey")
    cb = fig.colorbar(sc); cb.set_label("handoff time t_s")
    ax.set(xlabel="FNO state error at handoff  e_s", ylabel="benefit  (1 − hybrid/FNO)",
           title="The worse the handed-over state, the smaller the benefit")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig3_handoff_error_vs_benefit.png"), dpi=140); plt.close(fig)

    # ---- Figure 4: hybrid vs upper bound vs pure FNO (tail error) ----
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(ts, fno.mean(1), "o-", color=C["fno"], lw=2, label="Pure FNO")
    ax.plot(ts, hyb.mean(1), "s-", color=C["hyb"], lw=2, label="Hybrid (from FNO state)")
    ax.plot(ts, ub.mean(1),  "^-", color=C["ub"],  lw=2, label="Upper bound (from TRUE state)")
    ax.plot(ts, es.mean(1),  "x--", color="k", lw=1, alpha=0.7, label="FNO handoff-state error e_s")
    ax.set_yscale("log")
    ax.set(xlabel="handoff time t_s", ylabel="tail error (log scale)",
           title="Hybrid tail error ≈ inherited state error; numerical part adds ~none")
    ax.legend(); ax.grid(alpha=0.3, which="both"); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig4_hybrid_vs_upper_bound.png"), dpi=140); plt.close(fig)

    # ---- Figure 5: accuracy vs cost (numerical fraction) ----
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(numfrac, hyb.mean(1), "o-", color=C["hyb"], lw=2)
    for xf, ts_, hy in zip(numfrac, ts, hyb.mean(1)):
        ax.annotate(f"t_s={ts_:.1f}", (xf, hy), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set(xlabel="fraction of trajectory solved numerically (cost proxy)",
           ylabel="hybrid tail error",
           title="Accuracy vs cost: earlier handoff = more numerical work, lower error")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig5_accuracy_vs_cost.png"), dpi=140); plt.close(fig)
    # ---- Figure 6: shape (spectral) distance at handoff vs benefit ----
    fig, ax = plt.subplots(figsize=(7, 4.3))
    sc = ax.scatter(sd_es.ravel(), b.ravel(), c=np.repeat(ts, sd_es.shape[1]),
                    cmap="plasma", s=40, edgecolor="k", linewidth=0.3)
    ax.axhline(0.10, ls="--", color="grey")
    cb = fig.colorbar(sc); cb.set_label("handoff time t_s")
    ax.set(xlabel="shape (spectral) distance of FNO wave at handoff",
           ylabel="benefit  (1 - hybrid/FNO)",
           title="Shape drift at handoff also predicts the benefit")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig6_spectral_distance_vs_benefit.png"), dpi=140); plt.close(fig)
    print("wrote 6 figures ->", FIGDIR)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("compute", "all"): compute()
    if stage in ("plot", "all"):    plot()
