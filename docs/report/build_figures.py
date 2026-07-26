import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs", "report", "figures")
R = lambda p: json.load(open(os.path.join(ROOT, p), encoding="utf-8"))

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 10.5, "axes.titlesize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":",
})
C = {"latch": "#1a7a4a", "deadband": "#4f46e5", "naive": "#b45309",
     "hardcoded": "#b91c1c", "fixed_matched": "#6b7280",
     "FNO": "#1a7a4a", "DeepONet": "#b91c1c", "PINN": "#b45309",
     "ml": "#b91c1c", "num": "#4f46e5", "hyb": "#1a7a4a"}


def fig1_frontier():
    d = R("results/m3/step9d_coarse_integration/timed_cost_result_m2.json")
    ml, num = d["pure_ml"], d["pure_numerical"]
    fr = sorted(d["frontier"], key=lambda r: r["cost_s"])
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.errorbar([r["cost_s"] for r in fr], [100 * r["mean_error"] for r in fr],
                yerr=[100 * r["std_error"] for r in fr],
                xerr=[r["cost_std"] for r in fr],
                marker="o", ms=6, lw=1.8, capsize=3, color=C["hyb"],
                label="Hybrid (adaptive controller)", zorder=3)
    for r in fr:
        if r["target"] in (0.3, 0.05, 0.01):
            ax.annotate(f"target {r['target']:g}", (r["cost_s"], 100 * r["mean_error"]),
                        textcoords="offset points", xytext=(8, 6), fontsize=8.5, color=C["hyb"])
    ax.scatter([ml["cost_s"]], [100 * ml["mean_error"]], marker="s", s=70,
               color=C["ml"], label="Pure ML (FNO)", zorder=3)
    ax.scatter([num["cost_s"]], [100 * num["mean_error"]], marker="^", s=80,
               color=C["num"], label="Pure numerical (spectral)", zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("mean wall-clock cost per problem (s)")
    ax.set_ylabel("mean relative $L_2$ error (%)")
    ax.legend(loc="center right", framealpha=0.95)
    fig.savefig(os.path.join(OUT, "fig_6_5_1_frontier.png"))
    plt.close(fig)


def fig2_ablation():
    d = R("results/m3/step11_switching_ablation/switching_ablation.json")
    targets = [r["target"] for r in d["rows"]]
    pols = ["latch", "hardcoded", "naive", "deadband", "fixed_matched"]
    label = {"latch": "Adaptive (ours)", "hardcoded": "Fixed threshold (hand-tuned 0.4)",
             "naive": "Single threshold from map", "deadband": "Two-threshold hysteresis",
             "fixed_matched": "Fixed interval (cost-matched)"}
    style = {"latch": "-o", "hardcoded": "-s", "naive": "--^", "deadband": "--v", "fixed_matched": "-D"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for p in pols:
        errs = [100 * r["policies"][p]["mean_error"] for r in d["rows"]]
        hits = [100 * r["policies"][p]["hit_rate"] for r in d["rows"]]
        a1.plot(targets, errs, style[p], color=C[p], ms=5, lw=1.6, label=label[p])
        a2.plot(targets, hits, style[p], color=C[p], ms=5, lw=1.6)
    a1.set_xscale("log"); a1.invert_xaxis()
    a1.set_xlabel("requested accuracy target (tighter $\\rightarrow$)")
    a1.set_ylabel("achieved error (%)")
    a1.legend(framealpha=0.95, fontsize=8)
    a2.set_xscale("log"); a2.invert_xaxis()
    a2.set_xlabel("requested accuracy target (tighter $\\rightarrow$)")
    a2.set_ylabel("target hit-rate (%)")
    a2.set_ylim(-5, 105)
    fig.savefig(os.path.join(OUT, "fig_6_5_2_ablation.png"))
    plt.close(fig)


def fig3_regime():
    f = R("results/m3/step9d_coarse_integration/timed_cost_result_m2.json")
    g = R("results/m3/step9d_coarse_integration/timed_cost_result_deeponet.json")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for d, name in ((f, "FNO"), (g, "DeepONet")):
        num = d["pure_numerical"]["cost_s"]
        fr = sorted(d["frontier"], key=lambda r: r["cost_s"])
        ax.plot([r["cost_s"] / num for r in fr], [100 * r["mean_error"] for r in fr],
                "-o" if name == "FNO" else "-s", color=C[name], ms=6, lw=1.8,
                label=f"Hybrid with {name}")
    ax.axvline(1.0, color=C["num"], lw=1.5, ls="--")
    ax.text(1.05, 0.95, "pure numerical cost", rotation=90, va="top", fontsize=8.5, color=C["num"],
            transform=ax.get_xaxis_transform())
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("hybrid cost relative to that session's numerical solver")
    ax.set_ylabel("mean relative $L_2$ error (%)")
    ax.legend(framealpha=0.95)
    fig.savefig(os.path.join(OUT, "fig_6_5_3_regime.png"))
    plt.close(fig)


def fig4_pinn():
    d = R("results/m3/step12_pinn_regime/pinn_controller.json")
    targets = [r["target"] for r in d["rows"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(targets, [100 * r["mean_error"] for r in d["rows"]], "-o", color=C["PINN"],
            ms=6, lw=1.8, label="Controlled PINN error (training cost excluded)")
    ax.axhline(35.8, color=C["ml"], lw=1.4, ls="--", label="Standalone PINN extrapolation (35.8%)")
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel("requested accuracy target (tighter $\\rightarrow$)")
    ax.set_ylabel("achieved error (%)")
    ax.set_ylim(0, 42)
    ax2 = ax.twinx()
    ax2.bar(targets, [100 * r["hit_rate"] for r in d["rows"]],
            width=[t * 0.25 for t in targets], color=C["PINN"], alpha=0.25, label="hit-rate")
    ax2.set_ylabel("target hit-rate (%)", color="#8a6d3b")
    ax2.set_ylim(0, 105); ax2.grid(False)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=8.5)
    fig.savefig(os.path.join(OUT, "fig_6_5_3b_pinn.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig1_frontier(); fig2_ablation(); fig3_regime(); fig4_pinn()
    print("wrote 4 figures to", OUT)
