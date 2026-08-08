import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plot_style import PALETTE, apply_app_style

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda p: json.load(open(os.path.join(ROOT, p), encoding="utf-8"))
OUT = os.path.join(ROOT, "results", "m3", "step9d_coarse_integration", "fig_6_5_1_frontier.png")

apply_app_style()

d = R("results/m3/step9d_coarse_integration/timed_cost_result_m2.json")
ml, num = d["pure_ml"], d["pure_numerical"]
fr = sorted(d["frontier"], key=lambda r: r["cost_s"])

for r in fr:
    r["acc_x"] = ml["mean_error"] / r["mean_error"]
    r["cost_x"] = num["cost_s"] / r["cost_s"]

fig, (ax, tax) = plt.subplots(2, 1, figsize=(7.6, 6.7), height_ratios=[3.1, 1.6])

xmax = num["cost_s"] * 1.12
ymin, ymax = 8e-3, 12
ax.set_xlim(0, xmax)
ax.set_ylim(ymin, ymax)
ax.set_yscale("log")

ax.axhspan(ymin, 100 * ml["mean_error"], xmin=0, xmax=num["cost_s"] / xmax,
           color=PALETTE["emerald"], alpha=0.06, zorder=0)
ax.axhline(100 * ml["mean_error"], color=PALETTE["rose"], lw=1.3, ls="--", zorder=1)
ax.axvline(num["cost_s"], color=PALETTE["indigo"], lw=1.3, ls="--", zorder=1)

ax.errorbar([r["cost_s"] for r in fr], [100 * r["mean_error"] for r in fr],
            yerr=[100 * r["std_error"] for r in fr], xerr=[r["cost_std"] for r in fr],
            marker="o", ms=7, lw=2, capsize=3, color=PALETTE["emerald"],
            label="Hybrid (adaptive controller)", zorder=3)

ax.scatter([ml["cost_s"]], [100 * ml["mean_error"]], marker="s", s=90,
           color=PALETTE["rose"], label="Pure ML (FNO)", zorder=4)
ax.scatter([num["cost_s"]], [100 * num["mean_error"]], marker="^", s=100,
           color=PALETTE["indigo"], label="Pure numerical (spectral)", zorder=4)

ax.text(0.02, 100 * ml["mean_error"] * 1.15, "Pure ML (FNO)", color=PALETTE["rose"],
        fontsize=10, fontweight="bold", va="bottom")
ax.text(num["cost_s"] - 0.03, ymin * 1.4, "Pure numerical\n(spectral)", color=PALETTE["indigo"],
        fontsize=10, fontweight="bold", ha="right", va="bottom")

ax.text(num["cost_s"] * 0.42, 0.35, "cheaper than the numerical solver\nand more accurate than the ML surrogate",
        color=PALETTE["emerald"], fontsize=9.5, style="italic", ha="center", va="center")

ax.annotate("tighter target", xy=(fr[-1]["cost_s"] - 0.05, 100 * fr[-1]["mean_error"] - 0.35),
            xytext=(fr[1]["cost_s"] - 0.08, 100 * fr[1]["mean_error"] - 1.7),
            fontsize=9, style="italic", color=PALETTE["emerald"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["emerald"], lw=1.3,
                             connectionstyle="arc3,rad=0.25"))

ax.set_xlabel("mean wall-clock cost per problem (s)")
ax.set_ylabel("mean relative $L_2$ error (%, log scale)")
ax.legend(loc="upper right", framealpha=0.95, fontsize=9)

# ---- per-point multiples table, drawn in its own panel below so the curve stays uncluttered ----
tax.axis("off")
tax.set_title("each hybrid point, against both pure baselines", fontsize=10.5, fontweight="bold",
              color=PALETTE["ink"], loc="left", pad=10)

rows_by_target = {}
for r in fr:
    rows_by_target.setdefault(r["target"], r)
targets_ordered = [0.3, 0.2, 0.1, 0.05, 0.02, 0.01]

col_x = [0.0, 0.17, 0.40, 0.62, 0.86]
headers = ["target", "cost (s)", "error", "vs. pure ML", "vs. pure numerical"]
y0 = 0.92
row_h = 0.135
for cx, h in zip(col_x, headers):
    tax.text(cx, y0, h, fontsize=9.2, fontweight="bold", color=PALETTE["axis"], transform=tax.transAxes)
tax.plot([0, 1], [y0 - 0.05, y0 - 0.05], color=PALETTE["grid"], lw=1.2, transform=tax.transAxes)

for i, t in enumerate(targets_ordered):
    r = rows_by_target[t]
    y = y0 - 0.05 - row_h * (i + 1)
    vals = [f"{t:g}", f"{r['cost_s']:.2f}", f"{100*r['mean_error']:.2f}%",
            f"{r['acc_x']:.1f}× more accurate", f"{r['cost_x']:.1f}× cheaper"]
    colors = [PALETTE["ink"], PALETTE["ink"], PALETTE["ink"], PALETTE["rose"], PALETTE["indigo"]]
    for cx, v, c in zip(col_x, vals, colors):
        tax.text(cx, y, v, fontsize=9.2, color=c, transform=tax.transAxes)

tax.text(0.0, y0 - 0.05 - row_h * 7.1,
         "\"vs. pure ML\" = pure-ML error ÷ hybrid error at that target (how much more accurate). "
         "\"vs. pure numerical\" = numerical-only cost ÷ hybrid cost (how much cheaper). Real measured runs, not estimated.",
         fontsize=7.6, color=PALETTE["axis"], style="italic", transform=tax.transAxes, wrap=True)

fig.tight_layout()
fig.savefig(OUT)
plt.close(fig)

print("saved", OUT)
for r in fr:
    print(f"target {r['target']:>4} : {r['acc_x']:.2f}x more accurate than pure ML, "
          f"{r['cost_x']:.2f}x cheaper than pure numerical")
