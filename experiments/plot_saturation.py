import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "results", "m3", "threshold_calibration")
SATURATION_TARGET = 0.029


def main():
    with open(os.path.join(ROOT, "results", "m3", "achievability", "m3_achievability.json")) as f:
        data = json.load(f)

    rows = sorted(data["per_target"], key=lambda r: r["target"])
    targets = [r["target"] for r in rows]
    errors = [r["mean_error"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(targets, errors, marker="o", color="#2b5b84", linewidth=2, markersize=7,
            label="achieved error (measured, 10 held-out waves)")
    ax.axvspan(0.02, 0.05, color="#f8dfa0", alpha=0.5,
               label="saturation region (between measured targets 0.05 and 0.02)")
    ax.axvline(SATURATION_TARGET, color="#7a3a9c", linestyle="--", linewidth=1.5,
               label=f"accuracy-saturation reference point = {SATURATION_TARGET}")
    ax.set_xlabel("requested accuracy target")
    ax.set_ylabel("achieved mean error (%)")
    ax.set_title("Achieved error vs. requested target: saturation between 0.05 and 0.02")
    ax.invert_xaxis()
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out = os.path.join(OUT_DIR, "saturation_point.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
