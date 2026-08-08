import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from _load_pt_no_torch import load  # noqa: E402

OUT_DIR = os.path.join(ROOT, "results", "m3", "ic_representativeness")
HELD_OUT_10 = np.arange(900, 910)
HELD_OUT_20 = np.arange(900, 920)


def features(ICs, x):
    energy = np.linalg.norm(ICs, axis=1)
    roughness = np.sum(np.abs(np.diff(ICs, axis=1)), axis=1)
    zero_cross = np.sum(np.diff(np.sign(ICs), axis=1) != 0, axis=1).astype(float)

    F = np.fft.rfft(ICs, axis=1)
    P = np.abs(F[:, 1:5]) ** 2
    mode_idx = np.arange(1, 5)
    centroid = (P * mode_idx).sum(axis=1) / (P.sum(axis=1) + 1e-12)

    return {
        "energy": energy,
        "roughness_tv": roughness,
        "zero_crossings": zero_cross,
        "spectral_centroid_modes1_4": centroid,
    }


def percentile_rank(full_vals, subset_vals):
    return [float((full_vals < v).mean() * 100) for v in subset_vals]


def summarize(feat_full, idx, label):
    out = {}
    for name, full_vals in feat_full.items():
        sub_vals = full_vals[idx]
        ranks = percentile_rank(full_vals, sub_vals)
        out[name] = {
            "full_mean": float(full_vals.mean()),
            "full_std": float(full_vals.std()),
            "full_min": float(full_vals.min()),
            "full_max": float(full_vals.max()),
            "subset_mean": float(sub_vals.mean()),
            "subset_std": float(sub_vals.std()),
            "subset_percentile_ranks": [round(r, 1) for r in ranks],
        }
    return out


def plot(feat_full, idx10, idx20):
    names = ["energy", "roughness_tv", "zero_crossings", "spectral_centroid_modes1_4"]
    titles = ["Energy (L2 norm)", "Roughness (total variation)",
              "Zero crossings", "Spectral centroid (modes 1-4)"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, name, title in zip(axes.flat, names, titles):
        full_vals = feat_full[name]
        ax.hist(full_vals, bins=40, color="#8fb3d9", edgecolor="white", alpha=0.85,
                 label="all 1000 ICs")
        ax.scatter(full_vals[idx10], np.full(len(idx10), ax.get_ylim()[1] * 0.02),
                   color="#c0392b", marker="v", s=60, zorder=5, label="held-out 10 (900-909)")
        extra20 = np.setdiff1d(idx20, idx10)
        if len(extra20):
            ax.scatter(full_vals[extra20], np.full(len(extra20), ax.get_ylim()[1] * 0.02),
                       color="#c1841a", marker="v", s=45, zorder=4, label="extended 20 (910-919)")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Held-out test ICs (900-909 / 900-919) vs. the full 1000-IC distribution", fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "ic_representativeness.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    d = load()
    ICs, x = d["ICs"], d["x"]
    feat_full = features(ICs, x)

    summary_10 = summarize(feat_full, HELD_OUT_10, "held_out_10")
    summary_20 = summarize(feat_full, HELD_OUT_20, "held_out_20_extended")

    for name, s in summary_10.items():
        print(f"{name}: full mean={s['full_mean']:.4f} std={s['full_std']:.4f} | "
              f"subset(10) mean={s['subset_mean']:.4f} std={s['subset_std']:.4f}")
        print(f"    subset(10) percentile ranks within full 1000: {s['subset_percentile_ranks']}")

    plot(feat_full, HELD_OUT_10, HELD_OUT_20)

    result = {
        "purpose": "Checks whether the held-out ICs used in M3's analysis (900-909, and "
                   "the extended 900-919) are representative of the full 1000-IC dataset, "
                   "by comparing waveform-shape features (energy, amplitude, roughness, "
                   "spectral content) against the full distribution.",
        "held_out_10_ids": HELD_OUT_10.tolist(),
        "held_out_20_ids": HELD_OUT_20.tolist(),
        "held_out_10": summary_10,
        "held_out_20_extended": summary_20,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "ic_representativeness.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
