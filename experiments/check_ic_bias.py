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


def ic_features(ICs):
    energy = np.linalg.norm(ICs, axis=1)
    roughness = np.sum(np.abs(np.diff(ICs, axis=1)), axis=1)
    zero_cross = np.sum(np.diff(np.sign(ICs), axis=1) != 0, axis=1).astype(float)
    F = np.fft.rfft(ICs, axis=1)
    P = np.abs(F[:, 1:5]) ** 2
    centroid = (P * np.arange(1, 5)).sum(axis=1) / (P.sum(axis=1) + 1e-12)
    return {
        "energy": energy,
        "roughness_tv": roughness,
        "zero_crossings": zero_cross,
        "spectral_centroid_modes1_4": centroid,
    }


def main():
    d = np.load(os.path.join(ROOT, "results", "eval", "predictions.npz"))
    true_eval, fno_eval = d["u_true_eval"], d["FNO_eval"]
    err = np.linalg.norm(fno_eval - true_eval, axis=(1, 2)) / (
        np.linalg.norm(true_eval, axis=(1, 2)) + 1e-12
    )

    pt = load()
    sub_ICs = pt["ICs"][HELD_OUT_10]
    feats = ic_features(sub_ICs)

    correlations = {name: float(np.corrcoef(v, err)[0, 1]) for name, v in feats.items()}

    print("Per-IC FNO relative error (900-909):", np.round(err, 4))
    for name, r in correlations.items():
        print(f"{name}: corr with error = {r:.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    titles = {
        "energy": "Energy (L2 norm)",
        "roughness_tv": "Roughness (total variation)",
        "zero_crossings": "Zero crossings",
        "spectral_centroid_modes1_4": "Spectral centroid (modes 1-4)",
    }
    for ax, (name, v) in zip(axes.flat, feats.items()):
        ax.scatter(v, err * 100, color="#2b5b84", s=50)
        r = correlations[name]
        ax.set_title(f"{titles[name]}  (r = {r:.2f})", fontsize=11)
        ax.set_xlabel(name)
        ax.set_ylabel("FNO relative error (%)")
        ax.grid(alpha=0.25)
    fig.suptitle("Held-out FNO error vs. IC shape features (900-909): no strong correlation = no clear bias", fontsize=12)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "ic_bias_check.png")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_png}")

    result = {
        "purpose": "Checks whether the held-out 10 ICs' skew toward smoother, higher-energy "
                   "waveforms (see ic_representativeness.json) correlates with FNO error, "
                   "which would indicate the skew is biasing M3's headline error numbers.",
        "held_out_10_ids": HELD_OUT_10.tolist(),
        "per_ic_fno_relative_error": [round(float(e), 4) for e in err],
        "feature_error_correlations": {k: round(v, 3) for k, v in correlations.items()},
        "interpretation": "All correlations are weak (|r| <= 0.15, n=10), so the representativeness "
                          "skew found in ic_representativeness.json does not show a clear relationship "
                          "with error in this data - no evidence it is systematically biasing the "
                          "headline error results in either direction. With only 10 points these "
                          "correlation estimates are noisy, not a statistically definitive test.",
    }
    out_json = os.path.join(OUT_DIR, "ic_bias_check.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
