import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from hybrid_pde.trust.signals import shock_coeff, corrected_signal, energy_signal, roughness_signal
from hybrid_pde.trust.fuse import fit_fusion, fuse
from hybrid_pde.trust.horizon import fit_calibration, trust_score

FAIL = 0.10
OUT_DIR = os.path.join(ROOT, "results", "m3", "threshold_calibration")


def _sigs(u, coeff, x, t):
    return [corrected_signal(u, coeff, x, t), energy_signal(u, x), roughness_signal(u, x)]


def _relerr_traj(pred, ref):
    num = np.linalg.norm(pred - ref, axis=-1)
    den = np.linalg.norm(ref, axis=-1) + 1e-12
    return num / den


def load_trust():
    d = np.load(os.path.join(ROOT, "results", "eval", "predictions.npz"))
    x, t = d["x"], d["t"]
    true_seen, true_eval = d["u_true_seen"], d["u_true_eval"]
    fno_seen, fno_eval = d["FNO_seen"], d["FNO_eval"]

    coeff = shock_coeff(true_seen, x, t)
    err_seen = _relerr_traj(fno_seen, true_seen)
    fail_seen = (err_seen > FAIL).astype(float)
    fus_w = fit_fusion(_sigs(fno_seen, coeff, x, t), fail_seen)
    cal = fit_calibration(fuse(_sigs(fno_seen, coeff, x, t), fus_w), fail_seen)

    trust_eval = trust_score(fuse(_sigs(fno_eval, coeff, x, t), fus_w), cal)
    return t, trust_eval


def main():
    t, trust_eval = load_trust()
    floor = float(trust_eval.min())
    ceil = float(trust_eval.max())

    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(trust_eval.shape[0]):
        ax.plot(t, trust_eval[i], color="#2b5b84", alpha=0.35, linewidth=1)
    ax.axhline(floor, color="#c0392b", linestyle="-", linewidth=2, label=f"measured minimum = {floor:.3f}")
    ax.axhline(ceil, color="#2e8b57", linestyle="-", linewidth=2, label=f"measured maximum = {ceil:.3f}")
    ax.set_xlabel("time")
    ax.set_ylabel("trust score")
    ax.set_title("Trust signal on held-out data: observed operating range")
    ax.legend(fontsize=9, loc="center left")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out = os.path.join(OUT_DIR, "trust_range.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
