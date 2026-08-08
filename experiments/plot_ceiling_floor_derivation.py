import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from _plot_style import apply_app_style, PALETTE

from hybrid_pde.trust.signals import shock_coeff, corrected_signal, energy_signal, roughness_signal
from hybrid_pde.trust.fuse import fit_fusion, fuse
from hybrid_pde.trust.horizon import fit_calibration, trust_score
from hybrid_pde.control_214133E.controller import thresholds_for_target

FAIL = 0.10
TIGHTEST_TARGET = 0.01
LOOSEST_TARGET = 0.30
MARGIN_SIGMA = 2.0
OUT_DIR = os.path.join(ROOT, "results", "m3", "threshold_calibration")


def _sigs(u, coeff, x, t):
    return [corrected_signal(u, coeff, x, t), energy_signal(u, x), roughness_signal(u, x)]


def _relerr_traj(pred, ref):
    num = np.linalg.norm(pred - ref, axis=-1)
    den = np.linalg.norm(ref, axis=-1) + 1e-12
    return num / den


def load_trust_and_error():
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
    err_eval = _relerr_traj(fno_eval, true_eval)
    return t, trust_eval, err_eval, fno_eval, true_eval


def hybrid_error_and_cost(theta_lo, t, trust_eval, fno_eval, true_eval):
    errs, costs = [], []
    for i in range(trust_eval.shape[0]):
        below = np.where(trust_eval[i] < theta_lo)[0]
        j = below[0] if len(below) else len(t)
        u_hyb = fno_eval[i].copy()
        if j < len(t):
            u_hyb[j:] = true_eval[i, j:]
        e = np.linalg.norm(u_hyb - true_eval[i]) / (np.linalg.norm(true_eval[i]) + 1e-12)
        errs.append(e)
        costs.append((len(t) - j) / len(t))
    return float(np.mean(errs)), float(np.mean(costs))


def find_knee(grid, errs):
    xn = (grid - grid.min()) / (grid.max() - grid.min())
    yn = (errs - errs.min()) / (errs.max() - errs.min() + 1e-12)
    p1 = np.array([xn[0], yn[0]])
    p2 = np.array([xn[-1], yn[-1]])
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    dists = []
    for xi, yi in zip(xn, yn):
        p = np.array([xi, yi]) - p1
        proj = np.dot(p, line_vec_norm) * line_vec_norm
        dists.append(np.linalg.norm(p - proj))
    return grid[int(np.argmax(dists))]


def find_min_cost_theta(target, grid, table):
    feasible = [(th, e, c) for th, e, c in table if e <= target]
    return min(feasible, key=lambda r: r[2])[0] if feasible else None


def plot_ceiling(grid, errs, knee_theta, tight_theta, shipped_hi):
    apply_app_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, errs * 100, color=PALETTE["indigo"], linewidth=2, label="mean hybrid error (held-out)")
    ax.axvline(knee_theta, color=PALETTE["amber"], linestyle="--", linewidth=1.5,
               label=f"knee (diminishing returns) = {knee_theta:.2f}")
    ax.axvline(tight_theta, color=PALETTE["violet"], linestyle="--", linewidth=1.5,
               label=f"cost-min. theta for target {TIGHTEST_TARGET:.2f} = {tight_theta:.2f}")
    ax.axvline(shipped_hi, color=PALETTE["emerald"], linestyle="-", linewidth=2,
               label=f"shipped clamp_hi = {shipped_hi:.2f}")
    ax.set_xlabel("theta_lo (switch threshold)")
    ax.set_ylabel("mean hybrid error (%)")
    ax.set_title("Where the upper clamp (0.58) sits, vs. two independent reference points")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "ceiling_derivation.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_floor(trust_eval, floor, jitter_std, margin_point, shipped_lo):
    apply_app_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(trust_eval.flatten(), bins=60, color=PALETTE["indigo_light"], edgecolor="white", alpha=0.9)
    ax.axvline(floor, color=PALETTE["rose"], linestyle="-", linewidth=2,
               label=f"measured floor = {floor:.3f}")
    ax.axvline(margin_point, color=PALETTE["amber"], linestyle="--", linewidth=1.5,
               label=f"floor + {MARGIN_SIGMA:.0f} sigma jitter = {margin_point:.3f}")
    ax.axvline(shipped_lo, color=PALETTE["emerald"], linestyle="-", linewidth=2,
               label=f"shipped theta_lo @ target 0.30 = {shipped_lo:.3f}")
    ax.set_xlabel("trust score (held-out eval set, all steps)")
    ax.set_ylabel("count")
    ax.set_title("Where the loosest-target threshold (0.20) sits, vs. floor + noise margin")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "floor_derivation.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t, trust_eval, err_eval, fno_eval, true_eval = load_trust_and_error()

    grid = np.round(np.arange(0.15, 0.85, 0.01), 3)
    table = []
    for th in grid:
        e, c = hybrid_error_and_cost(th, t, trust_eval, fno_eval, true_eval)
        table.append((th, e, c))
    errs = np.array([r[1] for r in table])

    knee_theta = float(find_knee(grid, errs))
    tight_theta = float(find_min_cost_theta(TIGHTEST_TARGET, grid, table))
    shipped_hi, _ = thresholds_for_target(TIGHTEST_TARGET)
    plot_ceiling(grid, errs, knee_theta, tight_theta, shipped_hi)

    floor = float(trust_eval.min())
    jitter_std = float(np.diff(trust_eval, axis=1).std())
    margin_point = floor + MARGIN_SIGMA * jitter_std
    shipped_lo, _ = thresholds_for_target(LOOSEST_TARGET)
    plot_floor(trust_eval, floor, jitter_std, margin_point, shipped_lo)


if __name__ == "__main__":
    main()
