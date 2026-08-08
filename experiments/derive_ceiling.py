import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from hybrid_pde.trust.signals import shock_coeff, corrected_signal, energy_signal, roughness_signal
from hybrid_pde.trust.fuse import fit_fusion, fuse
from hybrid_pde.trust.horizon import fit_calibration, trust_score
from hybrid_pde.control_214133E.controller import thresholds_for_target

FAIL = 0.10
TIGHTEST_TARGET = 0.02


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


def main():
    t, trust_eval, err_eval, fno_eval, true_eval = load_trust_and_error()

    grid = np.round(np.arange(0.15, 0.85, 0.01), 3)
    table = []
    for th in grid:
        e, c = hybrid_error_and_cost(th, t, trust_eval, fno_eval, true_eval)
        table.append((th, e, c))
    errs = np.array([r[1] for r in table])

    knee_theta = float(find_knee(grid, errs))
    tight_theta = float(find_min_cost_theta(TIGHTEST_TARGET, grid, table))
    midpoint = round((knee_theta + tight_theta) / 2, 4)

    shipped, _ = thresholds_for_target(TIGHTEST_TARGET)

    print(f"Knee of the error-vs-threshold curve (diminishing-returns point): theta = {knee_theta:.3f}")
    print(f"Cost-minimal theta meeting the tightest evaluated target ({TIGHTEST_TARGET}): theta = {tight_theta:.3f}")
    print(f"Shipped clamp_hi in controller.py: {shipped:.3f}")

    return {
        "purpose": "Two formula-free reference points for controller.py's clamp_hi (0.58), both measured "
                   "on held-out data: the knee of the error-vs-threshold curve, and the cost-minimal "
                   "threshold meeting the tightest evaluated target. Neither calls thresholds_for_target() "
                   "to produce its value; the formula is only used afterward, to compare against what was "
                   "shipped.",
        "knee_point": {"theta_lo": round(knee_theta, 4)},
        "tightest_target_min_cost_point": {"target": TIGHTEST_TARGET, "theta_lo": round(tight_theta, 4)},
        "shipped_clamp_hi": round(shipped, 4),
    }


if __name__ == "__main__":
    import json

    result = main()
    out_dir = os.path.join(ROOT, "results", "m3", "threshold_calibration")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ceiling_derivation.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")
