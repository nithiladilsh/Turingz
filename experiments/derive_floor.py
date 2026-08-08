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
MARGIN_SIGMA = 2.0
LOOSEST_TARGET = 0.30


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

    return trust_score(fuse(_sigs(fno_eval, coeff, x, t), fus_w), cal)


def main():
    trust_eval = load_trust()

    floor = float(trust_eval.min())
    step_jitter_std = float(np.diff(trust_eval, axis=1).std())
    margin_point = round(floor + MARGIN_SIGMA * step_jitter_std, 4)

    shipped, _ = thresholds_for_target(LOOSEST_TARGET)

    print(f"Measured trust floor (held-out eval set): {floor:.4f}")
    print(f"Step-to-step trust jitter, 1 std dev (held-out eval set): {step_jitter_std:.4f}")
    print(f"Floor + {MARGIN_SIGMA:.0f} std dev of jitter: {margin_point:.4f}")
    print(f"Shipped theta_lo at the loosest evaluated target ({LOOSEST_TARGET}): {shipped:.4f}")

    return {
        "purpose": "Reference point for controller.py's theta_lo at the loosest evaluated target (0.20), "
                   "built from the measured trust floor plus a noise margin, both measured on held-out "
                   "data.",
        "measured_trust_floor": round(floor, 4),
        "step_jitter_std": round(step_jitter_std, 4),
        "margin_sigma": MARGIN_SIGMA,
        "floor_plus_margin": margin_point,
        "shipped_theta_lo_at_loosest_target": {"target": LOOSEST_TARGET, "theta_lo": round(shipped, 4)},
    }


if __name__ == "__main__":
    import json

    result = main()
    out_dir = os.path.join(ROOT, "results", "m3", "threshold_calibration")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "floor_derivation.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")
