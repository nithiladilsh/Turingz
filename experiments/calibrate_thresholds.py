import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from hybrid_pde.control_214133E.controller import thresholds_for_target

POINT_1_TARGET, POINT_1_THETA = 0.029, 0.58
POINT_2_TARGET, POINT_2_THETA = 0.30, 0.20

TRUST_RANGE_MIN, TRUST_RANGE_MAX = 0.183, 0.818

CLAMP_LO, CLAMP_HI = 0.12, 0.58


def fit_line(t1, y1, t2, y2):
    slope = (y2 - y1) / (t2 - t1)
    intercept = y1 - slope * t1
    return slope, intercept


def main():
    slope, intercept = fit_line(POINT_1_TARGET, POINT_1_THETA, POINT_2_TARGET, POINT_2_THETA)

    print("Calibration points")
    print(f"  1. accuracy-saturation point : target={POINT_1_TARGET} -> theta_lo={POINT_1_THETA}")
    print(f"  2. loosest evaluated target  : target={POINT_2_TARGET} -> theta_lo={POINT_2_THETA}")
    print()
    print(f"Line through points 1 and 2 :  theta_lo = {intercept:.4f} - {abs(slope):.4f} * target")
    print(f"Shipped in controller.py    :  theta_lo = 0.6200 - 1.4000 * target  (clipped to [{CLAMP_LO}, {CLAMP_HI}])")
    print()

    print(f"Clamp bounds vs the real measured trust range [{TRUST_RANGE_MIN}, {TRUST_RANGE_MAX}]")
    print(f"  upper clamp {CLAMP_HI} = the saturation value itself (going higher buys nothing, per T2)")
    print(f"  lower clamp {CLAMP_LO} = margin below the real measured minimum ({TRUST_RANGE_MIN}),")
    print(f"                   so the controller never asks for a threshold outside where the signal actually lives")
    print()

    print("Cross-check against the live formula in controller.py, for every evaluated target:")
    print(f"{'target':>8} {'reconstructed':>14} {'shipped (live import)':>24}")
    cross_check = []
    for target in [0.30, 0.20, 0.10, 0.05, 0.02, 0.01]:
        recon = max(CLAMP_LO, min(CLAMP_HI, intercept + slope * target))
        shipped, _ = thresholds_for_target(target)
        match = round(recon, 2) == round(shipped, 2)
        print(f"{target:>8.2f} {recon:>14.4f} {shipped:>18.4f}  {'match' if match else 'DIFFERS'}")
        cross_check.append({"target": target, "reconstructed_theta_lo": round(recon, 4),
                             "shipped_theta_lo": round(shipped, 4), "match": bool(match)})

    return {
        "purpose": "Derivation of controller.py's thresholds_for_target() constants "
                   "(0.62, 1.4, clamped [0.12, 0.58]) from two calibration points taken from the "
                   "system's measured behaviour. Complementary to verify_thresholds_for_target.py, "
                   "which independently checks the same constants against a brute-force sweep.",
        "calibration_points": {
            "1_accuracy_saturation": {"target": POINT_1_TARGET, "theta_lo": POINT_1_THETA},
            "2_loosest_evaluated_target": {"target": POINT_2_TARGET, "theta_lo": POINT_2_THETA},
        },
        "fitted_formula": {"intercept": round(intercept, 4), "slope": round(slope, 4)},
        "shipped_constants": {"intercept": 0.62, "slope": -1.4, "clamp_lo": CLAMP_LO, "clamp_hi": CLAMP_HI},
        "trust_signal_measured_range": {"min": TRUST_RANGE_MIN, "max": TRUST_RANGE_MAX},
        "cross_check_per_target": cross_check,
    }


if __name__ == "__main__":
    import json

    result = main()
    out_dir = os.path.join(ROOT, "results", "m3", "threshold_calibration")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "threshold_calibration.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")
