import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from hybrid_pde.control_214133E.controller import thresholds_for_target

CLAMP_LO = 0.12
TRUST_RANGE_MIN = 0.183
EVALUATED_TARGETS = [0.30, 0.20, 0.10, 0.05, 0.02, 0.01]


def main():
    rows = []
    for target in EVALUATED_TARGETS:
        raw = 0.62 - 1.4 * target
        shipped, _ = thresholds_for_target(target)
        reached = round(shipped, 4) == CLAMP_LO
        rows.append({
            "target": target,
            "theta_lo_unclamped": round(raw, 4),
            "theta_lo_shipped": round(shipped, 4),
            "lower_clamp_reached": bool(reached),
        })
        print(f"target={target:>5}  unclamped={raw:>7.4f}  shipped={shipped:.4f}  clamp reached: {reached}")

    crossover_target = round((0.62 - CLAMP_LO) / 1.4, 4)
    print(f"\nLower clamp would first activate at target = {crossover_target} (looser than any evaluated target)")

    return {
        "purpose": "Reference table for controller.py's lower clamp (0.12): theta_lo before and after "
                   "clamping, for every evaluated target, showing the clamp is never reached in practice.",
        "clamp_lo": CLAMP_LO,
        "measured_trust_floor": TRUST_RANGE_MIN,
        "per_target": rows,
        "clamp_first_activates_at_target": crossover_target,
    }


if __name__ == "__main__":
    import json

    result = main()
    out_dir = os.path.join(ROOT, "results", "m3", "threshold_calibration")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "lower_clamp_reference.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")
