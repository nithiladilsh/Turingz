from __future__ import annotations
import json, os, sys
from . import config
from .integrate import partial_main_coarse_timed


def report(d):
    ml, num = d["pure_ml"], d["pure_numerical"]
    print()
    print("FULL SYSTEM  -  %s + M1 coarse trust + M2 coupling + M3 controller" % d.get("model", "FNO"))
    print("=" * 78)
    print("%-22s %10s %10s" % ("baseline", "cost_s", "error"))
    print("-" * 78)
    print("%-22s %10.3f %9.2f%%" % ("pure ML", ml["cost_s"], 100 * ml["mean_error"]))
    print("%-22s %10.3f %9.4f%%" % ("pure numerical", num["cost_s"], 100 * num["mean_error"]))
    print()
    print("%-10s %10s %10s %10s %9s %9s" % ("target", "cost_s", "error", "vs num", "vs ML", "hit"))
    print("-" * 78)
    for r in d["frontier"]:
        print("%-10.3g %10.3f %9.2f%% %9.2fx %8.2fx %8.0f%%" % (
            r["target"], r["cost_s"], 100 * r["mean_error"],
            num["cost_s"] / max(r["cost_s"], 1e-9),
            ml["mean_error"] / max(r["mean_error"], 1e-9),
            100 * r["hit_rate"]))
    print("-" * 78)
    c = [r["cost_s"] for r in d["frontier"]]
    e = [r["mean_error"] for r in d["frontier"]]
    print("HEADLINE: %.2f-%.2f s at %.1f-%.1f%% error, versus %.2f s numerical and %.1f%% pure ML" % (
        min(c), max(c), 100 * min(e), 100 * max(e), num["cost_s"], 100 * ml["mean_error"]))
    print("          %.1f-%.1fx cheaper than numerical, up to %.1fx more accurate than pure ML" % (
        num["cost_s"] / max(c), num["cost_s"] / min(c), ml["mean_error"] / min(e)))
    print("          vs num / vs ML are ratios: higher is better")


def main(model="FNO"):
    d = partial_main_coarse_timed(model=model)
    out_dir = os.path.join(config.RESULTS_DIR, "m3", "step9d_coarse_integration")
    os.makedirs(out_dir, exist_ok=True)
    name = "timed_cost_result_m2.json" if model == "FNO" else "timed_cost_result_%s.json" % model.lower()
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1)
    report(d)
    print("\nsaved to", path)
    return d


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "FNO")
