import os
import json
import glob
import numpy as np

from .reliability import analyze_model, load_reference_any, roc_auc, THRESHOLD

DATA = "data/colehopf/burgers_1d_cole_hopf.pt"
EVAL_DIR = "results/evaluation"
OUT_DIR = "module1_reliability/results"


def _fmt(v):
    return "  n/a " if v != v else f"{v:.3f}" 


def main():
    reference = load_reference_any(DATA)
    os.makedirs(OUT_DIR, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(EVAL_DIR, "*_evaluation.json")))
    if not paths:
        print(f"No evaluation files in {EVAL_DIR}. Run common.evaluate first.")
        return

    reports = []
    pool = {"raw": [], "excess": [], "periodicity": [], "label": []}
    for p in paths:
        d = json.load(open(p))
        rep = analyze_model(d, reference, threshold=THRESHOLD)
        key = os.path.basename(p).replace("_evaluation.json", "")
        for k in pool:
            pool[k] += rep["_pool"][k]
        rep.pop("_pool")
        json.dump(rep, open(os.path.join(OUT_DIR, f"{key}_reliability.json"), "w"), indent=2)
        reports.append((key, rep))

    print("\n" + "=" * 80)
    print(f"  RELIABILITY OF TEMPORAL EXTRAPOLATION   (failure = relative error > {THRESHOLD:.0%})")
    print("=" * 80)
    print("  Per model (t_train_end = 1.0, so extrapolation is t in 1..2):")
    print(f"  {'model':<11}{'extrap fail %':<15}{'reliable horizon':<18}{'per-model AUC (excess)':<22}")
    print("  " + "-" * 66)
    for key, r in reports:
        ff = r["extrap_fail_fraction"] * 100
        hz = r["mean_reliable_horizon"]
        auc = r["detector_auc"]["excess_residual"]
        print(f"  {key:<11}{ff:>6.0f}%        {hz:<18.2f}{_fmt(auc):<22}")

    lab = np.asarray(pool["label"], dtype=np.int64)
    print("\n  POOLED detector quality across ALL models (mix of reliable + failed):")
    print(f"    points = {len(lab)}   failed = {lab.mean()*100:.0f}%")
    for name in ["raw", "excess", "periodicity"]:
        print(f"    AUC  {name:<12} = {_fmt(roc_auc(pool[name], lab))}")
    print("\n  AUC: 0.5 = useless, 1.0 = perfect.  Reports saved to", OUT_DIR)


if __name__ == "__main__":
    main()
