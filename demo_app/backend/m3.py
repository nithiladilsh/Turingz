import os
import json
from fastapi import APIRouter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
router = APIRouter()


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


@router.get("/api/m3/frontier")
def frontier():
    for rel in ("results/m3/step9d_coarse_integration/timed_cost_result_m2.json",
                "results/m3/step9d_coarse_integration/timed_cost_result.json"):
        try:
            d = _load(rel)
        except FileNotFoundError:
            continue
        rows = d["frontier"]
        return {
            "frontier": [{
                "target": r["target"],
                "cost": r.get("cost_s", r.get("cost")),
                "error": r["mean_error"],
                "hit_rate": r.get("hit_rate", 1.0),
            } for r in rows],
            "pure_ml": {"cost": d["pure_ml"].get("cost_s", d["pure_ml"].get("cost")),
                        "error": d["pure_ml"]["mean_error"]},
            "pure_numerical": {"cost": d["pure_numerical"].get("cost_s", d["pure_numerical"].get("cost")),
                               "error": d["pure_numerical"]["mean_error"]},
            "source": rel.split("/")[-1],
        }
    return {"error": "no frontier result found"}


@router.get("/api/m3/robustness")
def robustness():
    try:
        c = _load("results/m3/step8_robustness/verification.json")["checks"]
    except FileNotFoundError:
        return {"error": "no robustness result found"}
    a = c.get("adaptive_beats_fixed_matched_budget", {})
    o = c.get("ood_harder_than_indist", {})
    return {
        "adaptive_err": a.get("adaptive_mean_err"),
        "fixed_err": a.get("fixed_mean_err"),
        "adaptive_corr": a.get("adaptive_corr"),
        "fixed_corr": a.get("fixed_corr"),
        "indist_err": o.get("indist_mean_err_loose"),
        "ood_err": o.get("ood_mean_err_loose"),
    }
