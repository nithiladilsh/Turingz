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


@router.get("/api/m3/regime")
def regime():
    """Per-surrogate cost/accuracy from the team's measured deployment analysis."""
    try:
        c = _load("results/deployment/cost_summary.json")
    except FileNotFoundError:
        return {"error": "no cost summary found"}

    def g(m):
        v = c.get(m, {})
        return {"deploy_s": v.get("deploy_s"), "err_in": v.get("err_in"), "err_extrap": v.get("err_extrap")}

    return {
        "surrogates": {m: g(m) for m in ("FNO", "DeepONet", "PINN")},
        "numerical": {m: g(m) for m in ("ColeHopf", "Spectral", "FDM")},
    }


@router.get("/api/m3/frontiers")
def frontiers():
    """Both measured frontiers, each normalised by its OWN pure-numerical baseline
    so the two runs are comparable despite wall-clock variance between sessions."""
    files = {
        "FNO": "results/m3/step9d_coarse_integration/timed_cost_result_m2.json",
        "DeepONet": "results/m3/step9d_coarse_integration/timed_cost_result_deeponet.json",
    }
    out = {}
    for name, rel in files.items():
        try:
            d = _load(rel)
        except FileNotFoundError:
            continue
        num_c = d["pure_numerical"].get("cost_s") or d["pure_numerical"].get("cost") or 1.0
        mlv = d["pure_ml"]
        ml_c = mlv.get("cost_s") or mlv.get("cost") or 0.0
        rows = []
        for r in d["frontier"]:
            c = r.get("cost_s", r.get("cost")) or 0.0
            rows.append({"target": r["target"], "cost": c, "rel_cost": c / num_c,
                         "error": r["mean_error"], "hit_rate": r.get("hit_rate", 1.0)})
        out[name] = {
            "frontier": rows,
            "pure_ml": {"cost": ml_c, "rel_cost": ml_c / num_c, "error": mlv["mean_error"]},
            "pure_numerical": {"cost": num_c, "rel_cost": 1.0, "error": d["pure_numerical"]["mean_error"]},
        }
    return out


@router.get("/api/m3/costs")
def costs():
    """Full measured deployment-cost analysis for every solver (ML and numerical)."""
    try:
        c = _load("results/deployment/cost_summary.json")
    except FileNotFoundError:
        return {"error": "no cost summary found"}
    out = {"models": {}, "env": c.get("_env", {})}
    for m, v in c.items():
        if m == "_env":
            continue
        out["models"][m] = {
            "kind": "ml" if m in ("FNO", "DeepONet", "PINN") else "numerical",
            "params": v.get("params"), "params_mb": v.get("params_mb"), "disk_mb": v.get("disk_mb"),
            "infer_ms": v.get("infer_ms"), "infer_ms_std": v.get("infer_ms_std"),
            "deploy_s": v.get("deploy_s"), "throughput_ic_s": v.get("throughput_ic_s"),
            "rss_mb": v.get("rss_mb"),
            "err_in": v.get("err_in"), "err_extrap": v.get("err_extrap"),
            "train_s": v.get("train_s"), "adam_s": v.get("adam_s"), "lbfgs_s": v.get("lbfgs_s"),
            "scaling": v.get("scaling"),
        }
    return out
