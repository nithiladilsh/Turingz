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
    # adaptive-vs-fixed used to come from step8_robustness/verification.json, but the
    # team's own notes flag that exact figure as a stand-in ("the 0.0-vs-8.6 stand-in;
    # superseded by T2") -- adaptive error is never actually 0.0% on real ICs. T2 is
    # step11_switching_ablation/switching_ablation.json: the real "latch" (adaptive)
    # vs "fixed_matched" (fixed-interval, same correction budget) comparison across
    # all 10 held-out ICs, at every target. Report it at target 0.05, the same
    # headline target used elsewhere on this page.
    adaptive_err = adaptive_corr = fixed_err = fixed_corr = None
    try:
        ab = _load("results/m3/step11_switching_ablation/switching_ablation.json")
        row = min(ab["rows"], key=lambda r: abs(r["target"] - 0.05))
        adaptive_err = row["policies"]["latch"]["mean_error"]
        adaptive_corr = row["policies"]["latch"]["corrections"]
        fixed_err = row["policies"]["fixed_matched"]["mean_error"]
        fixed_corr = row["policies"]["fixed_matched"]["corrections"]
    except (FileNotFoundError, KeyError):
        pass

    indist_err = ood_err = None
    try:
        o = _load("results/m3/step8_robustness/verification.json")["checks"].get("ood_harder_than_indist", {})
        indist_err = o.get("indist_mean_err_loose")
        ood_err = o.get("ood_mean_err_loose")
    except FileNotFoundError:
        pass

    return {
        "adaptive_err": adaptive_err,
        "fixed_err": fixed_err,
        "adaptive_corr": adaptive_corr,
        "fixed_corr": fixed_corr,
        "indist_err": indist_err,
        "ood_err": ood_err,
        "source": "step11_switching_ablation (T2) @ target 0.05",
    }


@router.get("/api/m3/switching_ablation")
def switching_ablation():
    """T2's full policy ablation (latch/deadband/naive/hardcoded), across every
    target, on the same 10 held-out ICs. Separate from /api/m3/robustness's
    latch-vs-fixed-interval comparison above -- this compares latch (the
    production policy) against three alternative SMART-trigger designs, not
    against a brute-force baseline. Framed around hit-rate and target-
    responsiveness rather than raw error, because raw error alone isn't
    uniformly in latch's favor at loose targets (hardcoded happens to score
    lower there purely by luck, since it ignores the target entirely)."""
    try:
        d = _load("results/m3/step11_switching_ablation/switching_ablation.json")
    except FileNotFoundError:
        return {"error": "no switching ablation result found"}
    policies = ["latch", "deadband", "naive", "hardcoded"]
    hit_range = {}
    for p in policies:
        hits = [r["policies"][p]["hit_rate"] for r in d["rows"]]
        hit_range[p] = [min(hits), max(hits)]
    return {"target_response": d["target_response"], "hit_range": hit_range}


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


@router.get("/api/m3/pinn_regime")
def pinn_regime():
    """PINN's per-target rows (mean_error/hit_rate), same source the Findings
    tab's PINN card numbers come from -- so that card can show the real
    target-dependent range instead of a single hardcoded snapshot."""
    try:
        d = _load("results/m3/step12_pinn_regime/pinn_controller.json")
    except FileNotFoundError:
        return {"error": "no pinn regime result found"}
    return {"rows": [{"target": r["target"], "error": r["mean_error"], "hit_rate": r["hit_rate"]}
                      for r in d["rows"]]}


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


@router.get("/api/m3/achievability")
def achievability():
    """Hit-rate and mean error per requested accuracy target, on the 10 held-out
    ICs -- shows the controller tracking the target down to a real accuracy
    floor, not an unqualified 100% success story."""
    try:
        d = _load("results/m3/achievability/m3_achievability.json")
    except FileNotFoundError:
        return {"error": "no achievability result found"}
    rows = sorted(d["per_target"], key=lambda r: -r["target"])
    return {"rows": [{"target": r["target"], "mean_error": r["mean_error"], "hit_rate": r.get("hit_rate")} for r in rows]}


@router.get("/api/m3/cost_model")
def cost_model():
    """Validates the additive cost model (ml_steps*ml_step_s + correction_steps*
    correction_step_s) against real measured wall-clock time."""
    try:
        d = _load("results/m3/cost_model/m3_cost_model.json")
    except FileNotFoundError:
        return {"error": "no cost model result found"}
    return {
        "pearson_r": d.get("pearson_r"),
        "mape": d.get("mape"),
        "slope_measured_vs_predicted": d.get("slope_measured_vs_predicted"),
        "ml_step_s": d.get("ml_step_s"),
        "num_step_s": d.get("num_step_s"),
    }


@router.get("/api/m3/ood_frontier")
def ood_frontier():
    """In-distribution vs out-of-distribution cost/error, same controller and
    thresholds, on frequency-shifted and amplitude-shifted test waves."""
    try:
        d = _load("results/m3/ood_frontier/m3_ood_frontier.json")
    except FileNotFoundError:
        return {"error": "no ood frontier result found"}
    return {"indist": d.get("indist"), "ood": d.get("ood")}


@router.get("/api/m3/switch_timing")
def switch_timing():
    """How close the controller's real switch time is to a cost-optimal oracle's
    switch time, per target -- a control-precision check, not an error check."""
    try:
        d = _load("results/m3/error_decomposition/m3_error_decomposition.json")
    except FileNotFoundError:
        return {"error": "no error decomposition result found"}
    rows = sorted(d["decomposition"], key=lambda r: -r["target"])
    return {"rows": [{"target": r["target"], "real_switch_t": r["real_switch_t"],
                       "oracle_switch_t": r["oracle_switch_t"], "detection_lag_t": r["detection_lag_t"]}
                      for r in rows]}


@router.get("/api/m3/ic_representativeness")
def ic_representativeness():
    """Whether the 10 held-out test ICs are a representative sample of the full
    1000-IC dataset, and whether the resulting skew correlates with error."""
    out = {}
    try:
        out["representativeness"] = _load("results/m3/ic_representativeness/ic_representativeness.json")
    except FileNotFoundError:
        out["representativeness"] = None
    try:
        out["bias_check"] = _load("results/m3/ic_representativeness/ic_bias_check.json")
    except FileNotFoundError:
        out["bias_check"] = None
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
