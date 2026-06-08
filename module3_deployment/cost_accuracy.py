"""
================================================================================
COST  x  ACCURACY  (the trade-off the interim is built around)
Team : Turingz   File : module3_deployment/cost_accuracy.py

Joins the two halves of Module 3:
    * ACCURACY  - from the shared runner (common.evaluation -> common.metrics),
                  so the numbers are identical to what reliability/robustness use.
    * COST      - from cost_meter (latency, memory, throughput, #params).

and feeds them into hei.efficiency_index to produce the cost-accuracy trade-off
table + deployment recommendation.

Nothing here computes its own accuracy - it calls the shared pipeline. That is
the whole point of the remediation: one accuracy implementation for everyone.
================================================================================
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common import evaluation as EV          # noqa: E402  shared runner
from common import canonical_split as cs     # noqa: E402

from . import cost_meter
from . import hei as HEI


def _mean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None and v == v]
    return float(np.mean(vals)) if vals else None


def _accuracy_from_shared(solver, reference: Dict,
                          samples: List[int]) -> Dict[str, Optional[float]]:
    """Run the SHARED evaluation runner and distil headline accuracy numbers."""
    res = EV.evaluate_solver(solver, reference, sample_indices=samples,
                             keep_curves=False, with_signals=False)
    per = res["samples"]
    global_l2 = _mean([per[i]["relative_l2"]["mean"] for i in per])
    extrap_l2 = _mean([per[i]["windowed_rel_l2"]["extrap"]["mean"] for i in per])
    final_l2 = _mean([per[i]["relative_l2"]["final"] for i in per])
    return {
        "relative_l2_global": global_l2,
        "relative_l2_extrap": extrap_l2,
        "relative_l2_final": final_l2,
    }


def profile_solver(solver, reference: Dict, model_type: str,
                   train_wall_s: float = 0.0,
                   eval_samples: Optional[List[int]] = None,
                   cost_sample: int = 0, repeats: int = 10,
                   warmup: int = 2) -> Dict:
    """Full cost+accuracy profile for one loaded solver.

    train_wall_s : the model's wall_time_s for ONE training run (read it from
                   the train log the official train.py wrote, or from
                   cost_meter.measure_training). Used for amortised training cost.
    eval_samples : ICs to score accuracy on (default: canonical EVAL_IDX).
    cost_sample  : which IC to time inference on (default 0, the focal case).
    """
    if eval_samples is None:
        eval_samples = list(cs.EVAL_IDX)

    acc = _accuracy_from_shared(solver, reference, eval_samples)
    inf = cost_meter.measure_solver(solver, reference, cost_sample,
                                    repeats=repeats, warmup=warmup)

    latency_s = inf["latency_ms"]["median"] / 1000.0
    return {
        "name": getattr(solver, "name", model_type),
        "model_type": model_type.lower(),
        "trainings_per_benchmark": HEI.trainings_per_full_benchmark(model_type),
        "n_parameters": int(solver.num_parameters()),
        # accuracy (shared)
        "relative_l2_global": acc["relative_l2_global"],
        "relative_l2_extrap": acc["relative_l2_extrap"],
        "relative_l2_final": acc["relative_l2_final"],
        # cost (this module)
        "latency_s": latency_s,
        "latency_ms_median": inf["latency_ms"]["median"],
        "memory_mb": inf["memory"]["cpu_delta_mb"],
        "peak_cpu_mb": inf["memory"]["peak_cpu_mb"],
        "gpu_delta_mb": inf["memory"]["gpu_delta_mb"],
        "throughput_points_per_s": inf["throughput"]["points_per_s"],
        "throughput_rollouts_per_s": inf["throughput"]["rollouts_per_s"],
        "train_wall_s": float(train_wall_s),
        "cost_grid": inf["grid"],
    }


def build_trade_off(profiles: List[Dict],
                    accuracy_key: str = "relative_l2_global",
                    weights: Optional[HEI.HEIWeights] = None) -> Dict:
    """Turn a list of profile dicts into the HEI-ranked trade-off + recommendation.

    accuracy_key selects which accuracy figure drives the HEI; default is the
    global relative-L2, but 'relative_l2_extrap' is often the more honest
    deployment number because extrapolation is where ML solvers fail.
    """
    rows = []
    for p in profiles:
        rows.append({
            "name": p["name"],
            "model_type": p["model_type"],
            "relative_l2": p.get(accuracy_key) if p.get(accuracy_key) is not None else float("inf"),
            "latency_s": p.get("latency_s"),
            "memory_mb": p.get("memory_mb"),
            "train_wall_s": p.get("train_wall_s", 0.0),
        })
    scored = HEI.efficiency_index(rows, weights=weights)
    # carry the richer fields back onto the scored rows for plotting/JSON
    by_name = {p["name"]: p for p in profiles}
    for s in scored:
        s.update({k: v for k, v in by_name.get(s["name"], {}).items()
                  if k not in s})
    return {
        "accuracy_key": accuracy_key,
        "ranked": scored,
        "recommendation": HEI.recommend(scored),
    }
