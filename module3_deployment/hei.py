"""
================================================================================
HYBRID EFFICIENCY INDEX (HEI)
Team : Turingz   File : module3_deployment/hei.py

The HEI is Module 3's headline contribution: a single, transparent number that
captures "how much accuracy you get per unit of deployment cost", with the
paradigm asymmetry the compatibility review insists on baked in (a PINN pays
its training cost PER initial condition; an operator pays once and amortises).

It is used in two places:

  1. OFFLINE  -  efficiency_index(): ranks the solvers for the cost-accuracy
     write-up and the deployment recommendation (interim deliverable).

  2. ONLINE   -  RoutingPolicy: the latency-aware engine (app.py) uses it to
     decide, per step, whether to trust the fast ML prediction or pay for a
     numerical (Cole-Hopf) correction - given a live failure score from
     Module 1 and a strict latency / memory budget.

Nothing here re-implements accuracy or the dataset split; accuracy comes from
common.metrics and the per-IC accounting comes from common.canonical_split, so
the numbers stay consistent with the reliability and robustness modules.
================================================================================
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Make the project root importable so `common` resolves no matter where this
# module is invoked from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common import canonical_split as cs  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Paradigm accounting - the asymmetry the docs require us to state explicitly
# ─────────────────────────────────────────────────────────────────────────────
def trainings_per_full_benchmark(model_type: str) -> int:
    """How many model trainings it takes to cover the whole benchmark.

    Operators (FNO, DeepONet) train ONCE and serve every IC. The PINN is
    single-instance: it trains one model PER IC, so covering all evaluated ICs
    costs len(EVAL_IDX) trainings. (Mirrors the 'count_for_ic' idea from the
    compatibility review's proposed interface.)
    """
    return len(cs.EVAL_IDX) if model_type.lower() == "pinn" else 1


def amortized_training_cost_s(model_type: str, wall_time_s_per_training: float,
                              n_ics_served: Optional[int] = None) -> float:
    """Training seconds charged per served IC.

    PINN:     wall_time_s_per_training            (one model used for one IC)
    Operator: wall_time_s_per_training / n_served (one model amortised over all)
    """
    if n_ics_served is None:
        n_ics_served = len(cs.EVAL_IDX)
    if model_type.lower() == "pinn":
        return float(wall_time_s_per_training)
    return float(wall_time_s_per_training) / max(1, n_ics_served)


# ─────────────────────────────────────────────────────────────────────────────
# Offline efficiency index (the interim "HEI" number)
# ─────────────────────────────────────────────────────────────────────────────
def accuracy_score(relative_l2: float) -> float:
    """Map a relative-L2 error to a bounded accuracy score in (0, 1].

    score = 1 / (1 + rel_l2)   ->   rel_l2=0 gives 1.0; large error -> 0.
    Bounded and monotonic, so it never goes negative for a broken model the
    way (1 - rel_l2) would.
    """
    return 1.0 / (1.0 + max(0.0, float(relative_l2)))


def _normalize(values: List[float]) -> List[float]:
    """Scale a list of non-negative costs to (0, 1] by dividing by the max,
    so the most expensive candidate is 1.0 and HEI is comparable across rows."""
    finite = [v for v in values if v is not None and v == v and v > 0]
    hi = max(finite) if finite else 1.0
    out = []
    for v in values:
        if v is None or v != v:
            out.append(1.0)         # treat unknown as worst-case
        else:
            out.append(max(0.0, float(v)) / hi if hi > 0 else 0.0)
    return out


@dataclass
class HEIWeights:
    """Relative importance of each cost component in the deployment index.
    Defaults emphasise inference (latency, memory) because in deployment the
    model is trained once but served many times; training is amortised."""
    latency: float = 0.45
    memory: float = 0.30
    training: float = 0.25


def efficiency_index(rows: List[Dict], weights: Optional[HEIWeights] = None) -> List[Dict]:
    """Compute the Hybrid Efficiency Index for a set of solver measurements.

    Each input row is a dict with (at minimum):
        model_type        : 'pinn' | 'fno' | 'deeponet' | 'hybrid'
        name              : label for tables/plots
        relative_l2       : accuracy (lower is better) - from common.metrics
        latency_s         : median rollout latency in seconds (inference)
        memory_mb         : peak memory delta in MB (inference)
        train_wall_s      : wall_time_s for ONE training run

    Returns the same rows, each augmented with:
        accuracy_score, amortized_train_s, cost_index, HEI
    where HEI = accuracy_score / cost_index  (higher = more deployable).
    The cost_index blends min-max-normalised latency, memory and amortised
    training cost, so HEI is a *relative* ranking across the rows you pass in.
    """
    w = weights or HEIWeights()
    wsum = w.latency + w.memory + w.training

    amort = [amortized_training_cost_s(r["model_type"], r.get("train_wall_s", 0.0) or 0.0)
             for r in rows]
    n_lat = _normalize([r.get("latency_s") for r in rows])
    n_mem = _normalize([r.get("memory_mb") for r in rows])
    n_trn = _normalize(amort)

    out = []
    for r, a, nl, nm, nt in zip(rows, amort, n_lat, n_mem, n_trn):
        cost_index = (w.latency * nl + w.memory * nm + w.training * nt) / wsum
        cost_index = max(cost_index, 1e-9)
        acc = accuracy_score(r["relative_l2"])
        rr = dict(r)
        rr["accuracy_score"] = round(acc, 6)
        rr["amortized_train_s"] = round(a, 4)
        rr["cost_index"] = round(cost_index, 6)
        rr["HEI"] = round(acc / cost_index, 6)
        out.append(rr)
    out.sort(key=lambda d: d["HEI"], reverse=True)
    return out


def recommend(rows_with_hei: List[Dict]) -> Dict:
    """Turn HEI-scored rows into a one-line deployment recommendation."""
    if not rows_with_hei:
        return {"recommendation": "no data"}
    best_overall = rows_with_hei[0]
    best_accuracy = min(rows_with_hei, key=lambda d: d["relative_l2"])
    fastest = min(rows_with_hei, key=lambda d: (d.get("latency_s") or float("inf")))
    return {
        "best_efficiency": {"name": best_overall["name"], "HEI": best_overall["HEI"]},
        "best_accuracy": {"name": best_accuracy["name"],
                          "relative_l2": best_accuracy["relative_l2"]},
        "lowest_latency": {"name": fastest["name"],
                           "latency_s": fastest.get("latency_s")},
        "recommendation": (
            f"For latency-bound deployment use {best_overall['name']} "
            f"(highest HEI = {best_overall['HEI']}); when accuracy dominates "
            f"and cost is acceptable use {best_accuracy['name']} "
            f"(lowest relative-L2 = {best_accuracy['relative_l2']})."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Online routing policy (used by the FastAPI Engine Control Unit, app.py)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RoutingConfig:
    failure_threshold: float = 0.5      # Module-1 failure score above which we worry
    latency_budget_ms: float = 50.0     # hard per-step latency ceiling
    mem_headroom_mb: float = 512.0      # refuse numerical correction below this free RAM
    numerical_cost_ms: float = 25.0     # measured cost of one Cole-Hopf correction step


@dataclass
class RoutingDecision:
    route: str                          # 'ml' | 'numerical' | 'ml_forced_budget' | 'ml_forced_oom'
    reason: str
    hei: float
    failure_score: float


class RoutingPolicy:
    """Latency-aware decision: trust the fast ML step, or pay for a numerical
    correction? Encapsulates the HEI trade-off the README describes.

    Decision logic (transparent and tunable):
      * If the failure score is low -> keep the cheap ML step.
      * If it is high -> a numerical correction is worth it, BUT only if it
        fits the latency budget and there is enough free memory (OOM guard).
      * Otherwise fall back to ML and flag that the budget/headroom forced it.
    The step-level HEI = (expected accuracy gain) / (extra cost) quantifies
    whether the correction earns its keep; it is returned for logging/plots.
    """

    def __init__(self, config: Optional[RoutingConfig] = None):
        self.cfg = config or RoutingConfig()

    def step_hei(self, failure_score: float, extra_cost_ms: float) -> float:
        """Expected reliability recovered per millisecond of correction.
        failure_score in [0,1] proxies the accuracy we'd recover by correcting."""
        return float(failure_score) / max(extra_cost_ms, 1e-6)

    def decide(self, failure_score: float, free_mem_mb: float,
               numerical_cost_ms: Optional[float] = None) -> RoutingDecision:
        cfg = self.cfg
        ncost = cfg.numerical_cost_ms if numerical_cost_ms is None else numerical_cost_ms
        hei = self.step_hei(failure_score, ncost)

        if failure_score < cfg.failure_threshold:
            return RoutingDecision("ml", "failure score below threshold; ML is reliable",
                                   round(hei, 6), float(failure_score))
        if free_mem_mb < cfg.mem_headroom_mb:
            return RoutingDecision("ml_forced_oom",
                                   "correction warranted but insufficient free memory; "
                                   "throttling numerical engine to avoid OOM",
                                   round(hei, 6), float(failure_score))
        if ncost > cfg.latency_budget_ms:
            return RoutingDecision("ml_forced_budget",
                                   "correction warranted but exceeds latency budget; "
                                   "staying on ML to honour the budget",
                                   round(hei, 6), float(failure_score))
        return RoutingDecision("numerical",
                               "failure score high and correction fits budget + memory",
                               round(hei, 6), float(failure_score))
