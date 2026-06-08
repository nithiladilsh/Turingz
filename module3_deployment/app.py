"""
================================================================================
ENGINE CONTROL UNIT  -  latency-aware orchestration API
Team : Turingz   File : module3_deployment/app.py

This is the deployment side of Module 3 from the README: a FastAPI middleware
that ingests a live FAILURE SCORE (produced by Module 1's reliability detector)
and uses the Hybrid Efficiency Index to make a split-second routing decision -
trust the fast ML prediction, or pay for a numerical (Cole-Hopf) correction -
while honouring a strict latency budget and throttling the numerical engine to
avoid Out-Of-Memory crashes.

It reuses module3_deployment.hei.RoutingPolicy for the decision and
module3_deployment.cost_meter for live memory readings, so the online engine
and the offline cost study share one definition of cost and efficiency.

Run:
    uvicorn module3_deployment.app:app --reload --port 8000
or:
    python -m module3_deployment.app          # runs a no-server self-demo

Endpoints:
    GET  /health           liveness
    GET  /device           hardware + current free memory
    GET  /config           current routing thresholds
    POST /config           update thresholds (partial)
    POST /route            decide ml | numerical for one step
    POST /simulate         drive a synthetic failure-score trajectory and
                           return the per-step routing decisions (demo)
================================================================================
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import psutil  # noqa: E402

from module3_deployment.hei import RoutingPolicy, RoutingConfig  # noqa: E402
from module3_deployment import cost_meter                         # noqa: E402


_policy = RoutingPolicy(RoutingConfig())


def _free_mem_mb() -> float:
    return psutil.virtual_memory().available / (1024.0 ** 2)


def route_step(failure_score: float, free_mem_mb: Optional[float] = None,
               numerical_cost_ms: Optional[float] = None) -> dict:
    """Core decision used by both the HTTP layer and the self-demo."""
    fm = _free_mem_mb() if free_mem_mb is None else free_mem_mb
    d = _policy.decide(failure_score, fm, numerical_cost_ms)
    return {
        "route": d.route,
        "reason": d.reason,
        "step_hei": d.hei,
        "failure_score": d.failure_score,
        "free_mem_mb": round(fm, 1),
        "latency_budget_ms": _policy.cfg.latency_budget_ms,
        "numerical_cost_ms": (_policy.cfg.numerical_cost_ms
                              if numerical_cost_ms is None else numerical_cost_ms),
    }


def simulate(trajectory: List[float], numerical_cost_ms: Optional[float] = None,
             free_mem_mb: Optional[float] = None) -> dict:
    """Replay a failure-score trajectory through the policy (offline demo)."""
    steps = [route_step(s, free_mem_mb, numerical_cost_ms) for s in trajectory]
    n_corr = sum(1 for s in steps if s["route"] == "numerical")
    return {
        "n_steps": len(steps),
        "n_numerical_corrections": n_corr,
        "correction_rate": round(n_corr / len(steps), 3) if steps else 0.0,
        "steps": steps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI layer (only built if fastapi is installed - it's in requirements.txt)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="Turingz - Engine Control Unit (Module 3)",
                  version="1.0")

    class RouteRequest(BaseModel):
        failure_score: float
        free_mem_mb: Optional[float] = None
        numerical_cost_ms: Optional[float] = None

    class ConfigUpdate(BaseModel):
        failure_threshold: Optional[float] = None
        latency_budget_ms: Optional[float] = None
        mem_headroom_mb: Optional[float] = None
        numerical_cost_ms: Optional[float] = None

    class SimulateRequest(BaseModel):
        trajectory: List[float]
        numerical_cost_ms: Optional[float] = None
        free_mem_mb: Optional[float] = None

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "engine-control-unit"}

    @app.get("/device")
    def device():
        info = cost_meter.device_info()
        info["free_mem_mb"] = round(_free_mem_mb(), 1)
        return info

    @app.get("/config")
    def get_config():
        c = _policy.cfg
        return {
            "failure_threshold": c.failure_threshold,
            "latency_budget_ms": c.latency_budget_ms,
            "mem_headroom_mb": c.mem_headroom_mb,
            "numerical_cost_ms": c.numerical_cost_ms,
        }

    @app.post("/config")
    def set_config(update: ConfigUpdate):
        c = _policy.cfg
        for k, v in update.dict().items():
            if v is not None:
                setattr(c, k, v)
        return get_config()

    @app.post("/route")
    def route(req: RouteRequest):
        return route_step(req.failure_score, req.free_mem_mb, req.numerical_cost_ms)

    @app.post("/simulate")
    def simulate_endpoint(req: SimulateRequest):
        return simulate(req.trajectory, req.numerical_cost_ms, req.free_mem_mb)

    _FASTAPI_OK = True
except Exception:  # pragma: no cover - fastapi not installed
    app = None
    _FASTAPI_OK = False


# ─────────────────────────────────────────────────────────────────────────────
def _self_demo() -> None:
    print("Engine Control Unit - self demo (no server)")
    print("device:", cost_meter.device_info())
    # a failure score that climbs as the ML solver drifts in extrapolation
    traj = [0.05, 0.1, 0.2, 0.35, 0.55, 0.7, 0.85, 0.95]
    result = simulate(traj)
    print(f"\nfailure trajectory: {traj}")
    for s in result["steps"]:
        print(f"  fail={s['failure_score']:<5} -> {s['route']:<18} "
              f"(HEI={s['step_hei']}) : {s['reason']}")
    print(f"\ncorrections triggered: {result['n_numerical_corrections']}/"
          f"{result['n_steps']}  (rate={result['correction_rate']})")
    if not _FASTAPI_OK:
        print("\n[note] fastapi not importable here; install requirements to serve HTTP.")


if __name__ == "__main__":
    _self_demo()
