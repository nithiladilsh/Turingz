"""
================================================================================
MODULE 3 - DEPLOYMENT & COST ANALYSIS
Team : Turingz   Owner : 214133E (Mendis B.N.D.)
Package : module3_deployment/

This package is the deployment / computational-cost workstream. It is built
ENTIRELY on top of the shared framework (common/) and the AbstractSolver
contract - it never re-implements accuracy, the dataset split, or model
loading, so the cross-model comparison stays consistent with the other two
modules (reliability, robustness).

What it provides
----------------
    cost_meter.py    : measure runtime, peak memory, throughput, #params
                       (wraps fit + rollout uniformly for every solver)
    scalability.py   : computational complexity - cost vs grid size
    cost_accuracy.py : join shared accuracy (common.metrics) with cost
    hei.py           : Hybrid Efficiency Index (deployment efficiency score),
                       with the PINN per-IC accounting baked in
    run_cost.py      : one CLI - load any model via the shared loader, measure,
                       write results/cost/<model>.json in the uniform schema
    plots.py         : cost-accuracy trade-off + comparison charts
    app.py           : latency-aware FastAPI orchestrator (the "Engine Control
                       Unit") that uses the HEI to route ML vs numerical and
                       throttles to avoid OOM

Interim metrics covered (from the interim slide, Module 3):
    runtime, memory usage, computational complexity, accuracy-vs-cost, throughput
    + outputs: cost-accuracy curves, comparison charts, deployment recommendation
================================================================================
"""

from . import cost_meter, scalability, cost_accuracy, hei  # noqa: F401

__all__ = ["cost_meter", "scalability", "cost_accuracy", "hei"]
