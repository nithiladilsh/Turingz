"""
================================================================================
PLOTS  -  cost-accuracy trade-off + performance comparison charts
Team : Turingz   File : module3_deployment/plots.py

Renders the interim "Outputs": cost-accuracy trade-off curves and performance
comparison charts. Pure matplotlib (Agg backend), so it runs headless on any
machine. Inputs are the trade_off dict from cost_accuracy.build_trade_off.
================================================================================
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def render_trade_off(trade: Dict, out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    rows = trade["ranked"]
    written = []

    # ── 1. cost-accuracy scatter (latency vs accuracy, bubble = #params) ──────
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        lat = r.get("latency_s") or 0.0
        rel = r.get("relative_l2") or 0.0
        params = r.get("n_parameters") or 1
        size = 120 + (params ** 0.5)  # bubble grows with model size
        ax.scatter(lat, rel, s=size, alpha=0.7, edgecolors="k")
        ax.annotate(f"{r['name']}\nHEI={r['HEI']}", (lat, rel),
                    textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.set_xlabel("Inference latency  (s / rollout)  -  lower is better")
    ax.set_ylabel(f"Error  ({trade['accuracy_key']})  -  lower is better")
    ax.set_title("Cost-Accuracy Trade-off  (bubble size = #parameters)")
    ax.grid(True, alpha=0.3)
    p1 = os.path.join(out_dir, "cost_accuracy_tradeoff.png")
    fig.tight_layout(); fig.savefig(p1, dpi=140); plt.close(fig)
    written.append(p1)

    # ── 2. HEI ranking bar ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r["name"] for r in rows]
    heis = [r["HEI"] for r in rows]
    ax.barh(names, heis, color="#4C78A8")
    ax.invert_yaxis()
    ax.set_xlabel("Hybrid Efficiency Index (higher = more deployable)")
    ax.set_title("Deployment Efficiency Ranking (HEI)")
    for i, v in enumerate(heis):
        ax.text(v, i, f" {v}", va="center", fontsize=8)
    p2 = os.path.join(out_dir, "hei_ranking.png")
    fig.tight_layout(); fig.savefig(p2, dpi=140); plt.close(fig)
    written.append(p2)

    # ── 3. performance comparison (latency / memory / params) ────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = [
        ("latency_ms_median", "Latency (ms/rollout)"),
        ("memory_mb", "Peak memory delta (MB)"),
        ("n_parameters", "Parameters"),
    ]
    for ax, (key, label) in zip(axes, metrics):
        vals = [r.get(key) or 0 for r in rows]
        ax.bar(names, vals, color="#72B7B2")
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.suptitle("Performance Comparison")
    p3 = os.path.join(out_dir, "performance_comparison.png")
    fig.tight_layout(); fig.savefig(p3, dpi=140); plt.close(fig)
    written.append(p3)

    return written
