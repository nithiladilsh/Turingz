"""
================================================================================
SCALABILITY  /  COMPUTATIONAL COMPLEXITY
Team : Turingz   File : module3_deployment/scalability.py

Interim metric: "Computational Complexity - assesses the solver's scalability
with increasing grid sizes."

We measure inference latency and memory while sweeping the OUTPUT grid size
(nx x nt) the solver is asked to produce, then fit a power law on the log-log
curve to get an empirical complexity exponent:

        latency  ~=  c * (n_points) ** p

p ~ 1 means linear in output size; p > 1 means it scales worse. This is an
empirical, deployment-relevant complexity - exactly what a cost study reports.

Built on cost_meter.measure_inference, so the timing/memory convention is the
same one used everywhere else in Module 3.
================================================================================
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from . import cost_meter


def _interp_ic(ic: np.ndarray, x_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    """Resample an IC onto a new spatial grid (periodic linear interp). Unused by
    grid_sweep now - kept as a utility."""
    period = (x_src[-1] - x_src[0]) if len(x_src) > 1 else None
    return np.interp(x_dst, x_src, ic, period=period)


def grid_sweep(solver, reference: Dict, sample_index: int = 0,
               nx_list: Optional[List[int]] = None,
               nt_list: Optional[List[int]] = None,
               repeats: int = 5, warmup: int = 1) -> Dict:
    """Measure inference cost across a range of OUTPUT grid sizes.

    Returns per-grid latency/memory plus a fitted complexity exponent.
    """
    x = np.asarray(reference["x"], dtype=np.float64)
    t = np.asarray(reference["t"], dtype=np.float64)
    ic_native = reference["u"][sample_index, 0, :]

    nx_native, nt_native = len(x), len(t)
    if nx_list is None:
        nx_list = sorted({max(8, nx_native // 8), nx_native // 4,
                          nx_native // 2, nx_native})
    if nt_list is None:
        nt_list = [nt_native] * len(nx_list)

    x_lo, x_hi = float(x[0]), float(x[-1])
    t_lo, t_hi = float(t[0]), float(t[-1])

    # IMPORTANT: keep the IC at its NATIVE resolution. Operator models read the
    # IC at fixed sensor indices on the native grid, so the input must not be
    # resampled. We vary only the OUTPUT query grid (x_grid, t_grid) - that is
    # exactly "scalability with increasing grid size" for the produced field.
    points: List[Dict] = []
    for nx, nt in zip(nx_list, nt_list):
        x_grid = np.linspace(x_lo, x_hi, int(nx), endpoint=True)
        t_grid = np.linspace(t_lo, t_hi, int(nt), endpoint=True)
        m = cost_meter.measure_inference(solver, ic_native, x_grid, t_grid,
                                         repeats=repeats, warmup=warmup)
        points.append({
            "nx": int(nx), "nt": int(nt), "n_points": int(nx * nt),
            "latency_ms_median": m["latency_ms"]["median"],
            "peak_cpu_mb": m["memory"]["peak_cpu_mb"],
            "cpu_delta_mb": m["memory"]["cpu_delta_mb"],
            "points_per_s": m["throughput"]["points_per_s"],
        })

    exponent = _fit_loglog_exponent(
        [p["n_points"] for p in points],
        [p["latency_ms_median"] for p in points],
    )
    return {
        "solver": getattr(solver, "name", type(solver).__name__),
        "sample": int(sample_index),
        "points": points,
        "complexity_exponent_p": exponent,
        "interpretation": _interpret_exponent(exponent),
    }


def _fit_loglog_exponent(n_points: List[int], latency: List[float]) -> Optional[float]:
    """Slope of log(latency) vs log(n_points): the empirical complexity p."""
    xs = np.log(np.asarray(n_points, dtype=np.float64))
    ys = np.log(np.asarray(latency, dtype=np.float64) + 1e-12)
    if len(xs) < 2 or np.allclose(xs, xs[0]):
        return None
    slope = float(np.polyfit(xs, ys, 1)[0])
    return round(slope, 4)


def _interpret_exponent(p: Optional[float]) -> str:
    if p is None:
        return "not enough distinct grid sizes to fit a complexity exponent"
    if p < 0.5:
        return f"sub-linear (p={p}): cost barely grows with grid size"
    if p < 1.3:
        return f"~linear (p={p}): cost grows roughly proportionally to output size"
    if p < 2.2:
        return f"super-linear (p={p}): cost grows faster than the grid"
    return f"steep (p={p}): poor scaling with grid size"
