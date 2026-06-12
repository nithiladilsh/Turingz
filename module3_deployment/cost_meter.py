"""
================================================================================
COST METER  -  uniform computational-cost measurement for every solver
Team : Turingz   File : module3_deployment/cost_meter.py

The handoff (Section 5.3) says Module 3 must:
    * read training time from each model's fit result (wall_time_s) and model
      size from num_parameters() - both already uniform across the three;
    * wrap fit AND rollout in a cost meter to record inference time, peak
      memory and throughput identically for all models.

This file is that cost meter. It treats every solver purely through the
AbstractSolver contract (.fit / .rollout / .num_parameters / .name), so PINN,
FNO and DeepONet are measured by exactly the same code - no model-specific
branches, nothing re-implemented.

Memory is sampled in a background thread while the measured call runs, so the
peak is captured even if it happens mid-call. CPU RSS is always available via
psutil; GPU memory is captured via pynvml when a CUDA device is present, and
degrades gracefully to "not available" otherwise.
================================================================================
"""

from __future__ import annotations

import gc
import time
import threading
import statistics
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import psutil


# ─────────────────────────────────────────────────────────────────────────────
# Optional GPU memory probe (pynvml). Absent on CPU-only machines - that's fine.
# ─────────────────────────────────────────────────────────────────────────────
try:
    import pynvml  # type: ignore
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


def _gpu_mem_used_mb() -> Optional[float]:
    """Currently-used GPU memory in MB across device 0, or None if no GPU."""
    if not _NVML_OK:
        return None
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        return info.used / (1024.0 ** 2)
    except Exception:
        return None


def device_info() -> Dict[str, Any]:
    """Describe the hardware so cost numbers are interpretable later."""
    info: Dict[str, Any] = {
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024.0 ** 3), 2),
        "gpu_available": False,
        "gpu_name": None,
    }
    try:
        import torch  # lazy: cost meter must import on torch-free machines
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        if _NVML_OK:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                info["gpu_available"] = True
                name = pynvml.nvmlDeviceGetName(h)
                info["gpu_name"] = name.decode() if isinstance(name, bytes) else name
            except Exception:
                pass
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Background peak-memory sampler
# ─────────────────────────────────────────────────────────────────────────────
class _PeakMemorySampler:
    """Samples process RSS (and GPU memory) on a thread; records the peak.

    Usage:
        with _PeakMemorySampler() as s:
            <do work>
        s.peak_cpu_mb, s.peak_gpu_mb, s.baseline_cpu_mb
    """

    def __init__(self, interval_s: float = 0.01):
        self.interval_s = interval_s
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.baseline_cpu_mb = 0.0
        self.peak_cpu_mb = 0.0
        self.baseline_gpu_mb: Optional[float] = None
        self.peak_gpu_mb: Optional[float] = None
        # torch-CUDA per-process GPU memory: isolates the MODEL's own footprint
        # (more meaningful than pynvml's system-wide reading) and works even when
        # pynvml is not installed. Guarded so the meter still imports without torch.
        self._torch = None
        self._cuda = False
        try:
            import torch as _t
            if _t.cuda.is_available():
                self._torch = _t
                self._cuda = True
        except Exception:
            pass
        self.gpu_base_mb: Optional[float] = None
        self.gpu_peak_mb: Optional[float] = None

    def _rss_mb(self) -> float:
        return self._proc.memory_info().rss / (1024.0 ** 2)

    def _run(self):
        while not self._stop.is_set():
            self.peak_cpu_mb = max(self.peak_cpu_mb, self._rss_mb())
            g = _gpu_mem_used_mb()
            if g is not None:
                self.peak_gpu_mb = g if self.peak_gpu_mb is None else max(self.peak_gpu_mb, g)
            time.sleep(self.interval_s)

    def __enter__(self) -> "_PeakMemorySampler":
        gc.collect()
        self.baseline_cpu_mb = self._rss_mb()
        self.peak_cpu_mb = self.baseline_cpu_mb
        self.baseline_gpu_mb = _gpu_mem_used_mb()
        self.peak_gpu_mb = self.baseline_gpu_mb
        if self._cuda:
            try:
                self._torch.cuda.synchronize()
                self._torch.cuda.reset_peak_memory_stats()
                self.gpu_base_mb = self._torch.cuda.memory_allocated() / (1024.0 ** 2)
            except Exception:
                self._cuda = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        # one final reading in case the peak landed right at the end
        self.peak_cpu_mb = max(self.peak_cpu_mb, self._rss_mb())
        g = _gpu_mem_used_mb()
        if g is not None:
            self.peak_gpu_mb = g if self.peak_gpu_mb is None else max(self.peak_gpu_mb, g)
        if self._cuda:
            try:
                self._torch.cuda.synchronize()
                self.gpu_peak_mb = self._torch.cuda.max_memory_allocated() / (1024.0 ** 2)
            except Exception:
                pass

    def report(self) -> Dict[str, Optional[float]]:
        cpu_delta = round(self.peak_cpu_mb - self.baseline_cpu_mb, 3)
        gpu_delta = None
        peak_gpu = round(self.peak_gpu_mb, 3) if self.peak_gpu_mb is not None else None
        if self.peak_gpu_mb is not None and self.baseline_gpu_mb is not None:
            gpu_delta = round(self.peak_gpu_mb - self.baseline_gpu_mb, 3)
        # Prefer torch-CUDA per-process numbers when available — they isolate the
        # model's own GPU allocation instead of system-wide GPU usage.
        gpu_source = "pynvml" if peak_gpu is not None else None
        if self.gpu_peak_mb is not None:
            peak_gpu = round(self.gpu_peak_mb, 3)
            if self.gpu_base_mb is not None:
                gpu_delta = round(self.gpu_peak_mb - self.gpu_base_mb, 3)
            gpu_source = "torch.cuda"
        return {
            "peak_cpu_mb": round(self.peak_cpu_mb, 3),
            "cpu_delta_mb": cpu_delta,
            "peak_gpu_mb": peak_gpu,
            "gpu_delta_mb": gpu_delta,
            "gpu_source": gpu_source,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Loss extraction - models report final loss under slightly different keys
# (FNO: final_loss ; PINN: final_loss_train/test ; DeepONet: final_train_loss).
# We surface whatever is present without assuming one schema.
# ─────────────────────────────────────────────────────────────────────────────
def _extract_final_loss(fit_info: Dict[str, Any]) -> Optional[float]:
    for k in ("final_loss", "final_train_loss", "final_loss_train", "final_val_loss"):
        if k in fit_info and fit_info[k] is not None:
            v = fit_info[k]
            try:
                return float(np.sum(np.asarray(v, dtype=float)))
            except Exception:
                continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def measure_training(solver, dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Run solver.fit(dataset) once under the memory sampler.

    Captures the model's own wall_time_s (preferred - it times only the train
    loop) plus peak memory and parameter count. Use this when you want a single
    pass that BOTH trains and records the training cost. (If you retrain via the
    official train.py scripts instead, read wall_time_s from their saved logs -
    this function is the in-meter alternative.)
    """
    name = getattr(solver, "name", type(solver).__name__)
    with _PeakMemorySampler() as sampler:
        t0 = time.perf_counter()
        fit_info = solver.fit(dataset)
        wall_outer = time.perf_counter() - t0
    fit_info = fit_info or {}
    return {
        "phase": "training",
        "solver": name,
        # wall_time_s is the model's own measurement; wall_time_outer_s is ours
        "wall_time_s": float(fit_info.get("wall_time_s", wall_outer)),
        "wall_time_outer_s": round(wall_outer, 4),
        "n_parameters": int(fit_info.get("n_parameters", solver.num_parameters())),
        "final_loss": _extract_final_loss(fit_info),
        "memory": sampler.report(),
    }


def _torch_cuda():
    """Return the torch module if a usable CUDA device is present, else None."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch
    except Exception:
        pass
    return None


def measure_inference(solver, ic: np.ndarray, x_grid: np.ndarray,
                      t_grid: np.ndarray, repeats: int = 10,
                      warmup: int = 2) -> Dict[str, Any]:
    """Time full-horizon rollout(ic, x_grid, t_grid) and record memory + throughput.

    repeats : number of timed rollout calls (median is the headline number).
    warmup  : untimed calls first, so one-off costs (lazy CUDA init, JIT,
              allocator warm-up) don't pollute the measurement.

    Throughput is reported two ways:
        rollouts_per_s : full-field solves per second (deployment-level)
        points_per_s   : grid points produced per second (scale-normalised)
    """
    name = getattr(solver, "name", type(solver).__name__)
    nt, nx = len(t_grid), len(x_grid)
    n_points = nt * nx

    for _ in range(max(0, warmup)):
        solver.rollout(ic, x_grid, t_grid)

    latencies_ms: List[float] = []
    with _PeakMemorySampler() as sampler:
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            out = solver.rollout(ic, x_grid, t_grid)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    out = np.asarray(out)

    median_s = statistics.median(latencies_ms) / 1000.0
    return {
        "phase": "inference",
        "solver": name,
        "grid": {"nt": int(nt), "nx": int(nx), "n_points": int(n_points)},
        "repeats": int(repeats),
        "latency_ms": {
            "mean": round(statistics.mean(latencies_ms), 4),
            "median": round(statistics.median(latencies_ms), 4),
            "min": round(min(latencies_ms), 4),
            "max": round(max(latencies_ms), 4),
            "std": round(statistics.pstdev(latencies_ms), 4) if len(latencies_ms) > 1 else 0.0,
        },
        "throughput": {
            "rollouts_per_s": round(1.0 / median_s, 4) if median_s > 0 else None,
            "points_per_s": round(n_points / median_s, 2) if median_s > 0 else None,
        },
        "memory": sampler.report(),
        "output_shape": list(out.shape),
    }


def measure_solver(solver, reference: Dict[str, Any],
                   sample_index: int, repeats: int = 10,
                   warmup: int = 2) -> Dict[str, Any]:
    """Convenience: measure INFERENCE cost of an already-loaded solver on one IC
    from a loaded reference dict (see common.evaluation.load_reference).

    Training cost is handled separately (measure_training, or read wall_time_s
    from the train logs) because by the time we load a checkpoint the training
    has already happened.
    """
    u = reference["u"]
    x = reference["x"]
    t = reference["t"]
    ic = u[sample_index, 0, :]
    inf = measure_inference(solver, ic, x, t, repeats=repeats, warmup=warmup)
    inf["sample"] = int(sample_index)
    inf["n_parameters"] = int(solver.num_parameters())
    return inf
