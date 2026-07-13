from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np


@dataclass
class SolverCost:
    name: str
    latency_s: float = float("nan")
    latency_std: float = float("nan")
    per_step_s: float = float("nan")
    scaling_exponent: float = float("nan")


class Profiler:
    def __init__(self, repeats=30, warmup=5):
        self.repeats = repeats
        self.warmup = warmup

    @staticmethod
    def timeit(fn, repeats=30, warmup=5):
        for _ in range(warmup):
            fn()
        s = np.empty(repeats)
        for i in range(repeats):
            a = time.perf_counter()
            fn()
            s[i] = time.perf_counter() - a
        std = float(s.std(ddof=1)) if repeats > 1 else 0.0
        return float(s.mean()), std, float(np.median(s))

    @staticmethod
    def scaling_fit(sizes, times):
        sizes = np.asarray(sizes, dtype=float)
        times = np.asarray(times, dtype=float)
        return float(np.polyfit(np.log(sizes), np.log(times), 1)[0])

    def measure_latency(self, solver, ic, x, t):
        return self.timeit(lambda: solver.rollout(ic, x, t), self.repeats, self.warmup)

    def scaling_exponent(self, solver, ic, x, t, n_list):
        sizes, times = [], []
        for n in n_list:
            tt = np.linspace(float(t.min()), float(t.max()), int(n))
            _, _, med = self.timeit(lambda: solver.rollout(ic, x, tt),
                                    max(3, self.repeats // 3), self.warmup)
            sizes.append(int(n) * len(x))
            times.append(med)
        return self.scaling_fit(sizes, times), sizes, times

    def profile(self, solver, ic, x, t, n_list=None):
        _, std, med = self.measure_latency(solver, ic, x, t)
        per_step = med / max(1, len(t))
        exp = float("nan")
        if n_list is not None:
            exp, _, _ = self.scaling_exponent(solver, ic, x, t, n_list)
        return SolverCost(getattr(solver, "name", "solver"), med, std, per_step, exp)
