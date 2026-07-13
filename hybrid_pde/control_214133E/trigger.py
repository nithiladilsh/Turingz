from __future__ import annotations
import numpy as np
from .contracts import TrustSignal


class SyntheticTrust:
    def __init__(self, horizon, width=0.1, flag_at=0.5):
        self.horizon = horizon
        self.width = width
        self.flag_at = flag_at

    def __call__(self, state, t):
        trust = 1.0 / (1.0 + np.exp(-(self.horizon - t) / self.width))
        return float(trust), bool(trust < self.flag_at)


class RealTrust:
    def __init__(self, m1_estimator: TrustSignal):
        self._m1 = m1_estimator

    def __call__(self, state, t):
        return self._m1(state, t)


class TrustMonitorAdapter:
    def __init__(self, monitor):
        self._m = monitor

    def __call__(self, state, t):
        r = self._m.update(state, t)
        return float(r["trust"]), (not bool(r["ok"]))

    def reset(self):
        self._m.reset()
