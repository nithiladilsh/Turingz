from __future__ import annotations
import numpy as np


class CoarseDriftMonitor:
    def __init__(self, num, x, check_every=5, threshold=0.05):
        self.num = num
        self.x = x
        self.check_every = check_every
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.anchor = None
        self.anchor_t = None
        self.n = 0
        self._trust = 1.0
        self._flag = False
        self.last_div = 0.0

    def mark_corrected(self, state, t):
        self.anchor = np.asarray(state, dtype=float).copy()
        self.anchor_t = float(t)

    def __call__(self, state, t):
        state = np.asarray(state, dtype=float)
        t = float(t)
        if self.anchor is None:
            self.anchor = state.copy()
            self.anchor_t = t
        self.n += 1
        if self.n % self.check_every == 0 and t > self.anchor_t:
            ref = np.asarray(self.num.rollout(self.anchor, self.x, np.array([0.0, t - self.anchor_t])))[-1]
            div = float(np.linalg.norm(state - ref) / (np.linalg.norm(ref) + 1e-12))
            self.last_div = div
            self._trust = float(np.clip(1.0 - div / self.threshold, 0.0, 1.0))
            self._flag = bool(div > self.threshold)
        return self._trust, self._flag
