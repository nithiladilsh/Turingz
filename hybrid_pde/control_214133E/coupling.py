from __future__ import annotations
import numpy as np
from .contracts import Solver, Coupling, TrustSignal


class CouplingStub:
    name = "coupling-stub"

    def correct(self, state, x, t0, t1, num):
        state = np.asarray(state, dtype=float)
        tau = np.array([0.0, float(t1) - float(t0)])
        return np.asarray(num.rollout(state, x, tau))[-1]

    def rollout(self, ic, x, t, ml, num, trigger):
        t = np.asarray(t, dtype=float)
        u_ml = np.asarray(ml.rollout(ic, x, t), dtype=float)
        n_t = len(t)
        switch = None
        for i in range(n_t):
            _, flag = trigger(u_ml[i], float(t[i]))
            if flag:
                switch = i
                break
        if switch is None:
            return u_ml
        out = u_ml.copy()
        if switch == 0:
            seed, t0, sl = np.asarray(ic, float), float(t[0]), slice(0, n_t)
        else:
            seed, t0, sl = u_ml[switch - 1], float(t[switch - 1]), slice(switch, n_t)
        tau = np.concatenate([[0.0], t[sl] - t0])
        num_tail = np.asarray(num.rollout(seed, x, tau), dtype=float)
        out[sl] = num_tail[1:]
        return out


class RealCoupling:
    def __init__(self, m2_coupling: Coupling):
        self._m2 = m2_coupling

    def correct(self, state, x, t0, t1, num):
        return self._m2.correct(state, x, t0, t1, num)

    def rollout(self, ic, x, t, ml, num, trigger):
        return self._m2.rollout(ic, x, t, ml, num, trigger)
