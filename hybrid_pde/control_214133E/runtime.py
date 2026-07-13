from __future__ import annotations
import time
import numpy as np
from .contracts import CostReport, HybridResult
from .groundtruth import relative_l2


class HybridRuntime:
    def __init__(self, ml, num, trust, coupling, controller):
        self.ml = ml
        self.num = num
        self.trust = trust
        self.coupling = coupling
        self.controller = controller

    def run(self, ic, x, t, accuracy_target, reference=None):
        self.controller.configure(accuracy_target)
        self.controller.reset()
        if hasattr(self.trust, "reset"):
            self.trust.reset()
        t = np.asarray(t, dtype=float)
        u_ml = np.asarray(self.ml.rollout(ic, x, t), dtype=float)
        out = u_ml.copy()
        state = np.asarray(ic, dtype=float)
        prev_t = float(t[0])
        trust_curve = np.empty(len(t))
        switch_times = []
        ml_steps = 0
        corr = 0
        prev = False
        clock = time.perf_counter()
        for i in range(len(t)):
            cur = out[i - 1] if i > 0 else state
            tv, flag = self.trust(cur, float(t[i]))
            trust_curve[i] = tv
            dec = self.controller.decide(tv, flag, float(t[i]), i)
            if dec.correct:
                out[i] = self.coupling.correct(state, x, prev_t, float(t[i]), self.num)
                corr += 1
                if not prev:
                    switch_times.append(float(t[i]))
            else:
                out[i] = u_ml[i]
                ml_steps += 1
            state = out[i]
            prev_t = float(t[i])
            prev = dec.correct
        wall = time.perf_counter() - clock
        cost = CostReport(ml_steps=ml_steps, correction_steps=corr,
                          wall_time_s=wall, accuracy_target=float(accuracy_target))
        if reference is not None:
            err = relative_l2(out, np.asarray(reference, dtype=float))
            cost.achieved_error = err
            cost.met_target = bool(err <= accuracy_target)
        return HybridResult(out, cost, trust_curve, switch_times)
