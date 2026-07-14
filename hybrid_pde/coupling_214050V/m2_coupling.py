"""
M2 coupling adapter for the Module 3 hybrid engine (control_214133E).

Implements the control_214133E.contracts.Coupling protocol so Module 3's runtime
calls THIS module's verified ML-to-numerical handoff instead of its internal stub.

  rollout(ic, x, t, ml, num, trigger)
      The validated HARD one-way switch (Module 2's main method): run the ML solver,
      switch at the first trust trigger, and continue numerically from the handed-over
      ML state to the final time.
  correct(state, x, t0, t1, num)
      Single-step re-anchor primitive: advance a handed-over state by one output
      interval with the injected numerical solver. This is the primitive that Module
      3's adaptive controller schedules (enabling its periodic-correction policy).

The numerical continuation is the project's verified pseudo-spectral scheme; the
restart maths is restart_spectral.solve_from, proven bit-for-bit equal to the team
solver in verify_restart.py. Here the injected `num` solver is used so the coupling
stays independent of any particular numerical backend.
"""
import numpy as np


class M2Coupling:
    name = "m2-coupling-214050V"

    def correct(self, state, x, t0, t1, num):
        state = np.asarray(state, dtype=float)
        tau = np.array([0.0, float(t1) - float(t0)])
        return np.asarray(num.rollout(state, x, tau), dtype=float)[-1]

    def rollout(self, ic, x, t, ml, num, trigger):
        t = np.asarray(t, dtype=float)
        u_ml = np.asarray(ml.rollout(ic, x, t), dtype=float)
        switch = None
        for i in range(len(t)):
            _, flag = trigger(u_ml[i], float(t[i]))
            if flag:
                switch = i
                break
        if switch is None:                       # trust never fires -> stay on ML
            return u_ml
        out = u_ml.copy()
        sl = slice(switch, len(t))
        tau = t[sl] - t[switch]                  # tau[0] = 0  (re-anchor at the switch)
        tail = np.asarray(num.rollout(u_ml[switch], x, tau), dtype=float)
        out[sl] = tail                           # out[switch] == handed-over ML state
        return out
