import numpy as np


class M2Coupling:
    name = "m2-coupling-214050V"

    #re-anchor primitive that can be scheduled
    def correct(self, state, x, t0, t1, num):
        state = np.asarray(state, dtype=float)
        tau = np.array([0.0, float(t1) - float(t0)])
        return np.asarray(num.rollout(state, x, tau), dtype=float)[-1]

    #one-way switch
    def rollout(self, ic, x, t, ml, num, trigger):
        t = np.asarray(t, dtype=float)
        u_ml = np.asarray(ml.rollout(ic, x, t), dtype=float)
        switch = None
        for i in range(len(t)):
            _, flag = trigger(u_ml[i], float(t[i]))
            if flag:
                switch = i
                break
        if switch is None:                       
            return u_ml
        out = u_ml.copy()
        sl = slice(switch, len(t))
        tau = t[sl] - t[switch]                  
        tail = np.asarray(num.rollout(u_ml[switch], x, tau), dtype=float)
        out[sl] = tail                           
        return out
