from __future__ import annotations
import numpy as np
from .coupling import CouplingStub
from .trigger import SyntheticTrust
from .groundtruth import relative_l2

HORIZON = 1.0


class NumExact:
    name = "num-exact"
    def rollout(self, ic, x, t):
        ic = np.asarray(ic, float); t = np.asarray(t, float)
        return ic[None, :] * np.exp(-t)[:, None]


class MLDrift:
    name = "ml-drift"
    def rollout(self, ic, x, t):
        ic = np.asarray(ic, float); t = np.asarray(t, float)
        val = np.where(t <= HORIZON, np.exp(-t),
                       np.exp(-HORIZON) * (1.0 - 0.6 * (t - HORIZON)))
        return ic[None, :] * val[:, None]


def run():
    x = np.linspace(-1.0, 1.0, 512)
    t = np.linspace(0.0, 2.0, 200)
    ic = np.sin(np.pi * x)
    truth = ic[None, :] * np.exp(-t)[:, None]
    ml, num = MLDrift(), NumExact()
    trust = SyntheticTrust(horizon=HORIZON, width=0.05, flag_at=0.5)
    coupling = CouplingStub()
    u_ml = ml.rollout(ic, x, t)
    u_num = num.rollout(ic, x, t)
    u_hyb = coupling.rollout(ic, x, t, ml, num, trust)
    ext = t > HORIZON
    err = lambda u, m=None: relative_l2(u, truth, time_mask=m)
    print(f"{'method':<14}{'error (all t)':>16}{'error (t>1)':>16}")
    for name, u in [("pure-ML", u_ml), ("pure-numerical", u_num), ("hybrid", u_hyb)]:
        print(f"{name:<14}{err(u):>16.5f}{err(u, ext):>16.5f}")
    ok = (err(u_num) <= err(u_hyb) < err(u_ml))
    print("\nacceptance (num <= hybrid < ML):", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
