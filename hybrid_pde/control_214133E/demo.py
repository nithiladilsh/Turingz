from __future__ import annotations
import numpy as np
from .runtime import HybridRuntime
from .controller import AdaptiveController, thresholds_for_target
from .groundtruth import relative_l2


def default_standins():
    from ._smoke import MLDrift, NumExact, HORIZON
    from .trigger import SyntheticTrust
    from .coupling import CouplingStub
    x = np.linspace(-1, 1, 512)
    t = np.linspace(0, 2, 200)
    ic = np.sin(np.pi * x)
    reference = ic[None, :] * np.exp(-t)[:, None]
    return dict(ml=MLDrift(), num=NumExact(),
                trust=SyntheticTrust(HORIZON, width=0.08, flag_at=0.0),
                coupling=CouplingStub(), ic=ic, x=x, t=t, reference=reference)


def demo_frame(target, ml, num, trust, coupling, ic, x, t, reference, ml_step_s=1.0, num_step_s=10.0):
    lo, hi = thresholds_for_target(target)
    res = HybridRuntime(ml, num, trust, coupling, AdaptiveController(lo, hi)).run(ic, x, t, target, reference=reference)
    mlf = np.asarray(ml.rollout(ic, x, t))
    numf = np.asarray(num.rollout(ic, x, t))
    nt = len(t)
    comparison = {
        "pure-ML": {"error": relative_l2(mlf, reference), "cost": nt * ml_step_s},
        "pure-numerical": {"error": relative_l2(numf, reference), "cost": nt * num_step_s},
        "hybrid": {"error": res.cost.achieved_error,
                   "cost": res.cost.ml_steps * ml_step_s + res.cost.correction_steps * num_step_s},
    }
    return {"x": np.asarray(x), "t": np.asarray(t), "truth": np.asarray(reference),
            "ml": mlf, "num": numf, "hybrid": res.u, "trust": res.trust_curve,
            "switch": res.switch_times, "cost": res.cost, "comparison": comparison}
