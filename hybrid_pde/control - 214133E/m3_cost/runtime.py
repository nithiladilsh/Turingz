from dataclasses import dataclass
import numpy as np
from . import groundtruth as G
from . import trigger as TR
from . import controller as CT
from . import config as C


@dataclass
class HybridRuntime:
    surrogate: object
    unit_num_ms: dict
    unit_ml_ms: float = float("nan")
    solver: str = "spectral"
    trust_mode: str = "synthetic"
    controller: object = None
    blend: int = 2

    def _trust(self, u_ml, u_true=None):
        if self.trust_mode == "physics":
            return TR.physics_residual_trust(u_ml)[0]
        if self.trust_mode == "oracle":
            if u_true is None:
                raise ValueError("oracle trust needs ground truth")
            return TR.oracle_trust(u_ml, u_true)[0]
        if self.trust_mode == "synthetic":
            return TR.synthetic_trust()
        if callable(self.trust_mode):
            return self.trust_mode(u_ml)
        raise ValueError(self.trust_mode)

    def run(self, ic_or_index, accuracy_target, u_true=None, return_field=False):
        u_ml = self.surrogate.predict_rollout(ic_or_index)
        ic = (G.make_ics()[ic_or_index] if np.isscalar(ic_or_index)
              and not isinstance(ic_or_index, np.ndarray) else ic_or_index)
        if u_true is None:
            try:
                u_true = G.colehopf_solve(ic)
            except Exception:
                u_true = None

        trust = self._trust(u_ml, u_true)
        if self.controller is not None:
            j, solver, chosen = self.controller.plan(accuracy_target, trust)
        else:
            tau = float(np.clip(1.0 - accuracy_target, 0.05, 0.95))
            _, j = TR.reliable_horizon(trust, tau)
            solver, chosen = self.solver, {"tau": tau}

        r = CT.evaluate_policy(ic, u_ml, u_true if u_true is not None else u_ml,
                               j, solver, self.unit_num_ms[solver],
                               self.unit_ml_ms, self.blend)
        report = {"accuracy_target": float(accuracy_target),
                  "switch_index": int(j), "switch_time": float(G.T_GRID[j]),
                  "corrector": solver,
                  "numerical_steps_spent": r["n_num_steps"],
                  "ml_steps_used": r["n_ml_steps"],
                  "wall_ms": r["wall_ms"],
                  "achieved_extrap_error": (r["err_extrap"] if u_true is not None else None),
                  "target_met": (bool(r["err_extrap"] <= accuracy_target)
                                 if u_true is not None else None),
                  "handover_jump": r["handover_jump"],
                  "controller_choice": chosen}
        if return_field:
            from . import coupling as CP
            field_u, _ = CP.switch_rollout(ic, u_ml, j, solver=solver,
                                           unit_num_ms=self.unit_num_ms[solver],
                                           unit_ml_ms=self.unit_ml_ms, blend=self.blend)
            return report, field_u
        return report
