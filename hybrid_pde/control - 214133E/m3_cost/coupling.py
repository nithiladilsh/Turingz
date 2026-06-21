from dataclasses import dataclass
import numpy as np
from . import groundtruth as G


@dataclass
class CostReport:
    n_ml_steps: int = 0
    n_num_steps: int = 0
    solver: str = "spectral"
    unit_ml_ms: float = float("nan")
    unit_num_ms: float = 0.0
    blend_steps: int = 0

    @property
    def numerical_step_equivalents(self):
        return self.n_num_steps

    @property
    def wall_ms(self):
        ml = 0.0 if np.isnan(self.unit_ml_ms) else self.n_ml_steps * self.unit_ml_ms
        return ml + self.n_num_steps * self.unit_num_ms

    def as_dict(self):
        return {"n_ml_steps": self.n_ml_steps, "n_num_steps": self.n_num_steps,
                "solver": self.solver, "unit_ml_ms": self.unit_ml_ms,
                "unit_num_ms": self.unit_num_ms,
                "numerical_step_equivalents": self.numerical_step_equivalents,
                "wall_ms": self.wall_ms}


def switch_rollout(ic, u_ml, j_switch, solver="spectral",
                   unit_num_ms=0.0, unit_ml_ms=float("nan"), blend=0):
    advance = G.NUMERICAL_SOLVERS[solver]["advance"]
    nt = u_ml.shape[0]
    j_switch = int(np.clip(j_switch, 0, nt - 1))
    u = np.empty_like(u_ml)
    u[:j_switch + 1] = u_ml[:j_switch + 1]
    cost = CostReport(n_ml_steps=j_switch, solver=solver,
                      unit_num_ms=unit_num_ms, unit_ml_ms=unit_ml_ms,
                      blend_steps=int(blend))
    state = u_ml[j_switch].copy()
    for j in range(j_switch + 1, nt):
        state = advance(state, G.T_GRID[j - 1], G.T_GRID[j])
        cost.n_num_steps += 1
        k = j - j_switch
        if blend and k <= blend:
            w = k / (blend + 1)
            state = (1 - w) * u_ml[j] + w * state
        u[j] = state
    return u, cost


def masked_rollout(ic, u_ml, correct_mask, solver="spectral",
                   unit_num_ms=0.0, unit_ml_ms=float("nan")):
    advance = G.NUMERICAL_SOLVERS[solver]["advance"]
    nt = u_ml.shape[0]
    u = np.empty_like(u_ml); u[0] = u_ml[0]
    cost = CostReport(solver=solver, unit_num_ms=unit_num_ms, unit_ml_ms=unit_ml_ms)
    state = u_ml[0].copy()
    for j in range(1, nt):
        if correct_mask[j]:
            state = advance(state, G.T_GRID[j - 1], G.T_GRID[j])
            cost.n_num_steps += 1
        else:
            state = u_ml[j]
            cost.n_ml_steps += 1
        u[j] = state
    return u, cost


def pure_ml(u_ml, unit_ml_ms=float("nan")):
    cost = CostReport(n_ml_steps=u_ml.shape[0] - 1, unit_ml_ms=unit_ml_ms,
                      solver="none")
    return u_ml.copy(), cost


def pure_numerical(ic, solver="spectral", unit_num_ms=0.0):
    u = G.NUMERICAL_SOLVERS[solver]["solve"](ic)
    cost = CostReport(n_num_steps=u.shape[0] - 1, solver=solver,
                      unit_num_ms=unit_num_ms)
    return u, cost
