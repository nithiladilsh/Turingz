from dataclasses import dataclass
import numpy as np
from . import groundtruth as G
from . import trigger as TR
from . import coupling as CP


def evaluate_policy(ic, u_ml, u_true, j_switch, solver, unit_num_ms, unit_ml_ms,
                    blend=2):
    u_hyb, cost = CP.switch_rollout(ic, u_ml, j_switch, solver=solver,
                                    unit_num_ms=unit_num_ms, unit_ml_ms=unit_ml_ms,
                                    blend=blend)
    err_extrap = G.rel_l2(u_hyb, u_true, G.EXTRAP_MASK)
    err_full = G.rel_l2(u_hyb, u_true, np.ones(u_true.shape[0], bool))
    j = int(np.clip(j_switch, 1, u_hyb.shape[0] - 2))
    jump = (np.linalg.norm(u_hyb[j + 1] - u_hyb[j]) /
            (np.linalg.norm(u_hyb[j]) + 1e-12))
    return {"j_switch": int(j_switch), "solver": solver,
            "err_extrap": float(err_extrap), "err_full": float(err_full),
            "n_num_steps": cost.n_num_steps, "n_ml_steps": cost.n_ml_steps,
            "numerical_step_equivalents": cost.numerical_step_equivalents,
            "wall_ms": cost.wall_ms, "handover_jump": float(jump)}


@dataclass
class AdaptiveController:
    unit_num_ms: dict
    unit_ml_ms: float = float("nan")
    solvers: tuple = ("fdm", "spectral")
    taus: tuple = tuple(np.round(np.linspace(0.05, 0.95, 19), 3))
    blend: int = 2
    _table: list = None

    def calibrate(self, train_ics, surrogate, trust_fn):
        rows = []
        for tau in self.taus:
            for solver in self.solvers:
                accs, costs, jumps = [], [], []
                for ic, idx in train_ics:
                    u_ml = surrogate.predict_rollout(idx)
                    u_true = G.colehopf_solve(ic)
                    trust = trust_fn(u_ml, u_true)
                    _, j = TR.reliable_horizon(trust, tau)
                    r = evaluate_policy(ic, u_ml, u_true, j, solver,
                                        self.unit_num_ms[solver], self.unit_ml_ms,
                                        self.blend)
                    accs.append(r["err_extrap"]); costs.append(r["numerical_step_equivalents"])
                    jumps.append(r["handover_jump"])
                rows.append({"tau": float(tau), "solver": solver,
                             "acc_mean": float(np.mean(accs)),
                             "acc_p90": float(np.percentile(accs, 90)),
                             "cost_mean": float(np.mean(costs)),
                             "jump_mean": float(np.mean(jumps))})
        self._table = rows
        return rows

    def plan(self, eps, trust, conservative=True):
        if self._table is None:
            raise RuntimeError("calibrate() before plan().")
        key = "acc_p90" if conservative else "acc_mean"
        feasible = [r for r in self._table if r[key] <= eps]
        if not feasible:
            chosen = min(self._table, key=lambda r: r[key])
        else:
            chosen = min(feasible, key=lambda r: r["cost_mean"])
        _, j = TR.reliable_horizon(trust, chosen["tau"])
        return j, chosen["solver"], chosen


@dataclass
class FixedController:
    unit_num_ms: dict
    unit_ml_ms: float = float("nan")
    solver: str = "spectral"
    switch_time: float = 1.0

    def plan(self, eps, trust=None):
        j = int(np.searchsorted(G.T_GRID, self.switch_time))
        return j, self.solver, {"tau": None, "solver": self.solver}


@dataclass
class OracleController:
    unit_num_ms: dict
    unit_ml_ms: float = float("nan")
    solver: str = "spectral"

    def plan(self, eps, u_ml, u_true):
        err = G.rel_l2(u_ml, u_true)
        over = np.where(err > eps)[0]
        j = int(over[0]) if len(over) else len(err) - 1
        return j, self.solver, {"tau": "oracle", "solver": self.solver}
