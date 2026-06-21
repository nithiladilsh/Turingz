import numpy as np
from . import groundtruth as G
from . import controller as CT
from . import coupling as CP


def policy_cloud(ic, u_ml, u_true, unit_num, unit_ml=float("nan"),
                 solvers=("fdm", "spectral"), switch_grid=None, blend=2):
    nt = u_ml.shape[0]
    if switch_grid is None:
        switch_grid = list(range(2, nt, 8)) + [nt - 1]
    rows = []
    for solver in solvers:
        for j in switch_grid:
            rows.append(CT.evaluate_policy(ic, u_ml, u_true, j, solver,
                                           unit_num[solver], unit_ml, blend))
    return rows


def baselines(ic, u_ml, u_true, unit_num, unit_ml=float("nan"),
              ref_solver="spectral"):
    _, c_ml = CP.pure_ml(u_ml, unit_ml)
    u_num, c_num = CP.pure_numerical(ic, ref_solver, unit_num[ref_solver])
    full = np.ones(u_true.shape[0], bool)
    return {
        "pure_ml": {"err_extrap": G.rel_l2(u_ml, u_true, G.EXTRAP_MASK),
                    "err_full": G.rel_l2(u_ml, u_true, full),
                    "numerical_step_equivalents": 0, "wall_ms": c_ml.wall_ms},
        "pure_numerical": {"err_extrap": G.rel_l2(u_num, u_true, G.EXTRAP_MASK),
                           "err_full": G.rel_l2(u_num, u_true, full),
                           "numerical_step_equivalents": c_num.n_num_steps,
                           "wall_ms": c_num.wall_ms},
    }


def pareto_front(rows, cost_key="numerical_step_equivalents", acc_key="err_extrap"):
    pts = sorted(rows, key=lambda r: (r[cost_key], r[acc_key]))
    front, best_acc = [], np.inf
    for r in pts:
        if r[acc_key] < best_acc - 1e-12:
            front.append(r); best_acc = r[acc_key]
    return front


def _err_at_cost(front, budget, acc_key, cost_key):
    feasible = [r for r in front if r[cost_key] <= budget]
    return min((r[acc_key] for r in feasible), default=None)


def _cost_for_acc(front, target, acc_key, cost_key):
    ok = [r for r in front if r[acc_key] <= target]
    return min((r[cost_key] for r in ok), default=None)


def dominates_baselines(front, base, cost_key="numerical_step_equivalents",
                        acc_key="err_extrap"):
    num_acc = base["pure_numerical"][acc_key]
    num_cost = base["pure_numerical"][cost_key]
    ml_acc = base["pure_ml"][acc_key]
    floor = min(r[acc_key] for r in front)
    cost_at_floor = min(r[cost_key] for r in front if r[acc_key] <= floor * 1.1)
    half = _err_at_cost(front, 0.5 * num_cost, acc_key, cost_key)
    cost_beat_ml_2x = _cost_for_acc(front, ml_acc / 2.0, acc_key, cost_key)
    cost_match_num = _cost_for_acc(front, num_acc * 1.5, acc_key, cost_key)
    return {
        "pure_ml_acc": float(ml_acc),
        "pure_numerical_acc": float(num_acc), "pure_numerical_cost": int(num_cost),
        "hybrid_floor_acc": float(floor), "hybrid_floor_cost": int(cost_at_floor),
        "hybrid_acc_at_half_numerical_cost": (float(half) if half is not None else None),
        "cost_to_beat_ml_2x": (int(cost_beat_ml_2x) if cost_beat_ml_2x is not None else None),
        "speedup_to_beat_ml_2x": (float(num_cost / cost_beat_ml_2x)
                                  if cost_beat_ml_2x else None),
        "reaches_numerical_accuracy": bool(floor <= num_acc * 1.5),
        "speedup_at_matched_accuracy": (float(num_cost / cost_match_num)
                                        if cost_match_num else None),
    }
