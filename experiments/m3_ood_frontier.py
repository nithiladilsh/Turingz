from __future__ import annotations
import os, sys, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hybrid_pde.control_214133E import config
from hybrid_pde.control_214133E.groundtruth import load_reference, relative_l2
from hybrid_pde.control_214133E.integrate import (
    load_ml_solver, load_numerical_solver, load_coupling, per_step_costs)
from hybrid_pde.control_214133E.controller import AdaptiveController, thresholds_for_target
from hybrid_pde.control_214133E.runtime import HybridRuntime
from hybrid_pde.control_214133E.trigger import TrustMonitorAdapter
from hybrid_pde.control_214133E.robustness import hit_rate, mean_std
from hybrid_pde.trust.coarse_reference import CoarseReferenceMonitor

NU = 1.0 / (100.0 * np.pi)
L = 2.0


def cole_hopf(ic, x, t, dx):
    x = np.asarray(x, float); t = np.asarray(t, float)
    nx, nt = len(x), len(t)
    x_ext = np.concatenate([x - L, x, x + L])
    diff = x[:, None] - x_ext
    ics = np.asarray(ic, float)[None, :]
    cumint = np.concatenate(
        [np.zeros((1, 1)), np.cumsum(0.5 * (ics[:, :-1] + ics[:, 1:]) * dx, axis=1)], axis=1)
    a = -cumint / (2 * NU)
    pe = np.tile(np.exp(a - a.max(axis=1, keepdims=True)), (1, 3))
    u = np.empty((1, nt, nx)); u[:, 0] = ics
    for j in range(1, nt):
        kj = np.exp(-diff ** 2 / (4 * NU * t[j]))
        u[:, j] = (pe @ (diff * kj).T) / (pe @ kj.T) / t[j]
    return u[0]


def norm(u):
    u = u - u.mean()
    return u / (np.abs(u).max() + 1e-12)


def build_ood(x):
    waves = {}
    for m in (5, 6, 7, 8):
        waves["sin_%dpi" % m] = norm(np.sin(m * np.pi * x))
    for c, w in [(-0.3, 0.15), (0.3, 0.20), (0.0, 0.10)]:
        waves["gauss_c%+.1f_w%.2f" % (c, w)] = norm(np.exp(-((x - c) / w) ** 2))
    return waves


def frontier_on(problems, ml, num, coupling, x, t, targets, ml_c, num_c):
    rows = []
    for tg in targets:
        errs, corrs, mls = [], [], []
        for ic, ref in problems:
            lo, _ = thresholds_for_target(tg)
            trust = TrustMonitorAdapter(CoarseReferenceMonitor(ic, x, n=256))
            res = HybridRuntime(ml, num, trust, coupling,
                                AdaptiveController(lo, 1.1)).run(ic, x, t, tg, reference=ref)
            errs.append(res.cost.achieved_error)
            corrs.append(res.cost.correction_steps)
            mls.append(res.cost.ml_steps)
        me, se = mean_std(errs)
        cost = float(np.mean([m * ml_c + c * num_c for m, c in zip(mls, corrs)]))
        rows.append({"target": float(tg), "cost_s": cost, "mean_error": me, "std_error": se,
                     "hit_rate": hit_rate(errs, tg), "mean_corrections": float(np.mean(corrs))})
    return rows


def baselines(problems, ml, num, x, t, ml_c, num_c):
    nt = len(t)
    ml_e = [relative_l2(ml.rollout(ic, x, t), ref) for ic, ref in problems]
    num_e = [relative_l2(num.rollout(ic, x, t), ref) for ic, ref in problems]
    return {"pure_ml": {"cost_s": nt * ml_c, "mean_error": mean_std(ml_e)[0], "std_error": mean_std(ml_e)[1]},
            "pure_numerical": {"cost_s": nt * num_c, "mean_error": mean_std(num_e)[0], "std_error": mean_std(num_e)[1]}}


def _print_block(title, base, front):
    print("\n" + title)
    print("  pure ML       %7.3f s   %6.2f%%" % (base["pure_ml"]["cost_s"], 100 * base["pure_ml"]["mean_error"]))
    print("  pure numeric  %7.3f s   %6.2f%%" % (base["pure_numerical"]["cost_s"], 100 * base["pure_numerical"]["mean_error"]))
    print("  %-8s %8s %8s %8s %8s" % ("target", "error", "hit", "corr", "cost_s"))
    for r in front:
        print("  %-8.3g %7.2f%% %7.0f%% %8.1f %8.3f" % (
            r["target"], 100 * r["mean_error"], 100 * r["hit_rate"], r["mean_corrections"], r["cost_s"]))


def main():
    R = load_reference()
    x, t = R.x, R.t
    dx = L / len(x)

    chk = relative_l2(cole_hopf(R.ICs[900], x, t, dx), R.u[900])
    print("self-check 1  analytic reference vs committed dataset (IC 900): rel diff = %.2e  %s"
          % (chk, "OK" if chk < 1e-3 else "FAIL"))
    assert chk < 1e-3, "analytic Cole-Hopf reference does not match the committed dataset"

    ml = load_ml_solver()
    num = load_numerical_solver()
    coupling = load_coupling()
    ml_c, num_c = per_step_costs(ml, num, R.ICs[config.TEST_IC_INDICES[0]], x, t)
    targets = config.DEFAULT_ACCURACY_TARGETS

    idx = config.TEST_IC_INDICES
    indist = [(R.ICs[i], R.u[i]) for i in idx]
    indist_front = frontier_on(indist, ml, num, coupling, x, t, targets, ml_c, num_c)
    indist_base = baselines(indist, ml, num, x, t, ml_c, num_c)

    waves = build_ood(x)
    ood = [(ic, cole_hopf(ic, x, t, dx)) for ic in waves.values()]
    ood_front = frontier_on(ood, ml, num, coupling, x, t, targets, ml_c, num_c)
    ood_base = baselines(ood, ml, num, x, t, ml_c, num_c)

    out = {"grid": {"nx": int(len(x)), "nt": int(len(t))},
           "ml_step_s": ml_c, "num_step_s": num_c,
           "indist": {"names": ["test_%d" % i for i in idx], "baselines": indist_base, "frontier": indist_front},
           "ood": {"names": list(waves.keys()), "baselines": ood_base, "frontier": ood_front}}

    out_dir = os.path.join(config.RESULTS_DIR, "m3", "ood_frontier")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "m3_ood_frontier.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1, default=float)

    _print_block("IN-DISTRIBUTION (self-check 2: should reproduce the committed frontier)", indist_base, indist_front)
    _print_block("OUT-OF-DISTRIBUTION (modes 5-8 + Gaussian bumps)", ood_base, ood_front)
    print("\nsaved", path)
    return out


if __name__ == "__main__":
    main()
