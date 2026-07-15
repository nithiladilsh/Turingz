from __future__ import annotations
import numpy as np
from .runtime import HybridRuntime
from .controller import AdaptiveController, thresholds_for_target
from .robustness import hit_rate, mean_std
from .profiler import Profiler

_NU = 1.0 / (100.0 * np.pi)
_L = 2.0
_NX = 512
_K = 2.0 * np.pi * np.arange(_NX // 2 + 1) / _L
_MASK = np.arange(_NX // 2 + 1) <= _NX // 3
_DT = 1e-4


class FunctionSolver:
    def __init__(self, name, rollout_fn):
        self.name = name
        self._f = rollout_fn

    def rollout(self, ic, x, t):
        return self._f(ic, x, t)


def _rhs(uh):
    u = np.fft.irfft(uh * _MASK, n=_NX)
    return -0.5j * _K * np.fft.rfft(u * u)


def _step(uh, E, E2, h):
    k1 = _rhs(uh)
    k2 = _rhs(E2 * uh + 0.5 * h * E2 * k1)
    k3 = _rhs(E2 * uh + 0.5 * h * k2)
    k4 = _rhs(E * uh + h * E2 * k3)
    uh = E * uh + h / 6 * (E * k1 + 2 * E2 * k2 + 2 * E2 * k3 + k4)
    uh[-1] = 0.0
    return uh


def spectral_rollout(ic, x, t):
    ic = np.asarray(ic, dtype=float)
    t = np.asarray(t, dtype=float)
    uh = np.fft.rfft(ic)
    out = np.empty((len(t), _NX))
    out[0] = ic
    for j in range(1, len(t)):
        h = t[j] - t[j - 1]
        m = max(1, round(h / _DT))
        h /= m
        E, E2 = np.exp(-_NU * _K ** 2 * h), np.exp(-_NU * _K ** 2 * h * 0.5)
        for _ in range(m):
            uh = _step(uh, E, E2, h)
        out[j] = np.fft.irfft(uh * _MASK, n=_NX)
    return out


def fno_rollout(model, ic, x, t):
    import torch
    ic_t = torch.as_tensor(np.asarray(ic, dtype=np.float32))
    x_t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    t_t = torch.as_tensor(np.asarray(t, dtype=np.float32))
    nx = x_t.numel()
    K = t_t.numel()
    inp = torch.stack([ic_t.view(1, -1).repeat(K, 1),
                       t_t.view(-1, 1).repeat(1, nx),
                       x_t.view(1, -1).repeat(K, 1)], dim=1)
    with torch.no_grad():
        return model(inp).squeeze(1).cpu().numpy()


def load_ml_solver():
    import torch
    from neuralop.models import FNO
    from . import config
    cfg = torch.load(config.RESULTS_DIR / "fno" / "fno_config.pt", weights_only=False, map_location="cpu")
    model = FNO(n_modes=(cfg["n_modes"],), hidden_channels=cfg["hidden_channels"], in_channels=3, out_channels=1,
                lifting_channel_ratio=2, projection_channel_ratio=2)
    model.load_state_dict(torch.load(config.RESULTS_DIR / "fno" / "fno.pt", map_location="cpu", weights_only=False))
    model.eval()
    return FunctionSolver("FNO", lambda ic, x, t: fno_rollout(model, ic, x, t))


def load_numerical_solver():
    return FunctionSolver("spectral", spectral_rollout)


def load_trust(model_name="FNO"):
    from . import config
    from hybrid_pde.trust.monitor import load_params, TrustMonitor
    from .trigger import TrustMonitorAdapter
    params = load_params(str(config.RESULTS_DIR / "trust" / ("trust_params_%s.npz" % model_name)))
    return TrustMonitorAdapter(TrustMonitor(params))


def load_coupling():
    from hybrid_pde.coupling_214050V.m2_coupling import M2Coupling
    from .coupling import RealCoupling
    return RealCoupling(M2Coupling())


def per_step_costs(ml, num, ic, x, t, repeats=20, warmup=3):
    p = Profiler(repeats=repeats, warmup=warmup)
    return p.profile(ml, ic, x, t).per_step_s, p.profile(num, ic, x, t).per_step_s


def run_frontier(ml, num, trust, coupling, problems, x, t, targets, ml_step_s, num_step_s):
    rows = []
    for tg in targets:
        errs = []
        cost = None
        for ic, ref in problems:
            lo, hi = thresholds_for_target(tg)
            res = HybridRuntime(ml, num, trust, coupling, AdaptiveController(lo, hi)).run(ic, x, t, tg, reference=ref)
            errs.append(res.cost.achieved_error)
            cost = res.cost.ml_steps * ml_step_s + res.cost.correction_steps * num_step_s
        m, sd = mean_std(errs)
        rows.append({"target": float(tg), "cost": float(cost), "mean_error": m, "std_error": sd, "hit_rate": hit_rate(errs, tg)})
    return rows


def main(targets=None):
    from . import config
    from .groundtruth import load_reference
    ml = load_ml_solver()
    num = load_numerical_solver()
    trust = load_trust()
    coupling = load_coupling()
    R = load_reference()
    idx = config.TEST_IC_INDICES
    problems = [(R.ICs[i], R.u[i]) for i in idx]
    ml_c, num_c = per_step_costs(ml, num, R.ICs[idx[0]], R.x, R.t)
    tg = targets if targets is not None else config.DEFAULT_ACCURACY_TARGETS
    return run_frontier(ml, num, trust, coupling, problems, R.x, R.t, tg, ml_c, num_c)


def partial_main(targets=None, real_trust=False):
    from . import config
    from .groundtruth import load_reference
    from .trigger import SyntheticTrust
    from .coupling import CouplingStub
    ml = load_ml_solver()
    num = load_numerical_solver()
    R = load_reference()
    idx = config.TEST_IC_INDICES
    problems = [(R.ICs[i], R.u[i]) for i in idx]
    trust = load_trust() if real_trust else SyntheticTrust(R.t_train_end, width=0.08, flag_at=0.0)
    ml_c, num_c = per_step_costs(ml, num, R.ICs[idx[0]], R.x, R.t)
    tg = targets if targets is not None else config.DEFAULT_ACCURACY_TARGETS
    return run_frontier(ml, num, trust, CouplingStub(), problems, R.x, R.t, tg, ml_c, num_c)


def dry_run():
    from ._smoke import MLDrift, NumExact, HORIZON
    from .trigger import SyntheticTrust
    from .coupling import CouplingStub
    x = np.linspace(-1, 1, 512)
    t = np.linspace(0, 2, 200)
    problems = [(np.sin(k * np.pi * x), np.sin(k * np.pi * x)[None, :] * np.exp(-t)[:, None]) for k in (1, 2, 3)]
    ml, num = MLDrift(), NumExact()
    ml_c, num_c = per_step_costs(ml, num, problems[0][0], x, t)
    trust = SyntheticTrust(HORIZON, width=0.08, flag_at=0.0)
    return run_frontier(ml, num, trust, CouplingStub(), problems, x, t, [0.30, 0.10, 0.05, 0.02], ml_c, num_c)


def save_report(rows, out_dir, label="frontier"):
    import os, json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    json.dump(rows, open(os.path.join(out_dir, label + ".json"), "w"), indent=2, default=float)
    co = [r["cost"] for r in rows]
    me = [r["mean_error"] for r in rows]
    sd = [r["std_error"] for r in rows]
    tg = [r["target"] for r in rows]
    plt.figure(figsize=(7, 4.3))
    plt.errorbar(co, me, yerr=sd, fmt="o-", color="#1E8449", lw=2, capsize=4)
    for a, c, e in zip(tg, co, me):
        plt.annotate(f"{a}", (c, e), textcoords="offset points", xytext=(6, 5), fontsize=7)
    plt.xlabel("cost  (lower is better)")
    plt.ylabel("mean relative L2 error  (lower is better)")
    plt.title("Module 3 frontier - " + label)
    plt.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, label + ".png"), dpi=140)
    plt.close()
    return os.path.join(out_dir, label + ".png")


def partial_baselines():
    from . import config
    from .groundtruth import load_reference, relative_l2
    from .robustness import mean_std
    ml = load_ml_solver()
    num = load_numerical_solver()
    R = load_reference()
    idx = config.TEST_IC_INDICES
    x, t = R.x, R.t
    ml_c, num_c = per_step_costs(ml, num, R.ICs[idx[0]], x, t)
    nt = len(t)
    ml_errs = [relative_l2(ml.rollout(R.ICs[i], x, t), R.u[i]) for i in idx]
    num_errs = [relative_l2(num.rollout(R.ICs[i], x, t), R.u[i]) for i in idx]
    return {"pure_ml": {"cost": nt * ml_c, "mean_error": mean_std(ml_errs)[0], "std_error": mean_std(ml_errs)[1]},
            "pure_numerical": {"cost": nt * num_c, "mean_error": mean_std(num_errs)[0], "std_error": mean_std(num_errs)[1]}}


def spectral_rollout_coarse(ic, x, t, dt=2e-2):
    ic = np.asarray(ic, dtype=float)
    t = np.asarray(t, dtype=float)
    uh = np.fft.rfft(ic)
    out = np.empty((len(t), _NX))
    out[0] = ic
    for j in range(1, len(t)):
        h = t[j] - t[j - 1]
        m = max(1, round(h / dt))
        h /= m
        E, E2 = np.exp(-_NU * _K ** 2 * h), np.exp(-_NU * _K ** 2 * h * 0.5)
        for _ in range(m):
            uh = _step(uh, E, E2, h)
        out[j] = np.fft.irfft(uh * _MASK, n=_NX)
    return out


def partial_main_coarse(targets=None):
    from . import config
    from .groundtruth import load_reference
    from .coupling import CouplingStub
    from .trigger import TrustMonitorAdapter
    from hybrid_pde.trust.coarse_reference import CoarseReferenceMonitor
    ml = load_ml_solver()
    num = load_numerical_solver()
    R = load_reference()
    idx = config.TEST_IC_INDICES
    x, t = R.x, R.t
    ml_c, num_c = per_step_costs(ml, num, R.ICs[idx[0]], x, t)
    tg = targets if targets is not None else config.DEFAULT_ACCURACY_TARGETS
    rows = []
    for target in tg:
        lo, _ = thresholds_for_target(target)
        errs = []
        cost = None
        for i in idx:
            ic, ref = R.ICs[i], R.u[i]
            trust = TrustMonitorAdapter(CoarseReferenceMonitor(ic, x, n=256))
            res = HybridRuntime(ml, num, trust, CouplingStub(), AdaptiveController(lo, 1.1)).run(ic, x, t, target, reference=ref)
            errs.append(res.cost.achieved_error)
            cost = res.cost.ml_steps * ml_c + res.cost.correction_steps * num_c
        m, sd = mean_std(errs)
        rows.append({"target": float(target), "cost": float(cost), "mean_error": m, "std_error": sd, "hit_rate": hit_rate(errs, target)})
    return rows


def partial_main_coarse_timed(targets=None):
    import time
    from . import config
    from .groundtruth import load_reference, relative_l2
    from .coupling import CouplingStub
    from .trigger import TrustMonitorAdapter
    from hybrid_pde.trust.coarse_reference import CoarseReferenceMonitor
    ml = load_ml_solver()
    num = load_numerical_solver()
    R = load_reference()
    idx = config.TEST_IC_INDICES
    x, t = R.x, R.t
    ml.rollout(R.ICs[idx[0]], x, t)   # warm up
    num.rollout(R.ICs[idx[0]], x, t)
    ml_times, num_times, ml_errs, num_errs = [], [], [], []
    for i in idx:
        ic, ref = R.ICs[i], R.u[i]
        a = time.perf_counter(); um = ml.rollout(ic, x, t); ml_times.append(time.perf_counter() - a); ml_errs.append(relative_l2(um, ref))
        a = time.perf_counter(); un = num.rollout(ic, x, t); num_times.append(time.perf_counter() - a); num_errs.append(relative_l2(un, ref))
    tg = targets if targets is not None else config.DEFAULT_ACCURACY_TARGETS
    rows = []
    for target in tg:
        lo, _ = thresholds_for_target(target)
        errs, costs = [], []
        for i in idx:
            ic, ref = R.ICs[i], R.u[i]
            trust = TrustMonitorAdapter(CoarseReferenceMonitor(ic, x, n=256))
            rt = HybridRuntime(ml, num, trust, CouplingStub(), AdaptiveController(lo, 1.1))
            a = time.perf_counter(); res = rt.run(ic, x, t, target, reference=ref); wall = time.perf_counter() - a
            errs.append(res.cost.achieved_error); costs.append(wall)
        me, se = mean_std(errs); cm, cs = mean_std(costs)
        rows.append({"target": float(target), "cost_s": cm, "cost_std": cs, "mean_error": me, "std_error": se, "hit_rate": hit_rate(errs, target)})
    return {"frontier": rows,
            "pure_ml": {"cost_s": mean_std(ml_times)[0], "mean_error": mean_std(ml_errs)[0]},
            "pure_numerical": {"cost_s": mean_std(num_times)[0], "mean_error": mean_std(num_errs)[0]}}
