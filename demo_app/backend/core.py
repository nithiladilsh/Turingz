import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from hybrid_pde.trust.monitor import TrustMonitor, load_params, _frame as _mon_frame, _hi_mask as _mon_hi
from hybrid_pde.trust.coarse_reference import CoarseReferenceMonitor
from hybrid_pde.trust.signals import corrected_signal, energy_signal, roughness_signal, shock_coeff

# order of the fused signals inside the trust monitor
SIGNAL_KEYS = ["residual", "energy", "roughness", "momentum"]

NU = 1.0 / (100 * np.pi)
L = 2.0
FAIL = 0.10

_pred = np.load(os.path.join(ROOT, "results", "eval", "predictions.npz"))
X = _pred["x"].astype(float)
T = _pred["t"].astype(float)
NX = len(X)
DX = X[1] - X[0]
_x_ext = np.concatenate([X - L, X, X + L])
_diff = X[:, None] - _x_ext

PINN_ICS = _pred["u_true_eval"][:, 0, :]
PINN_TRUE = _pred["u_true_eval"]
PINN_PRED = _pred["PINN"]

PARAMS = {m: load_params(os.path.join(ROOT, "results", "trust", "trust_params_%s.npz" % m))
          for m in ["PINN", "FNO", "DeepONet"]}
COEFF = shock_coeff(_pred["u_true_seen"], X, T)


def build_ic(modes=4, amplitude=1.0, phase=0.0, seed=0):
    rng = np.random.default_rng(int(seed))
    u = sum(rng.standard_normal() * np.sin(2 * np.pi * m * X / L + phase + rng.uniform(0, 2 * np.pi))
            for m in range(1, int(modes) + 1))
    u = u / (np.abs(u).max() + 1e-12)
    return (float(amplitude) * u).astype(float)


def cole_hopf(ic):
    ic = np.asarray(ic, float)[None, :]
    cumint = np.concatenate([np.zeros((1, 1)),
                             np.cumsum(0.5 * (ic[:, :-1] + ic[:, 1:]) * DX, axis=1)], axis=1)
    a = -cumint / (2 * NU)
    pe = np.tile(np.exp(a - a.max(axis=1, keepdims=True)), (1, 3))
    out = np.empty((len(T), NX)); out[0] = ic[0]
    for j in range(1, len(T)):
        K = np.exp(-_diff ** 2 / (4 * NU * T[j]))
        out[j] = ((pe @ (_diff * K).T) / (pe @ K.T) / T[j])[0]
    return out


def _predict_operator(model, ic):
    import torch
    torch.set_default_device("cpu")
    RES = os.path.join(ROOT, "results")
    if model == "FNO":
        from neuralop.models import FNO
        cfg = torch.load(os.path.join(RES, "fno", "fno_config.pt"), map_location="cpu", weights_only=False)
        m = FNO(n_modes=(cfg["n_modes"],), hidden_channels=cfg["hidden_channels"], in_channels=3, out_channels=1)
        m.load_state_dict(torch.load(os.path.join(RES, "fno", "fno.pt"), map_location="cpu", weights_only=False))
        m.eval()
        xt = torch.tensor(X, dtype=torch.float32); tt = torch.tensor(T, dtype=torch.float32)
        ict = torch.tensor(ic, dtype=torch.float32)
        inp = torch.stack([ict.unsqueeze(0).repeat(len(T), 1),
                           tt.view(-1, 1).repeat(1, NX),
                           xt.view(1, -1).repeat(len(T), 1)], dim=1)
        with torch.no_grad():
            return m(inp).squeeze(1).numpy()
    from hybrid_pde.solvers.ml.deepOnet.deeponet import load_model
    s = load_model(os.path.join(RES, "deeponet"), X)
    return s.predict_grid(np.asarray(ic)[None, :], X, T)[0]


def get_prediction(model, ic=None, pinn_index=0):
    if model == "PINN":
        i = int(pinn_index)
        return PINN_ICS[i].copy(), PINN_PRED[i], PINN_TRUE[i]
    ic = np.asarray(ic, float)
    return ic, _predict_operator(model, ic), cole_hopf(ic)


def _signal_curves(pred):
    p = pred[None]
    return (corrected_signal(p, COEFF, X, T)[0], energy_signal(p, X)[0], roughness_signal(p, X)[0])


def _signal_activity(pred, params):
    """Replays the exact 4 signals the TrustMonitor fuses and returns, per frame,
    each signal's z-scored 'activity' (how far above its calibrated baseline it is,
    mapped to 0..1) plus the fixed fusion weights. This is what actually drives the
    trust score -- not the raw magnitudes, which live on very different scales."""
    p = params
    dx, dt = float(p["dx"]), float(p["dt"])
    hi = _mon_hi(pred.shape[-1], dx)
    mean, std, w = np.asarray(p["mean"], float), np.asarray(p["std"], float), np.asarray(p["w"], float)
    ecum, eprev = 0.0, 0.5 * dx * (pred[0] ** 2).sum()
    M0, mdmax = None, 0.0
    Z = np.zeros((len(pred), 4))
    for n in range(len(pred)):
        prev = pred[n - 1] if n >= 1 else pred[n]
        s, ecum, eprev = _mon_frame(pred[n], prev, ecum, eprev, dx, dt, float(p["coeff"]), hi)
        if M0 is None:
            M0 = s[3]
        mdmax = max(mdmax, abs(s[3] - M0))
        s = s.copy(); s[3] = mdmax
        Z[n] = (s - mean) / std
    # map z to a 0..1 "activity" level: baseline (z=0) -> 0.5, ~+3 sigma -> ~1
    level = np.clip(0.5 + Z / 6.0, 0.0, 1.0)
    return w, Z, level


def stream_run(model, ic=None, pinn_index=0, mode="reference_free"):
    ic0, pred, true = get_prediction(model, ic, pinn_index)
    res_c, ene_c, rou_c = _signal_curves(pred)
    sig_w, sig_Z, sig_lvl = _signal_activity(pred, PARAMS[model])
    weights = {k: round(float(sig_w[i]), 3) for i, k in enumerate(SIGNAL_KEYS)}
    if model == "FNO" and mode == "coarse":
        mon = CoarseReferenceMonitor(ic0, X, n=256)
    else:
        mon = TrustMonitor(PARAMS[model])
    switch_t = None
    for n in range(len(T)):
        out = mon.update(pred[n], float(T[n]))
        err = float(np.linalg.norm(pred[n] - true[n]) / (np.linalg.norm(true[n]) + 1e-12))
        if switch_t is None and not out["ok"]:
            switch_t = float(T[n])
        yield {
            "t": float(T[n]),
            "u": np.round(pred[n], 4).tolist(),
            "true": np.round(true[n], 4).tolist(),
            "trust": round(float(out["trust"]), 3),
            "ok": bool(out["ok"]),
            "true_error": round(err, 3),
            "signals": {"residual": round(float(res_c[n]), 4),
                        "energy": round(float(ene_c[n]), 4),
                        "roughness": round(float(rou_c[n]), 4)},
            "weights": weights,
            "levels": {k: round(float(sig_lvl[n, i]), 3) for i, k in enumerate(SIGNAL_KEYS)},
            "contrib": {k: round(float(sig_w[i] * sig_Z[n, i]), 4) for i, k in enumerate(SIGNAL_KEYS)},
            "switch_t": switch_t,
            "mode": mode,
        }


def meta():
    return {"x": np.round(X, 4).tolist(), "t": np.round(T, 4).tolist(),
            "n_pinn_ics": int(len(PINN_ICS)), "fail_threshold": FAIL,
            "params": {m: {"CUT": round(float(PARAMS[m]["CUT"]), 2), "K": int(PARAMS[m]["K"])} for m in PARAMS}}


def reliability_stream(model, ic=None, pinn_index=0):
    ic0, pred, true = get_prediction(model, ic, pinn_index)
    e = np.linalg.norm(pred - true, axis=1) / (np.linalg.norm(true, axis=1) + 1e-12)
    horizon = float(T[np.argmax(e > FAIL)]) if (e > FAIL).any() else float(T[-1])
    inw = float(e[T <= 1.0].mean())
    ext = float(e[T > 1.0].mean())
    for k in range(len(T)):
        yield {
            "t": float(T[k]),
            "u": np.round(pred[k], 4).tolist(),
            "true": np.round(true[k], 4).tolist(),
            "error": round(float(e[k]), 4),
            "horizon": round(horizon, 2),
            "in_window": round(inw, 3),
            "extrap": round(ext, 3),
        }


# ================= Module 2: Coupling (Dharmapala R.D. 214050V) =================
# The demo calls the REAL M2Coupling adapter -- the same object Module 3's
# runtime uses -- with the verified pseudo-spectral restart stepper loaded
# straight from restart_spectral.py (proven bit-for-bit equal to the
# production solver). No demo-only reimplementation of the mechanism.
import json as _json
import importlib.util as _ilu

_M2_DIR = os.path.join(ROOT, "hybrid_pde", "coupling - 214050V")
_spec = _ilu.spec_from_file_location(
    "m2_restart_spectral", os.path.join(_M2_DIR, "restart_spectral.py"))
_rs = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_rs)

from hybrid_pde.coupling_214050V.m2_coupling import M2Coupling


class _SpectralNum:
    """Numerical-solver adapter for M2Coupling.rollout: integrates the verified
    pseudo-spectral scheme (restart_spectral's exact stepper: same K, 2/3 mask,
    IF-RK4, dt=1e-4, Nyquist zeroing) over a local time grid tau, tau[0]=0."""
    name = "spectral-restart"

    def rollout(self, ic, x, t):
        t = np.asarray(t, float)
        u0 = np.asarray(ic, float)
        out = np.empty((len(t), _rs.NX))
        out[0] = u0
        uh = np.fft.rfft(u0)
        for j in range(1, len(t)):
            h = t[j] - t[j - 1]
            m = max(1, round(h / _rs.DT_TARGET))
            h = h / m
            E = np.exp(-NU * _rs.K ** 2 * h)
            E2 = np.exp(-NU * _rs.K ** 2 * h * 0.5)
            for _ in range(m):
                uh = _rs._step(uh, E, E2, h)
            out[j] = np.fft.irfft(uh * _rs.MASK, n=_rs.NX)
        return out


class _MLArray:
    """Solver-protocol wrapper around a precomputed ML prediction array."""
    name = "ml-prediction"

    def __init__(self, pred):
        self.pred = pred

    def rollout(self, ic, x, t):
        return self.pred


def _tail(a, b, i0):
    num = np.sqrt(((a[i0:] - b[i0:]) ** 2).sum(-1))
    den = np.sqrt((b[i0:] ** 2).sum(-1)) + 1e-12
    c = num / den
    return float(np.trapezoid(c, T[i0:]) / (T[-1] - T[i0] + 1e-12))


def stream_coupling(model, ic=None, pinn_index=0, switch_mode="manual", t_s=1.0):
    """Precompute the full hybrid rollout with the real M2Coupling, then stream."""
    ic0, pred, true = get_prediction(model, ic, pinn_index)

    trust_curve = None
    if switch_mode == "trust":
        mon = TrustMonitor(PARAMS[model])

        def trigger(u, t):
            out = mon.update(u, float(t))
            return float(out["trust"]), (not out["ok"])

        disp = TrustMonitor(PARAMS[model])          # same params, for display
        trust_curve = [float(disp.update(pred[n], float(T[n]))["trust"])
                       for n in range(len(T))]
    else:
        def trigger(u, t):
            return 1.0, (float(t) >= float(t_s) - 1e-9)

    hybrid = M2Coupling().rollout(ic0, X, T, _MLArray(pred), _SpectralNum(), trigger)

    diff = np.sqrt(((hybrid - pred) ** 2).sum(-1))
    idx = np.nonzero(diff > 1e-12)[0]
    switch_i = int(idx[0] - 1) if len(idx) else None   # out[switch] == ML state
    switch_t = float(T[switch_i]) if switch_i is not None else None

    i1 = int(np.argmin(np.abs(T - 1.0)))
    summary = {
        "switch_t": switch_t,
        "numerical_fraction": (round(1.0 - (switch_i / (len(T) - 1)), 3)
                               if switch_i is not None else 0.0),
        "ml_tail_1_2": round(_tail(pred, true, i1), 4),
        "hybrid_tail_1_2": round(_tail(hybrid, true, i1), 4),
        "handoff_jump": (float(np.linalg.norm(hybrid[switch_i] - pred[switch_i]))
                         if switch_i is not None else 0.0),
    }
    if summary["ml_tail_1_2"] > 0:
        summary["benefit"] = round(1.0 - summary["hybrid_tail_1_2"] / summary["ml_tail_1_2"], 3)

    for n in range(len(T)):
        ml_e = float(np.linalg.norm(pred[n] - true[n]) / (np.linalg.norm(true[n]) + 1e-12))
        hy_e = float(np.linalg.norm(hybrid[n] - true[n]) / (np.linalg.norm(true[n]) + 1e-12))
        yield {
            "t": float(T[n]),
            "ml": np.round(pred[n], 4).tolist(),
            "true": np.round(true[n], 4).tolist(),
            "hybrid": np.round(hybrid[n], 4).tolist(),
            "ml_err": round(ml_e, 4),
            "hybrid_err": round(hy_e, 4),
            "trust": (round(trust_curve[n], 3) if trust_curve else None),
            "switch_t": switch_t,
            "switched": bool(switch_i is not None and n >= switch_i),
        }
    yield {"summary": summary}


def coupling_meta():
    fig = os.path.join(ROOT, "results", "module2", "figures")
    with open(os.path.join(fig, "handoff_sweep_results.json")) as f:
        sweep = _json.load(f)
    out = {
        "sweep": [{"t_s": r["t_s"], "benefit": r["benefit"],
                   "hybrid_tail": r["hybrid_tail"], "fno_tail": r["fno_tail"],
                   "numerical_fraction": r["numerical_fraction"]}
                  for r in sweep["results"]],
        "viability": sweep["viability_rule"],
        "headline": {"error_cut": "13.4% -> 1.0%", "benefit": "92%",
                     "oracle_floor": "~1e-6", "restart_verified": "rel diff 0.0"},
    }
    bpath = os.path.join(fig, "restart_safety_boundary.json")
    if os.path.exists(bpath):
        with open(bpath) as f:
            b = _json.load(f)
        rows = sorted(b["boundary"].values(), key=lambda r: r["re_cell"])
        out["boundary"] = {
            "re_cell": [r["re_cell"] for r in rows],
            "verified_tail": [r["verified_tail"] for r in rows],
            "careless_tail": [r["careless_tail"] for r in rows],
            "careless_unstable": [r["careless_unstable"] for r in rows],
            "crossings": b.get("boundary_crossings_re_cell", {}),
        }
    return out


# ============ Cole-Hopf page (Dharmapala R.D. 214050V -- ground truth) ============
def stream_colehopf(ic):
    """Exact Cole-Hopf solution + live cross-verification against the
    independent pseudo-spectral solver (the project's trust argument for
    the reference: two independent methods agree)."""
    ic = np.asarray(ic, float)
    ch = cole_hopf(ic)
    sp = _SpectralNum().rollout(ic, X, T)
    dis = np.sqrt(((ch - sp) ** 2).sum(-1)) / (np.sqrt((sp ** 2).sum(-1)) + 1e-12)
    summary = {"mean_disagreement": float(dis[1:].mean()),
               "max_disagreement": float(dis[1:].max())}
    for n in range(len(T)):
        yield {"t": float(T[n]),
               "ch": np.round(ch[n], 4).tolist(),
               "sp": np.round(sp[n], 4).tolist(),
               "disagreement": float(dis[n])}
    yield {"summary": summary}


# ========= Robustness page (Dharmapala R.D. 214050V -- OOD + spectral signal) =========
def spectral_distance(a, b, alpha=1.0):
    """Shape (Fourier-amplitude) distance -- same definition as the Module 2
    evaluation code (make_figures.py)."""
    A = np.abs(np.fft.rfft(a, axis=-1))
    B = np.abs(np.fft.rfft(b, axis=-1))
    k = np.arange(A.shape[-1])
    w = (1.0 + k) ** alpha
    return np.sqrt((w * (A - B) ** 2).sum(-1)) / (np.sqrt((w * B ** 2).sum(-1)) + 1e-12)


def robustness_ic(preset):
    """OOD presets -- identical definitions to ood_experiment.py."""
    norm = lambda u: u / (np.abs(u).max() + 1e-12)
    if preset == "in_dist":
        return norm(np.sin(np.pi * X))
    if preset == "high_freq":
        return norm(np.sin(6 * np.pi * X))          # beyond trained band (modes 1-4)
    if preset == "gaussian":
        return norm(np.exp(-(X ** 2) / (2 * 0.10 ** 2)))  # localized bump
    raise ValueError("unknown preset " + str(preset))


def stream_robustness(model, preset=None, ic=None, pinn_index=0):
    ic0 = robustness_ic(preset) if preset else ic
    ic0, pred, true = get_prediction(model, ic=ic0, pinn_index=pinn_index)
    i1 = int(np.argmin(np.abs(T - 1.0)))
    err = np.sqrt(((pred - true) ** 2).sum(-1)) / (np.sqrt((true ** 2).sum(-1)) + 1e-12)
    sd = np.array([float(spectral_distance(pred[n], true[n])) for n in range(len(T))])
    cross = np.nonzero(err > FAIL)[0]
    summary = {
        "in_window_err": round(float(np.trapezoid(err[:i1 + 1], T[:i1 + 1]) / (T[i1] - T[0] + 1e-12)), 4),
        "extrap_err": round(float(np.trapezoid(err[i1:], T[i1:]) / (T[-1] - T[i1] + 1e-12)), 4),
        "reliable_horizon": (float(T[cross[0]]) if len(cross) else 2.0),
        "highk_energy_frac_end": float(np.abs(np.fft.rfft(pred[-1]))[len(X) // 6:len(X) // 3 + 1].__pow__(2).sum()
                                       / (np.abs(np.fft.rfft(pred[-1])).__pow__(2).sum() + 1e-12)),
    }
    for n in range(len(T)):
        yield {"t": float(T[n]),
               "u": np.round(pred[n], 4).tolist(),
               "true": np.round(true[n], 4).tolist(),
               "err": round(float(err[n]), 4),
               "sd": round(float(sd[n]), 4)}
    yield {"summary": summary}
