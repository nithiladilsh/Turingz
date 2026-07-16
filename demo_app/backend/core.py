import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from hybrid_pde.trust.monitor import TrustMonitor, load_params
from hybrid_pde.trust.coarse_reference import CoarseReferenceMonitor
from hybrid_pde.trust.signals import corrected_signal, energy_signal, roughness_signal, shock_coeff

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


def stream_run(model, ic=None, pinn_index=0, mode="reference_free"):
    ic0, pred, true = get_prediction(model, ic, pinn_index)
    res_c, ene_c, rou_c = _signal_curves(pred)
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
            "switch_t": switch_t,
            "mode": mode,
        }


def meta():
    return {"x": np.round(X, 4).tolist(), "t": np.round(T, 4).tolist(),
            "n_pinn_ics": int(len(PINN_ICS)), "fail_threshold": FAIL,
            "params": {m: {"CUT": round(float(PARAMS[m]["CUT"]), 2), "K": int(PARAMS[m]["K"])} for m in PARAMS}}
