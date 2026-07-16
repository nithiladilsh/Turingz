import json
import asyncio
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import core
from hybrid_pde.coupling_214050V.m2_coupling import M2Coupling

router = APIRouter()


def cole_hopf_from(u0, taus):
    u0 = np.asarray(u0, float)[None, :]
    taus = np.asarray(taus, float)
    cumint = np.concatenate(
        [np.zeros((1, 1)),
         np.cumsum(0.5 * (u0[:, :-1] + u0[:, 1:]) * core.DX, axis=1)], axis=1)
    a = -cumint / (2 * core.NU)
    pe = np.tile(np.exp(a - a.max(axis=1, keepdims=True)), (1, 3))
    out = np.empty((len(taus), core.NX))
    out[0] = u0[0]
    for j in range(1, len(taus)):
        tj = float(taus[j])
        if tj <= 0:
            out[j] = u0[0]
            continue
        K = np.exp(-core._diff ** 2 / (4 * core.NU * tj))
        out[j] = ((pe @ (core._diff * K).T) / (pe @ K.T) / tj)[0]
    return out


class _MLStub:
    def __init__(self, pred):
        self.pred = pred

    def rollout(self, ic, x, t):
        return self.pred


class _NumSolver:
    def rollout(self, u0, x, tau):
        return cole_hopf_from(u0, tau)


def _relerr(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def _build(model, ic, pinn_index, mode):
    ic0, pred, true = core.get_prediction(model, ic, pinn_index)
    T, X, nt = core.T, core.X, len(core.T)
    if model == "FNO" and mode == "coarse":
        mon = core.CoarseReferenceMonitor(ic0, X, n=256)
    else:
        mon = core.TrustMonitor(core.PARAMS[model])
    trust = np.empty(nt)
    okc = np.empty(nt, bool)
    switch = None
    for n in range(nt):
        o = mon.update(pred[n], float(T[n]))
        trust[n] = float(o["trust"])
        okc[n] = bool(o["ok"])
        if switch is None and not o["ok"]:
            switch = n
    idx = {"i": 0}

    def trigger(u, t):
        i = idx["i"]
        idx["i"] = i + 1
        return float(trust[i]), (not bool(okc[i]))

    hybrid = np.asarray(M2Coupling().rollout(ic0, X, T, _MLStub(pred), _NumSolver(), trigger), float)
    return T, pred, true, hybrid, trust, okc, switch, nt


async def _run(ws, req):
    model = req.get("model", "FNO")
    ic = req.get("ic")
    pinn_index = int(req.get("pinn_index", 0))
    mode = req.get("mode", "reference_free")
    loop = asyncio.get_running_loop()
    T, pred, true, hybrid, trust, okc, switch, nt = await loop.run_in_executor(
        None, _build, model, ic, pinn_index, mode)
    ml_per, num_per = 0.21 / nt, 2.54 / nt
    s = switch if switch is not None else nt
    for n in range(nt):
        await ws.send_text(json.dumps({
            "t": float(T[n]),
            "u_ml": np.round(pred[n], 4).tolist(),
            "u_hybrid": np.round(hybrid[n], 4).tolist(),
            "true": np.round(true[n], 4).tolist(),
            "trust": round(float(trust[n]), 3),
            "ok": bool(okc[n]),
            "switch_t": (float(T[switch]) if switch is not None else None),
            "ml_steps": int(min(n + 1, s)),
            "corr_steps": int(max(0, n + 1 - s)),
            "hybrid_error": round(_relerr(hybrid[n], true[n]), 3),
            "ml_error": round(_relerr(pred[n], true[n]), 3),
        }))
        await asyncio.sleep(0.03)
    ml_steps, corr_steps = s, nt - s
    await ws.send_text(json.dumps({"done": True, "summary": {
        "ml_steps": int(ml_steps),
        "corr_steps": int(corr_steps),
        "cost_hybrid": round(ml_steps * ml_per + corr_steps * num_per, 2),
        "cost_ml": 0.21,
        "cost_num": 2.54,
        "err_hybrid": round(_relerr(hybrid[-1], true[-1]), 3),
        "err_ml": round(_relerr(pred[-1], true[-1]), 3),
        "err_num": 0.0,
        "switch_t": (float(T[switch]) if switch is not None else None),
    }}))


@router.websocket("/ws/pipeline")
async def ws_pipeline(ws: WebSocket):
    await ws.accept()
    try:
        req = json.loads(await ws.receive_text())
        await _run(ws, req)
    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_text(json.dumps({"error": str(e)}))
