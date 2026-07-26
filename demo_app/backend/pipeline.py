import os
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


def _build(model, ic, pinn_index, mode, target=0.05):
    from hybrid_pde.control_214133E.controller import AdaptiveController, thresholds_for_target
    ic0, pred, true = core.get_prediction(model, ic, pinn_index)
    T, X, nt = core.T, core.X, len(core.T)
    if model == "FNO" and mode == "coarse":
        mon = core.CoarseReferenceMonitor(ic0, X, n=256)
    else:
        mon = core.TrustMonitor(core.PARAMS[model])
    lo, _ = thresholds_for_target(float(target))
    ctrl = AdaptiveController(lo, 1.1)
    ctrl.configure(float(target))
    ctrl.reset()
    trust = np.empty(nt)
    corr = np.empty(nt, bool)
    switch = None
    for n in range(nt):
        o = mon.update(pred[n], float(T[n]))
        trust[n] = float(o["trust"])
        dec = ctrl.decide(trust[n], not bool(o["ok"]), float(T[n]), n)
        corr[n] = bool(dec.correct)
        if switch is None and dec.correct:
            switch = n
    idx = {"i": 0}

    def trigger(u, t):
        i = idx["i"]
        idx["i"] = i + 1
        return float(trust[i]), bool(corr[i])

    hybrid = np.asarray(M2Coupling().rollout(ic0, X, T, _MLStub(pred), _NumSolver(), trigger), float)
    return T, pred, true, hybrid, trust, corr, switch, nt


async def _run(ws, req):
    model = req.get("model", "FNO")
    ic = req.get("ic")
    pinn_index = int(req.get("pinn_index", 0))
    mode = req.get("mode", "reference_free")
    target = float(req.get("target", 0.05))
    loop = asyncio.get_running_loop()
    T, pred, true, hybrid, trust, okc, switch, nt = await loop.run_in_executor(
        None, _build, model, ic, pinn_index, mode, target)
    ml_per, num_per = _per_step_rates(nt)
    s = switch if switch is not None else nt
    for n in range(nt):
        await ws.send_text(json.dumps({
            "t": float(T[n]),
            "u_ml": np.round(pred[n], 4).tolist(),
            "u_hybrid": np.round(hybrid[n], 4).tolist(),
            "true": np.round(true[n], 4).tolist(),
            "trust": round(float(trust[n]), 3),
            "ok": bool(not okc[n]),
            "switch_t": (float(T[switch]) if switch is not None else None),
            "ml_steps": int(min(n + 1, s)),
            "corr_steps": int(max(0, n + 1 - s)),
            "hybrid_error": round(_relerr(hybrid[n], true[n]), 3),
            "ml_error": round(_relerr(pred[n], true[n]), 3),
            "cost_s": round(min(n + 1, s) * ml_per + max(0, n + 1 - s) * num_per, 3),
            "cost_num_so_far": round((n + 1) * num_per, 3),
        }))
        await asyncio.sleep(0.03)
    ml_steps, corr_steps = s, nt - s
    await ws.send_text(json.dumps({"done": True, "summary": {
        "ml_steps": int(ml_steps),
        "corr_steps": int(corr_steps),
        "cost_hybrid": round(ml_steps * ml_per + corr_steps * num_per, 2),
        "cost_ml": round(ml_per * nt, 2),
        "cost_num": round(num_per * nt, 2),
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


# ---------------- live cost-control run (Module 3) ----------------

def _per_step_rates(nt):
    """Per-step cost rates taken from the measured timed frontier, so a live run
    lands on the same frontier the Findings tab shows."""
    ml_per = num_per = None
    try:
        p = os.path.join(core.ROOT, "results", "m3", "step9d_coarse_integration", "timed_cost_result_m2.json")
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        ml_per = d["pure_ml"]["cost_s"] / nt
        num_per = d["pure_numerical"]["cost_s"] / nt
    except Exception as e:
        raise RuntimeError(
            "per-step costs unavailable: could not read timed_cost_result_m2.json (%s). "
            "Run: python -m hybrid_pde.control_214133E._run_full" % e)
    return ml_per, num_per


def _cc_setup(model, ic, pinn_index, target):
    from hybrid_pde.control_214133E.controller import AdaptiveController, thresholds_for_target
    from hybrid_pde.trust.coarse_reference import CoarseReferenceMonitor
    ic0, pred, true = core.get_prediction(model, ic, pinn_index)
    lo, hi = thresholds_for_target(float(target))
    mon = CoarseReferenceMonitor(ic0, core.X, n=256)
    ctrl = AdaptiveController(lo, 1.1)
    ctrl.configure(float(target))
    ctrl.reset()
    return ic0, pred, true, mon, ctrl, lo, hi


async def _run_costcontrol(ws, req):
    model = req.get("model", "FNO")
    ic = req.get("ic")
    pinn_index = int(req.get("pinn_index", 0))
    target = float(req.get("target", 0.05))
    loop = asyncio.get_running_loop()
    ic0, pred, true, mon, ctrl, lo, hi = await loop.run_in_executor(
        None, _cc_setup, model, ic, pinn_index, target)

    X, T = core.X, core.T
    nt = len(T)
    ml_per, num_per = _per_step_rates(nt)
    coupling, num = M2Coupling(), _NumSolver()

    state = np.asarray(ic0, dtype=float)
    prev_t = float(T[0])
    ml_steps = corr_steps = 0
    cost = 0.0
    switch_t = None

    for n in range(nt):
        o = mon.update(state, float(T[n]))
        trust = float(o["trust"])
        flag = not bool(o["ok"])
        dec = ctrl.decide(trust, flag, float(T[n]), n)
        if dec.correct:
            state = np.asarray(coupling.correct(state, X, prev_t, float(T[n]), num), dtype=float)
            corr_steps += 1
            cost += num_per
            if switch_t is None:
                switch_t = float(T[n])
        else:
            state = np.asarray(pred[n], dtype=float)
            ml_steps += 1
            cost += ml_per
        prev_t = float(T[n])
        err = float(np.linalg.norm(state - true[n]) / (np.linalg.norm(true[n]) + 1e-12))
        await ws.send_text(json.dumps({
            "t": float(T[n]),
            "u": np.round(state, 4).tolist(),
            "true": np.round(true[n], 4).tolist(),
            "trust": round(trust, 3),
            "correcting": bool(dec.correct),
            "ml_steps": ml_steps,
            "corr_steps": corr_steps,
            "cost_s": round(cost, 4),
            "error": round(err, 4),
            "switch_t": switch_t,
        }))
        await asyncio.sleep(0.02)

    final_err = float(np.linalg.norm(state - true[-1]) / (np.linalg.norm(true[-1]) + 1e-12))
    await ws.send_text(json.dumps({"summary": {
        "target": target, "theta_lo": round(lo, 2), "theta_hi": None, "latched": True,
        "ml_steps": ml_steps, "corr_steps": corr_steps,
        "cost_s": round(cost, 3), "error": round(final_err, 4),
        "hit": bool(final_err <= target),
        "rel_cost": round(cost / max(num_per * nt, 1e-9), 3),
        "switch_t": switch_t,
    }}))
    await ws.send_text(json.dumps({"done": True}))


@router.websocket("/ws/costcontrol")
async def ws_costcontrol(ws: WebSocket):
    await ws.accept()
    try:
        req = json.loads(await ws.receive_text())
        await _run_costcontrol(ws, req)
    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_text(json.dumps({"error": str(e)}))


async def _run_race(ws, req):
    model = req.get("model", "FNO")
    ic = req.get("ic")
    pinn_index = int(req.get("pinn_index", 0))
    target = float(req.get("target", 0.05))
    loop = asyncio.get_running_loop()
    ic0, pred, true, mon, ctrl, lo, hi = await loop.run_in_executor(
        None, _cc_setup, model, ic, pinn_index, target)

    X, T = core.X, core.T
    nt = len(T)
    ml_per, num_per = _per_step_rates(nt)
    coupling, num = M2Coupling(), _NumSolver()
    numer = cole_hopf_from(ic0, T)

    hyb = np.asarray(ic0, dtype=float)
    prev_t = float(T[0])
    ml_c = num_c = hyb_c = 0.0
    corr_steps = ml_steps = 0
    marks = []

    await ws.send_text(json.dumps({"init": {
        "nt": nt, "t_end": float(T[-1]), "target": target,
        "theta_lo": round(lo, 2), "theta_hi": None, "latched": True,
        "ml_total": round(ml_per * nt, 3), "num_total": round(num_per * nt, 3),
    }}))

    for n in range(nt):
        o = mon.update(hyb, float(T[n]))
        trust = float(o["trust"])
        dec = ctrl.decide(trust, not bool(o["ok"]), float(T[n]), n)
        if dec.correct:
            hyb = np.asarray(coupling.correct(hyb, X, prev_t, float(T[n]), num), dtype=float)
            corr_steps += 1
            hyb_c += num_per
            marks.append(round(float(T[n]), 3))
        else:
            hyb = np.asarray(pred[n], dtype=float)
            ml_steps += 1
            hyb_c += ml_per
        prev_t = float(T[n])
        ml_c += ml_per
        num_c += num_per

        e = lambda a: round(float(np.linalg.norm(a - true[n]) / (np.linalg.norm(true[n]) + 1e-12)), 5)
        await ws.send_text(json.dumps({
            "t": round(float(T[n]), 3), "i": n,
            "trust": round(trust, 3), "correcting": bool(dec.correct),
            "ml":  {"cost": round(ml_c, 4),  "err": e(pred[n]),  "u": np.round(pred[n], 3).tolist()},
            "num": {"cost": round(num_c, 4), "err": e(numer[n]), "u": np.round(numer[n], 3).tolist()},
            "hyb": {"cost": round(hyb_c, 4), "err": e(hyb),      "u": np.round(hyb, 3).tolist()},
            "corr_steps": corr_steps, "ml_steps": ml_steps,
        }))
        await asyncio.sleep(0.035)

    fe = lambda a: round(float(np.linalg.norm(a - true[-1]) / (np.linalg.norm(true[-1]) + 1e-12)), 5)
    await ws.send_text(json.dumps({"summary": {
        "target": target, "corr_steps": corr_steps, "ml_steps": ml_steps, "marks": marks,
        "ml":  {"cost": round(ml_c, 3),  "err": fe(pred[-1])},
        "num": {"cost": round(num_c, 3), "err": fe(numer[-1])},
        "hyb": {"cost": round(hyb_c, 3), "err": fe(hyb)},
        "saving": round(1.0 - hyb_c / max(num_c, 1e-9), 3),
        "speedup": round(num_c / max(hyb_c, 1e-9), 2),
        "acc_gain": round(fe(pred[-1]) / max(fe(hyb), 1e-9), 2),
    }}))
    await ws.send_text(json.dumps({"done": True}))


@router.websocket("/ws/race")
async def ws_race(ws: WebSocket):
    await ws.accept()
    try:
        await _run_race(ws, json.loads(await ws.receive_text()))
    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_text(json.dumps({"error": str(e)}))
