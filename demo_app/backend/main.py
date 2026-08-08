import os
import json
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import core

app = FastAPI(title="Turingz Hybrid PDE Demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_RESULTS_M3 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "m3"))
if os.path.isdir(_RESULTS_M3):
    app.mount("/static/m3", StaticFiles(directory=_RESULTS_M3), name="m3_static")


@app.get("/api/meta")
def meta():
    return core.meta()


class ICRequest(BaseModel):
    modes: int = 4
    amplitude: float = 1.0
    phase: float = 0.0
    seed: int = 0


@app.post("/api/build_ic")
def build_ic(req: ICRequest):
    u = core.build_ic(req.modes, req.amplitude, req.phase, req.seed)
    return {"x": core.X.round(4).tolist(), "ic": u.round(4).tolist()}


@app.get("/api/pinn_ic/{index}")
def pinn_ic(index: int):
    return {"x": core.X.round(4).tolist(), "ic": core.PINN_ICS[index].round(4).tolist()}


@app.get("/api/real_test_ic/{index}")
def real_test_ic(index: int):
    """One of the 10 official held-out test ICs (dataset index 900+index) --
    the exact set the reported hit-rate numbers were measured on."""
    ic, _true = core.real_test_ic(index)
    return {"x": core.X.round(4).tolist(), "ic": ic.round(4).tolist()}


@app.websocket("/ws/reliability")
async def ws_reliability(ws: WebSocket):
    await ws.accept()
    try:
        req = json.loads(await ws.receive_text())
        gen = core.reliability_stream(req.get("model", "FNO"), ic=req.get("ic"), pinn_index=req.get("pinn_index", 0))
        for frame in gen:
            await ws.send_text(json.dumps(frame))
            await asyncio.sleep(0.03)
        await ws.send_text(json.dumps({"done": True}))
    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_text(json.dumps({"error": str(e)}))


@app.websocket("/ws/trust")
async def ws_trust(ws: WebSocket):
    await ws.accept()
    try:
        req = json.loads(await ws.receive_text())
        model = req.get("model", "FNO")
        mode = req.get("mode", "reference_free")
        ic = req.get("ic")
        pinn_index = req.get("pinn_index", 0)
        gen = core.stream_run(model, ic=ic, pinn_index=pinn_index, mode=mode)
        for frame in gen:
            await ws.send_text(json.dumps(frame))
            await asyncio.sleep(0.03)
        await ws.send_text(json.dumps({"done": True}))
    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_text(json.dumps({"error": str(e)}))

import pipeline
app.include_router(pipeline.router)

import m3
app.include_router(m3.router)


@app.get("/api/coupling_meta")
def coupling_meta():
    return core.coupling_meta()


@app.websocket("/ws/coupling")
async def ws_coupling(ws: WebSocket):
    await ws.accept()
    try:
        req = json.loads(await ws.receive_text())
        gen = core.stream_coupling(
            req.get("model", "FNO"),
            ic=req.get("ic"),
            pinn_index=req.get("pinn_index", 0),
            switch_mode=req.get("switch_mode", "manual"),
            t_s=req.get("t_s", 1.0),
        )
        for frame in gen:
            await ws.send_text(json.dumps(frame))
            if "summary" not in frame:
                await asyncio.sleep(0.03)
        await ws.send_text(json.dumps({"done": True}))
    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_text(json.dumps({"error": str(e)}))


@app.websocket("/ws/colehopf")
async def ws_colehopf(ws: WebSocket):
    await ws.accept()
    try:
        req = json.loads(await ws.receive_text())
        gen = core.stream_colehopf(req.get("ic"))
        for frame in gen:
            await ws.send_text(json.dumps(frame))
            if "summary" not in frame:
                await asyncio.sleep(0.02)
        await ws.send_text(json.dumps({"done": True}))
    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_text(json.dumps({"error": str(e)}))


@app.websocket("/ws/robustness")
async def ws_robustness(ws: WebSocket):
    await ws.accept()
    try:
        req = json.loads(await ws.receive_text())
        gen = core.stream_robustness(
            req.get("model", "FNO"),
            preset=req.get("preset"),
            ic=req.get("ic"),
            pinn_index=req.get("pinn_index", 0),
        )
        for frame in gen:
            await ws.send_text(json.dumps(frame))
            if "summary" not in frame:
                await asyncio.sleep(0.03)
        await ws.send_text(json.dumps({"done": True}))
    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_text(json.dumps({"error": str(e)}))
