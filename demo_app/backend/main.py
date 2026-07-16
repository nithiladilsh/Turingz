import json
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import core

app = FastAPI(title="Turingz Hybrid PDE Demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


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
