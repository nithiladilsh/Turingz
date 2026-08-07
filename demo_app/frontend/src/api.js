export const API = "http://localhost:8000";
export const WS = "ws://localhost:8000";

export async function getMeta() {
  const r = await fetch(`${API}/api/meta`);
  return r.json();
}

export async function buildIC(modes, amplitude, phase = 0, seed = 0) {
  const r = await fetch(`${API}/api/build_ic`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ modes, amplitude, phase, seed }),
  });
  return r.json();
}

export async function pinnIC(index) {
  const r = await fetch(`${API}/api/pinn_ic/${index}`);
  return r.json();
}

// one of the 10 official held-out test ICs (dataset index 900+index) --
// the exact set the reported hit-rate numbers were measured on
export async function realTestIC(index) {
  const r = await fetch(`${API}/api/real_test_ic/${index}`);
  return r.json();
}

export async function pinnRegime() {
  const r = await fetch(`${API}/api/m3/pinn_regime`);
  return r.json();
}

export async function switchingAblation() {
  const r = await fetch(`${API}/api/m3/switching_ablation`);
  return r.json();
}

// opens a websocket at `path`, calls onFrame(frame) for each streamed frame, onDone when finished
function runWS(path, payload, onFrame, onDone, onError) {
  const ws = new WebSocket(`${WS}${path}`);
  ws.onopen = () => ws.send(JSON.stringify(payload));
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    // backend error envelope is always a string; data frames may legitimately
    // carry a numeric field named "error" (e.g. reliability), so only treat
    // string errors as failures.
    if (typeof msg.error === "string") return onError && onError(msg.error);
    if (msg.done) { onDone && onDone(); ws.close(); return; }
    onFrame(msg);
  };
  ws.onerror = () => onError && onError("Could not reach backend at " + WS + ". Is it running?");
  return ws;
}

export const runTrust = (payload, onFrame, onDone, onError) => runWS("/ws/trust", payload, onFrame, onDone, onError);
export const runFDM = (payload, onFrame, onDone, onError) => runWS("/ws/fdm", payload, onFrame, onDone, onError);
export const runReliability = (payload, onFrame, onDone, onError) => runWS("/ws/reliability", payload, onFrame, onDone, onError);

export async function fdmEval() {
  const r = await fetch(`${API}/api/fdm_eval`);
  return r.json();
}
export async function getCouplingMeta() {
  const r = await fetch(`${API}/api/coupling_meta`);
  return r.json();
}

// opens a websocket for the coupling demo; onFrame per streamed frame,
// onSummary for the final metrics, onDone when finished
export function runCoupling(payload, onFrame, onSummary, onDone, onError) {
  const ws = new WebSocket(`${WS}/ws/coupling`);
  ws.onopen = () => ws.send(JSON.stringify(payload));
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (typeof msg.error === "string") return onError && onError(msg.error);
    if (msg.done) { onDone && onDone(); ws.close(); return; }
    if (msg.summary) return onSummary && onSummary(msg.summary);
    onFrame(msg);
  };
  ws.onerror = () => onError && onError("Could not reach backend at " + WS + ". Is it running?");
  return ws;
}

function wsRun(path, payload, onFrame, onSummary, onDone, onError) {
  const ws = new WebSocket(`${WS}${path}`);
  ws.onopen = () => ws.send(JSON.stringify(payload));
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (typeof msg.error === "string") return onError && onError(msg.error);
    if (msg.done) { onDone && onDone(); ws.close(); return; }
    if (msg.summary) return onSummary && onSummary(msg.summary);
    onFrame(msg);
  };
  ws.onerror = () => onError && onError("Could not reach backend at " + WS + ". Is it running?");
  return ws;
}

export const runColeHopf = (payload, ...cbs) => wsRun("/ws/colehopf", payload, ...cbs);
export const runRobustness = (payload, ...cbs) => wsRun("/ws/robustness", payload, ...cbs);
