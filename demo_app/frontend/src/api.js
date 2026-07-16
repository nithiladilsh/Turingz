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

// opens a websocket, calls onFrame(frame) for each streamed frame, onDone when finished
export function runTrust(payload, onFrame, onDone, onError) {
  const ws = new WebSocket(`${WS}/ws/trust`);
  ws.onopen = () => ws.send(JSON.stringify(payload));
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.error) return onError && onError(msg.error);
    if (msg.done) { onDone && onDone(); ws.close(); return; }
    onFrame(msg);
  };
  ws.onerror = () => onError && onError("Could not reach backend at " + WS + ". Is it running?");
  return ws;
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
    if (msg.error) return onError && onError(msg.error);
    if (msg.done) { onDone && onDone(); ws.close(); return; }
    if (msg.summary) return onSummary && onSummary(msg.summary);
    onFrame(msg);
  };
  ws.onerror = () => onError && onError("Could not reach backend at " + WS + ". Is it running?");
  return ws;
}
