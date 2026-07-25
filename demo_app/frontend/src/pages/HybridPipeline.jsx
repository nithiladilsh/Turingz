import { useEffect, useRef, useState } from "react";
import { Card, Stat, Banner } from "../components/ui.jsx";
import { LineChart, Gauge } from "../components/Charts.jsx";
import { WS, getMeta, buildIC, pinnIC } from "../api.js";

const MODELS = ["FNO", "PINN", "DeepONet"];
const inactiveBtn = "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700";

// self-contained socket so this page never edits shared api.js
function runPipeline(payload, onFrame, onDone, onError) {
  const ws = new WebSocket(`${WS}/ws/pipeline`);
  ws.onopen = () => ws.send(JSON.stringify(payload));
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.error) return onError && onError(m.error);
    if (m.done) { onDone && onDone(m.summary); ws.close(); return; }
    onFrame(m);
  };
  ws.onerror = () => onError && onError("Could not reach backend at " + WS + ". Is it running?");
  return ws;
}

function Bar({ label, value, max, display, color }) {
  const pct = Math.max(2, Math.min(100, (value / (max || 1)) * 100));
  return (
    <div className="flex items-center gap-2 text-xs">
      <div className="w-32 text-slate-500 dark:text-slate-400">{label}</div>
      <div className="flex-1 bg-slate-100 dark:bg-slate-700/40 rounded h-5">
        <div className="h-5 rounded" style={{ width: pct + "%", background: color }} />
      </div>
      <div className="w-20 text-right text-slate-700 dark:text-slate-200">{display}</div>
    </div>
  );
}

export default function HybridPipeline() {
  const [meta, setMeta] = useState(null);
  const [model, setModel] = useState("FNO");
  const [modes, setModes] = useState(6);
  const [amplitude, setAmplitude] = useState(1.0);
  const [pinnIndex, setPinnIndex] = useState(0);
  const [fnoMode, setFnoMode] = useState("coarse");
  const [ic, setIc] = useState(null);
  const [frame, setFrame] = useState(null);
  const [hist, setHist] = useState([]);
  const [summary, setSummary] = useState(null);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => { getMeta().then(setMeta).catch(() => setErr("Backend not reachable. Start it with: uvicorn main:app --port 8000")); }, []);

  useEffect(() => {
    if (!meta) return;
    if (model === "PINN") pinnIC(pinnIndex).then((d) => setIc(d.ic)).catch(() => {});
    else buildIC(modes, amplitude).then((d) => setIc(d.ic)).catch(() => {});
  }, [meta, model, modes, amplitude, pinnIndex]);

  function run() {
    if (wsRef.current) wsRef.current.close();
    setHist([]); setFrame(null); setSummary(null); setErr(null); setRunning(true);
    const payload = model === "PINN"
      ? { model, pinn_index: pinnIndex }
      : { model, ic, mode: model === "FNO" ? fnoMode : "reference_free" };
    wsRef.current = runPipeline(payload,
      (f) => { setFrame(f); setHist((h) => [...h, f]); },
      (s) => { setSummary(s); setRunning(false); },
      (e) => { setErr(e); setRunning(false); });
  }

  const x = meta?.x || [];
  const muted = "text-slate-500 dark:text-slate-400";
  const faint = "text-slate-400 dark:text-slate-500";
  const ood = model !== "PINN" && modes > 4;
  const switched = frame && !frame.ok;
  const pct = (v) => `${Math.round((v || 0) * 100)}%`;
  const errMax = summary ? Math.max(summary.err_ml, summary.err_hybrid, 0.001) : 1;
  const costMax = summary ? summary.cost_num : 1;

  return (
    <div>
      <h1 className="text-2xl font-bold text-slate-800 dark:text-slate-100">Hybrid engine — full pipeline</h1>
      <p className="text-slate-600 dark:text-slate-300 mt-1 max-w-3xl text-sm">
        The whole system in one run: the ML model predicts, <b>Module 1</b> scores trust and decides when to switch,
        <b> Module 2</b> hands the state over to the numerical solver, and <b>Module 3</b> accounts for the cost —
        landing at usable accuracy for a fraction of the numerical cost.
      </p>
      {err && <div className="mt-3 text-sm text-rose-600 dark:text-rose-300 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 rounded-lg px-3 py-2">{err}</div>}

      <div className="mt-5 grid grid-cols-[320px_1fr] gap-5">
        {/* CONTROLS */}
        <div className="space-y-4">
          <Card title="1. ML model">
            <div className="flex gap-2">
              {MODELS.map((m) => (
                <button key={m} onClick={() => setModel(m)}
                  className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium border ${model === m ? "bg-indigo-600 text-white border-indigo-600" : inactiveBtn}`}>{m}</button>
              ))}
            </div>
          </Card>

          <Card title="2. Initial condition"
            subtitle={model === "PINN" ? "PINN is trained per wave — pick one." : "More modes = sharper, more out-of-distribution."}>
            {model === "PINN" ? (
              <select value={pinnIndex} onChange={(e) => setPinnIndex(+e.target.value)}
                className="w-full bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-2 text-sm">
                {Array.from({ length: meta?.n_pinn_ics || 0 }, (_, i) => (<option key={i} value={i}>Trained wave #{i}</option>))}
              </select>
            ) : (
              <div className="space-y-3">
                <div>
                  <div className={`flex justify-between text-xs ${muted}`}><span>Sine modes</span><span>{modes}</span></div>
                  <input type="range" min="1" max="12" value={modes} onChange={(e) => setModes(+e.target.value)} className="w-full" />
                  {ood && <div className="text-xs text-amber-600 dark:text-amber-400 mt-1">Above 4 modes = out-of-distribution — ML will drift.</div>}
                </div>
                <div>
                  <div className={`flex justify-between text-xs ${muted}`}><span>Amplitude</span><span>{amplitude.toFixed(1)}</span></div>
                  <input type="range" min="0.5" max="1.5" step="0.1" value={amplitude} onChange={(e) => setAmplitude(+e.target.value)} className="w-full" />
                </div>
              </div>
            )}
            {model === "FNO" && (
              <div className="mt-3">
                <div className={`text-xs mb-1 ${muted}`}>Trust mode (Module 1)</div>
                <div className="flex gap-2">
                  {[["reference_free", "Reference-free"], ["coarse", "Cheap-reference"]].map(([v, l]) => (
                    <button key={v} onClick={() => setFnoMode(v)}
                      className={`flex-1 px-2 py-1.5 rounded-lg text-xs border ${fnoMode === v ? "bg-slate-800 dark:bg-slate-600 text-white border-slate-800 dark:border-slate-600" : inactiveBtn}`}>{l}</button>
                  ))}
                </div>
              </div>
            )}
            {ic && (
              <div className="mt-3">
                <div className={`text-xs mb-1 ${faint}`}>Starting wave</div>
                <LineChart series={[{ x, y: ic, color: "#6366f1", width: 2 }]} xr={[-1, 1]} yr={[-1.6, 1.6]} h={120} xlabel="x" />
              </div>
            )}
          </Card>

          <button onClick={run} disabled={running || !ic}
            className="w-full px-4 py-2.5 rounded-lg bg-emerald-600 text-white text-sm font-semibold hover:bg-emerald-700 disabled:opacity-50">
            {running ? "Running…" : "▶ Run full pipeline"}
          </button>
        </div>

        {/* RESULTS */}
        <div className="space-y-4">
          <Banner ok={!switched}
            text={!frame ? "Ready — press run" : switched
              ? `Module 2 hand-off active — numerical correction running (switched at t = ${(frame.switch_t ?? 0).toFixed(2)})`
              : `Module 1 trusts the ML model — running fast (t = ${frame.t.toFixed(2)})`} />

          <Card title="Full hybrid solution u(x, t)" subtitle={frame ? `time t = ${frame.t.toFixed(2)}` : "—"}>
            <LineChart h={230} xr={[-1, 1]} yr={[-1.6, 1.6]} xlabel="x"
              series={frame ? [
                { x, y: frame.true, color: "#94a3b8", dashed: true, width: 1.5 },
                { x, y: frame.u_ml, color: "#e11d48", dashed: true, width: 1.5 },
                { x, y: frame.u_hybrid, color: "#059669", width: 2.5 },
              ] : []} />
            <div className={`text-xs mt-1 ${faint}`}>green = hybrid (tracks truth) · red dashed = pure ML (drifts) · grey dashed = true answer</div>
          </Card>

          <div className="grid grid-cols-3 gap-4">
            <Card title="Module 1 — Trust">
              <Gauge value={frame ? frame.trust : 1} />
              <Stat label="switch fired at" value={frame?.switch_t != null ? frame.switch_t.toFixed(2) : "—"} tone="red" />
            </Card>
            <Card title="Module 2 — Hand-off">
              <div className="text-sm text-slate-600 dark:text-slate-300 space-y-2">
                <Banner ok={!switched} text={switched ? "Numerical correction active" : "Not yet — ML still trusted"} />
                <p className={`text-xs ${muted}`}>At the switch the ML state is re-anchored into the verified numerical solver — a stable, jump-free hand-off.</p>
              </div>
            </Card>
            <Card title="Module 3 — Cost">
              <div className="grid grid-cols-1 gap-2">
                <Stat label="ML steps (cheap)" value={frame ? frame.ml_steps : "—"} tone="green" />
                <Stat label="correction steps" value={frame ? frame.corr_steps : "—"} tone="indigo" />
              </div>
            </Card>
          </div>

          <Card title="Cost vs accuracy — only the hybrid clears both bars"
            subtitle={summary ? "final run · lower is better on both" : "run to populate"}>
            {summary ? (
              <div className="space-y-3">
                <div className="space-y-1.5">
                  <div className={`text-xs font-medium ${muted}`}>Error</div>
                  <Bar label="pure ML" value={summary.err_ml} max={errMax} display={pct(summary.err_ml)} color="#e11d48" />
                  <Bar label="hybrid" value={summary.err_hybrid} max={errMax} display={pct(summary.err_hybrid)} color="#059669" />
                  <Bar label="pure numerical" value={summary.err_num} max={errMax} display={pct(summary.err_num)} color="#4f46e5" />
                </div>
                <div className="space-y-1.5">
                  <div className={`text-xs font-medium ${muted}`}>Cost (relative seconds)</div>
                  <Bar label="pure ML" value={summary.cost_ml} max={costMax} display={`${summary.cost_ml}s`} color="#e11d48" />
                  <Bar label="hybrid" value={summary.cost_hybrid} max={costMax} display={`${summary.cost_hybrid}s`} color="#059669" />
                  <Bar label="pure numerical" value={summary.cost_num} max={costMax} display={`${summary.cost_num}s`} color="#4f46e5" />
                </div>
                <p className="text-sm text-slate-600 dark:text-slate-300">
                  Pure ML is cheap but drifts; pure numerical is accurate but slow. The hybrid keeps ML accuracy honest at a fraction of numerical cost.
                </p>
              </div>
            ) : <p className={`text-sm ${faint}`}>Run the pipeline to see the head-to-head.</p>}
          </Card>

          <Card title="Trust and error over time">
            <LineChart h={200} xr={[0, 2]} yr={[0, 1]} vline={frame?.switch_t ?? null} xlabel="time t"
              series={[
                { x: hist.map((f) => f.t), y: hist.map((f) => f.trust), color: "#4f46e5", width: 2.5 },
                { x: hist.map((f) => f.t), y: hist.map((f) => Math.min(1, f.ml_error)), color: "#e11d48", dashed: true, width: 1.5 },
                { x: hist.map((f) => f.t), y: hist.map((f) => Math.min(1, f.hybrid_error)), color: "#059669", width: 2 },
              ]} />
            <div className={`text-xs mt-1 ${faint}`}>indigo = trust · red dashed = pure-ML error · green = hybrid error · red line = switch</div>
          </Card>
        </div>
      </div>
    </div>
  );
}
