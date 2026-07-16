import { useEffect, useRef, useState } from "react";
import { Card, Stat, Banner } from "../components/ui.jsx";
import { LineChart, Gauge } from "../components/Charts.jsx";
import { getMeta, buildIC, pinnIC, runTrust } from "../api.js";

const MODELS = ["FNO", "DeepONet", "PINN"];

export default function TrustPage() {
  const [meta, setMeta] = useState(null);
  const [model, setModel] = useState("FNO");
  const [modes, setModes] = useState(4);
  const [amplitude, setAmplitude] = useState(1.0);
  const [pinnIndex, setPinnIndex] = useState(0);
  const [fnoMode, setFnoMode] = useState("reference_free");
  const [ic, setIc] = useState(null);
  const [frame, setFrame] = useState(null);
  const [hist, setHist] = useState([]);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => { getMeta().then(setMeta).catch(() => setErr("Backend not reachable. Start it with: uvicorn main:app")); }, []);

  useEffect(() => {
    if (!meta) return;
    if (model === "PINN") pinnIC(pinnIndex).then((d) => setIc(d.ic)).catch(() => {});
    else buildIC(modes, amplitude).then((d) => setIc(d.ic)).catch(() => {});
  }, [meta, model, modes, amplitude, pinnIndex]);

  function run() {
    if (wsRef.current) wsRef.current.close();
    setHist([]); setFrame(null); setErr(null); setRunning(true);
    const payload = model === "PINN"
      ? { model, pinn_index: pinnIndex }
      : { model, ic, mode: model === "FNO" ? fnoMode : "reference_free" };
    wsRef.current = runTrust(payload,
      (f) => { setFrame(f); setHist((h) => [...h, f]); },
      () => setRunning(false),
      (e) => { setErr(e); setRunning(false); });
  }

  const x = meta?.x || [];
  const p = meta?.params?.[model];
  const cut = p?.CUT ?? 0.5;
  const ood = model !== "PINN" && modes > 4;

  return (
    <div>
      <h1 className="text-2xl font-bold text-slate-800">Trust Score — live demo</h1>
      <p className="text-slate-600 mt-1 max-w-3xl text-sm">
        Give a starting wave, run the ML model, and watch the trust score fall and the switch fire — all with no true answer used by the module.
      </p>
      {err && <div className="mt-3 text-sm text-rose-600 bg-rose-50 border border-rose-200 rounded-lg px-3 py-2">{err}</div>}

      <div className="mt-5 grid grid-cols-[320px_1fr] gap-5">
        {/* CONTROLS */}
        <div className="space-y-4">
          <Card title="1. Choose the ML model">
            <div className="flex gap-2">
              {MODELS.map((m) => (
                <button key={m} onClick={() => setModel(m)}
                  className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium border ${model === m
                    ? "bg-indigo-600 text-white border-indigo-600" : "bg-white text-slate-600 border-slate-200"}`}>
                  {m}
                </button>
              ))}
            </div>
          </Card>

          <Card title="2. Set the initial condition"
            subtitle={model === "PINN"
              ? "PINN is trained per wave — pick one of its trained inputs."
              : "Build a wave. More modes = sharper, more unfamiliar input."}>
            {model === "PINN" ? (
              <select value={pinnIndex} onChange={(e) => setPinnIndex(+e.target.value)}
                className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm">
                {Array.from({ length: meta?.n_pinn_ics || 0 }, (_, i) => (
                  <option key={i} value={i}>Trained wave #{i}</option>
                ))}
              </select>
            ) : (
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-xs text-slate-500"><span>Sine modes</span><span>{modes}</span></div>
                  <input type="range" min="1" max="12" value={modes} onChange={(e) => setModes(+e.target.value)} className="w-full" />
                  {ood && <div className="text-xs text-amber-600 mt-1">Above 4 modes = out-of-distribution (unfamiliar) input</div>}
                </div>
                <div>
                  <div className="flex justify-between text-xs text-slate-500"><span>Amplitude</span><span>{amplitude.toFixed(1)}</span></div>
                  <input type="range" min="0.5" max="1.5" step="0.1" value={amplitude} onChange={(e) => setAmplitude(+e.target.value)} className="w-full" />
                </div>
              </div>
            )}
            {model === "FNO" && (
              <div className="mt-3">
                <div className="text-xs text-slate-500 mb-1">Trust mode</div>
                <div className="flex gap-2">
                  {[["reference_free", "Reference-free"], ["coarse", "Cheap-reference"]].map(([v, l]) => (
                    <button key={v} onClick={() => setFnoMode(v)}
                      className={`flex-1 px-2 py-1.5 rounded-lg text-xs border ${fnoMode === v
                        ? "bg-slate-800 text-white border-slate-800" : "bg-white text-slate-600 border-slate-200"}`}>{l}</button>
                  ))}
                </div>
              </div>
            )}
            {ic && (
              <div className="mt-3">
                <div className="text-xs text-slate-400 mb-1">Starting wave preview</div>
                <LineChart series={[{ x, y: ic, color: "#6366f1", width: 2 }]} xr={[-1, 1]} yr={[-1.6, 1.6]} h={130} xlabel="x" />
              </div>
            )}
          </Card>

          <button onClick={run} disabled={running || !ic}
            className="w-full px-4 py-2.5 rounded-lg bg-emerald-600 text-white text-sm font-semibold hover:bg-emerald-700 disabled:opacity-50">
            {running ? "Running…" : "▶ Run simulation"}
          </button>
        </div>

        {/* RESULTS */}
        <div className="space-y-4">
          <Banner ok={!frame || frame.ok}
            text={!frame ? "Ready — press run" : frame.ok
              ? `Using the ML model (fast) — t = ${frame.t.toFixed(2)}`
              : `Switched to the numerical solver at t = ${(frame.switch_t ?? 0).toFixed(2)}`} />

          <div className="grid grid-cols-2 gap-4">
            <Card title="Solution wave u(x, t)" subtitle={frame ? `time t = ${frame.t.toFixed(2)}` : "—"}>
              <LineChart h={190} xr={[-1, 1]} yr={[-1.6, 1.6]} xlabel="x"
                series={frame ? [
                  { x, y: frame.true, color: "#94a3b8", dashed: true, width: 1.5 },
                  { x, y: frame.u, color: frame.ok ? "#059669" : "#e11d48", width: 2.5 },
                ] : []} />
              <div className="text-xs text-slate-400 mt-1">green/red = ML prediction · grey dashed = true answer</div>
            </Card>

            <Card title="Trust score">
              <Gauge value={frame ? frame.trust : 1} />
              <div className="grid grid-cols-2 gap-2 mt-3">
                <Stat label="switch fired at" value={frame?.switch_t != null ? frame.switch_t.toFixed(2) : "—"} tone="red" />
                <Stat label="true error now" value={frame ? `${Math.round(frame.true_error * 100)}%` : "—"} />
              </div>
            </Card>
          </div>

          <Card title="Trust and true error over time" subtitle={`switch fires when trust stays below ${cut} for ${p?.K ?? 4} steps`}>
            <LineChart h={200} xr={[0, 2]} yr={[0, 1]} hline={cut} vline={frame?.switch_t ?? null} xlabel="time t"
              series={[
                { x: hist.map((f) => f.t), y: hist.map((f) => f.trust), color: "#4f46e5", width: 2.5 },
                { x: hist.map((f) => f.t), y: hist.map((f) => Math.min(1, f.true_error)), color: "#94a3b8", dashed: true, width: 1.5 },
              ]} />
            <div className="text-xs text-slate-400 mt-1">indigo = trust · grey dashed = true error · dashed line = cutoff · red line = switch</div>
          </Card>

          <Card title="How the trust score is calculated (live)">
            {frame ? (
              <div className="text-sm text-slate-600 space-y-2">
                <div className="grid grid-cols-3 gap-2">
                  <Stat label="physics residual" value={frame.signals.residual.toFixed(3)} />
                  <Stat label="energy drift" value={frame.signals.energy.toFixed(3)} />
                  <Stat label="roughness" value={frame.signals.roughness.toFixed(3)} />
                </div>
                <p>
                  {model === "FNO" && fnoMode === "coarse"
                    ? "Cheap-reference mode: a small coarse solver runs alongside FNO and the trust score comes from how far FNO has drifted from it."
                    : "These reference-free signals are combined into one fused number, calibrated to a 0–1 trust score. When the score stays below the cutoff for a few steps in a row, the switch latches on."}
                </p>
                <p className="text-slate-500">
                  cutoff = {cut} · patience K = {p?.K ?? 4} · fail tolerance = 10% error.
                </p>
              </div>
            ) : <p className="text-sm text-slate-400">Run a simulation to see the live signal breakdown.</p>}
          </Card>
        </div>
      </div>
    </div>
  );
}
