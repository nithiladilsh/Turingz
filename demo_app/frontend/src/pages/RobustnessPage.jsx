import { useEffect, useRef, useState } from "react";
import { Card, Stat, Banner } from "../components/ui.jsx";
import { LineChart } from "../components/Charts.jsx";
import { getMeta, buildIC, runRobustness } from "../api.js";

const MODELS = ["FNO", "DeepONet"];
const PRESETS = [
  ["in_dist", "familiar wave", "sin(πx) — inside the training family"],
  ["high_freq", "wigglier wave", "sin(6πx) — a frequency the models never saw (trained on modes 1–4)"],
  ["gaussian", "localized bump", "a shape unlike any training sinusoid"],
  ["custom", "build your own", "use the sliders — more modes = further out of distribution"],
];
const inactiveBtn = "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700";

export default function RobustnessPage() {
  const [meta, setMeta] = useState(null);
  const [model, setModel] = useState("FNO");
  const [preset, setPreset] = useState("high_freq");
  const [modes, setModes] = useState(6);
  const [amplitude, setAmplitude] = useState(1.0);
  const [ic, setIc] = useState(null);
  const [frame, setFrame] = useState(null);
  const [hist, setHist] = useState([]);
  const [summary, setSummary] = useState(null);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => { getMeta().then(setMeta).catch(() => setErr("Backend not reachable. Start it with: uvicorn main:app")); }, []);
  useEffect(() => {
    if (!meta || preset !== "custom") return;
    buildIC(modes, amplitude).then((d) => setIc(d.ic)).catch(() => {});
  }, [meta, preset, modes, amplitude]);

  function run() {
    if (wsRef.current) wsRef.current.close();
    setHist([]); setFrame(null); setSummary(null); setErr(null); setRunning(true);
    const payload = preset === "custom" ? { model, ic } : { model, preset };
    wsRef.current = runRobustness(payload,
      (f) => { setFrame(f); setHist((h) => [...h, f]); },
      (s) => setSummary(s),
      () => setRunning(false),
      (e) => { setErr(e); setRunning(false); });
  }

  const x = meta?.x || [];
  const muted = "text-slate-500 dark:text-slate-400";
  const yr = frame
    ? [Math.min(...frame.true, ...frame.u, -1.1), Math.max(...frame.true, ...frame.u, 1.1)]
    : [-1.1, 1.1];
  const maxErr = Math.max(0.3, ...hist.map((f) => f.err));
  const maxSd = Math.max(0.3, ...hist.map((f) => f.sd));

  return (
    <div>
      <h1 className="text-2xl font-bold text-slate-800 dark:text-slate-100">Robustness — where the ML models fail</h1>
      <p className="text-slate-600 dark:text-slate-300 mt-1 max-w-3xl text-sm">
        Feed a trained model an unfamiliar input and watch it fail once it predicts the future — while a
        frequency-space (spectral) signal tracks the failure. This measured failure is what the coupling
        module corrects, and the same spectral idea returns there as the restart-safety diagnostic.
      </p>
      {err && <div className="mt-3 text-sm text-rose-600 dark:text-rose-300 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 rounded-lg px-3 py-2">{err}</div>}

      <div className="mt-5 grid grid-cols-[320px_1fr] gap-5">
        <div className="space-y-4">
          <Card title="1. Choose the ML model"
            subtitle="PINN is trained per wave, so out-of-distribution inputs don't apply to it.">
            <div className="flex gap-2">
              {MODELS.map((m) => (
                <button key={m} onClick={() => setModel(m)}
                  className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium border ${model === m
                    ? "bg-indigo-600 text-white border-indigo-600" : inactiveBtn}`}>
                  {m}
                </button>
              ))}
            </div>
          </Card>

          <Card title="2. Choose how unfamiliar the input is">
            <div className="space-y-2">
              {PRESETS.map(([v, label, desc]) => (
                <button key={v} onClick={() => setPreset(v)}
                  className={`w-full text-left px-3 py-2 rounded-lg text-sm border ${preset === v
                    ? "bg-indigo-600 text-white border-indigo-600" : inactiveBtn}`}>
                  <div className="font-medium">{label}</div>
                  <div className={`text-xs ${preset === v ? "text-indigo-100" : "text-slate-400 dark:text-slate-500"}`}>{desc}</div>
                </button>
              ))}
            </div>
            {preset === "custom" && (
              <div className="space-y-3 text-sm mt-3">
                <label className={`block ${muted}`}>
                  modes: {modes} {modes > 4 && <span className="text-amber-600 dark:text-amber-400">(beyond trained band)</span>}
                  <input type="range" min="1" max="8" value={modes}
                    onChange={(e) => setModes(+e.target.value)} className="w-full" />
                </label>
                <label className={`block ${muted}`}>
                  amplitude: {amplitude.toFixed(2)}
                  <input type="range" min="0.2" max="1.5" step="0.05" value={amplitude}
                    onChange={(e) => setAmplitude(+e.target.value)} className="w-full" />
                </label>
              </div>
            )}
          </Card>

          <button onClick={run} disabled={running || (preset === "custom" && !ic)}
            className="w-full px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium text-sm transition">
            {running ? "Running…" : "Run the model"}
          </button>

          {summary && (
            <Card title="Result">
              <div className="grid grid-cols-2 gap-2">
                <Stat label="error in-window (t ≤ 1)" value={`${(summary.in_window_err * 100).toFixed(1)}%`} tone="green" />
                <Stat label="error extrapolating (t > 1)" value={`${(summary.extrap_err * 100).toFixed(1)}%`} tone="red" />
                <Stat label="reliable horizon (10% band)" value={`t = ${summary.reliable_horizon.toFixed(2)}`} tone="indigo" />
                <Stat label="failure detected without truth?" value={summary.extrap_err > 0.1 ? "spectral signal rises" : "stays reliable"} />
              </div>
              <div className="mt-3">
                <Banner ok={summary.extrap_err < 0.1}
                  text={summary.extrap_err < 0.1
                    ? "The model stays reliable on this input."
                    : "The model fails on this input — exactly the failure the coupling module corrects."} />
              </div>
            </Card>
          )}
        </div>

        <div className="space-y-4">
          <Card title="Live solution"
            subtitle={frame ? `t = ${frame.t.toFixed(2)}${frame.t > 1 ? " — extrapolating beyond training" : " — inside the training window"}` : "run to start"}>
            <LineChart
              series={[
                { x, y: frame ? frame.true : [], color: "#94a3b8", dashed: true },
                { x, y: frame ? frame.u : [], color: "#e11d48", width: 2.5 },
              ]}
              xr={[-1, 1]} yr={yr} h={220} xlabel="x" ylabel="u(x, t)" />
            <div className="flex gap-4 mt-1 text-xs">
              <span className="text-slate-400">— true (Cole–Hopf)</span>
              <span className="text-rose-500 font-medium">— {model} prediction</span>
            </div>
          </Card>

          <div className="grid grid-cols-2 gap-4">
            <Card title="Error vs time" subtitle="how wrong the model is — climbs after the dashed training horizon">
              <LineChart
                series={[{ x: hist.map((f) => f.t), y: hist.map((f) => f.err), color: "#e11d48", width: 2.5 }]}
                xr={[0, 2]} yr={[0, maxErr]} vline={1.0} hline={0.1} h={180}
                xlabel="t" ylabel="relative L2 error" />
              <p className={`text-[11px] mt-1 ${muted}`}>dotted line = 10% failure band · red vertical = training horizon t = 1</p>
            </Card>
            <Card title="Spectral distance vs time" subtitle="my frequency-space signal — tracks the failure alongside the error">
              <LineChart
                series={[{ x: hist.map((f) => f.t), y: hist.map((f) => f.sd), color: "#2563eb", width: 2.5 }]}
                xr={[0, 2]} yr={[0, maxSd]} vline={1.0} h={180}
                xlabel="t" ylabel="spectral distance" />
              <p className={`text-[11px] mt-1 ${muted}`}>the same spectral-content idea is the restart-safety diagnostic on the Coupling page</p>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
