import { useEffect, useRef, useState } from "react";
import { Card, Stat, Banner } from "../components/ui.jsx";
import { LineChart } from "../components/Charts.jsx";
import { getMeta, buildIC, pinnIC, getCouplingMeta, runCoupling } from "../api.js";

const MODELS = ["FNO", "DeepONet", "PINN"];
const inactiveBtn =
  "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700";

const log10 = (v) => (v > 0 ? Math.log10(v) : -12);

export default function CouplingPage() {
  const [meta, setMeta] = useState(null);
  const [cmeta, setCmeta] = useState(null);
  const [model, setModel] = useState("FNO");
  const [modes, setModes] = useState(4);
  const [amplitude, setAmplitude] = useState(1.0);
  const [pinnIndex, setPinnIndex] = useState(0);
  const [switchMode, setSwitchMode] = useState("manual"); // manual | trust
  const [ts, setTs] = useState(1.0);
  const [ic, setIc] = useState(null);
  const [frame, setFrame] = useState(null);
  const [hist, setHist] = useState([]);
  const [summary, setSummary] = useState(null);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => {
    getMeta().then(setMeta).catch(() => setErr("Backend not reachable. Start it with: uvicorn main:app"));
    getCouplingMeta().then(setCmeta).catch(() => {});
  }, []);

  useEffect(() => {
    if (!meta) return;
    if (model === "PINN") pinnIC(pinnIndex).then((d) => setIc(d.ic)).catch(() => {});
    else buildIC(modes, amplitude).then((d) => setIc(d.ic)).catch(() => {});
  }, [meta, model, modes, amplitude, pinnIndex]);

  function run() {
    if (wsRef.current) wsRef.current.close();
    setHist([]); setFrame(null); setSummary(null); setErr(null); setRunning(true);
    const payload = model === "PINN"
      ? { model, pinn_index: pinnIndex, switch_mode: switchMode, t_s: ts }
      : { model, ic, switch_mode: switchMode, t_s: ts };
    wsRef.current = runCoupling(payload,
      (f) => { setFrame(f); setHist((h) => [...h, f]); },
      (s) => setSummary(s),
      () => setRunning(false),
      (e) => { setErr(e); setRunning(false); });
  }

  const x = meta?.x || [];
  const muted = "text-slate-500 dark:text-slate-400";
  const boundaryTs = cmeta?.viability?.boundary_t_s ?? 1.47;
  const sweep = cmeta?.sweep || [];
  const bnd = cmeta?.boundary;

  const yr = frame
    ? [Math.min(...frame.true, ...frame.hybrid, -1.1), Math.max(...frame.true, ...frame.hybrid, 1.1)]
    : [-1.1, 1.1];

  return (
    <div>
      <h1 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
        Coupling — the verified ML→numerical handoff
      </h1>
      <p className="text-slate-600 dark:text-slate-300 mt-1 max-w-3xl text-sm">
        Run the fast ML model, hand the state to the numerical solver at the switch — with zero jump —
        and watch the hybrid stay on the true solution while pure ML drifts. The demo calls the real
        M2Coupling adapter, the same object Module 3&apos;s runtime uses.
      </p>
      {err && (
        <div className="mt-3 text-sm text-rose-600 dark:text-rose-300 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 rounded-lg px-3 py-2">
          {err}
        </div>
      )}

      <div className="mt-5 grid grid-cols-[320px_1fr] gap-5">
        {/* CONTROLS */}
        <div className="space-y-4">
          <Card title="1. Choose the ML model">
            <div className="flex gap-2">
              {MODELS.map((m) => (
                <button key={m} onClick={() => setModel(m)}
                  className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium border ${
                    model === m ? "bg-indigo-600 text-white border-indigo-600" : inactiveBtn}`}>
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
                className="w-full bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-2 text-sm">
                {Array.from({ length: meta?.n_pinn_ics || 0 }, (_, i) => (
                  <option key={i} value={i}>trained wave #{i}</option>
                ))}
              </select>
            ) : (
              <div className="space-y-3 text-sm">
                <label className={`block ${muted}`}>
                  modes: {modes}
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

          <Card title="3. Choose the switch"
            subtitle="Manual = you pick the handoff time. Trust = Module 1's live signal fires it.">
            <div className="flex gap-2">
              {[["manual", "Manual t_s"], ["trust", "Trust-triggered"]].map(([v, label]) => (
                <button key={v} onClick={() => setSwitchMode(v)}
                  className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium border ${
                    switchMode === v ? "bg-indigo-600 text-white border-indigo-600" : inactiveBtn}`}>
                  {label}
                </button>
              ))}
            </div>
            {switchMode === "manual" && (
              <label className={`block mt-3 text-sm ${muted}`}>
                switch time t_s = {ts.toFixed(2)}
                {ts > boundaryTs && (
                  <span className="ml-2 text-amber-600 dark:text-amber-400">
                    past the viability boundary ({boundaryTs.toFixed(2)})
                  </span>
                )}
                <input type="range" min="0.5" max="1.9" step="0.05" value={ts}
                  onChange={(e) => setTs(+e.target.value)} className="w-full" />
              </label>
            )}
          </Card>

          <button onClick={run} disabled={running || !ic}
            className="w-full px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium text-sm transition">
            {running ? "Running…" : "Run the hybrid"}
          </button>

          {summary && (
            <Card title="Result">
              <div className="grid grid-cols-2 gap-2">
                <Stat label="pure-ML error [1,2]" value={`${(summary.ml_tail_1_2 * 100).toFixed(1)}%`} tone="red" />
                <Stat label="hybrid error [1,2]" value={`${(summary.hybrid_tail_1_2 * 100).toFixed(1)}%`} tone="green" />
                <Stat label="benefit" value={summary.benefit != null ? `${(summary.benefit * 100).toFixed(0)}%` : "—"} tone="indigo" />
                <Stat label="numerical work" value={`${(summary.numerical_fraction * 100).toFixed(0)}%`} />
              </div>
              <div className="mt-3">
                <Banner ok={summary.handoff_jump === 0}
                  text={`handoff jump = ${summary.handoff_jump} — continuous by construction`} />
              </div>
            </Card>
          )}
        </div>

        {/* CHARTS */}
        <div className="space-y-4">
          <Card title="Live solution"
            subtitle={frame ? `t = ${frame.t.toFixed(2)}${frame.switched ? " — numerical continuation active" : " — ML rolling"}` : "run to start"}>
            <LineChart
              series={[
                { x, y: frame ? frame.true : [], color: "#94a3b8", dashed: true },
                { x, y: frame ? frame.ml : [], color: "#e11d48" },
                { x, y: frame ? frame.hybrid : [], color: "#4f46e5", width: 2.5 },
              ]}
              xr={[-1, 1]} yr={yr} h={230} xlabel="x" ylabel="u(x, t)" />
            <div className="flex gap-4 mt-1 text-xs">
              <span className="text-slate-400">— true (Cole–Hopf)</span>
              <span className="text-rose-500">— pure ML</span>
              <span className="text-indigo-500 font-medium">— hybrid (M2)</span>
            </div>
          </Card>

          <Card title="Error over time" subtitle="the red line drifts after the training horizon; the hybrid is pinned back at the switch">
            <LineChart
              series={[
                { x: hist.map((f) => f.t), y: hist.map((f) => f.ml_err), color: "#e11d48" },
                { x: hist.map((f) => f.t), y: hist.map((f) => f.hybrid_err), color: "#4f46e5", width: 2.5 },
              ]}
              xr={[0, 2]} yr={[0, Math.max(0.3, ...hist.map((f) => f.ml_err))]}
              vline={frame?.switch_t ?? null} h={190} xlabel="t" ylabel="relative L2 error" />
          </Card>

          <div className="grid grid-cols-2 gap-4">
            <Card title="When does the handoff help?"
              subtitle={`benefit vs switch time (n = 20 held-out waves) — boundary at t_s ≈ ${boundaryTs.toFixed(2)} ≈ FNO reliable horizon`}>
              <LineChart
                series={[{ x: sweep.map((r) => r.t_s), y: sweep.map((r) => r.benefit), color: "#4f46e5", width: 2.5 }]}
                xr={[1.0, 1.8]} yr={[0, 1]} vline={boundaryTs} h={170}
                xlabel="switch time t_s" ylabel="benefit (error reduction)" />
            </Card>

            <Card title="Why the restart must be exact"
              subtitle={`verified vs approximate (no de-aliasing) restart — unsafe beyond Re_cell ≈ ${(cmeta?.boundary?.crossings?.["1pct"] ?? 3.2).toFixed(1)}; approximate blows up at Re ≈ 17.6`}>
              {bnd ? (
                <LineChart
                  series={[
                    { x: bnd.re_cell.map(log10), y: bnd.verified_tail.map(log10), color: "#4f46e5", width: 2.5 },
                    { x: bnd.re_cell.map(log10),
                      y: bnd.careless_tail.map((v) => (v == null ? 0 : log10(v))),
                      color: "#e11d48", dashed: true },
                  ]}
                  xr={[-0.2, 1.4]} yr={[-13, 1]}
                  vline={log10(cmeta?.boundary?.crossings?.["1pct"] ?? 3.2)} h={170}
                  xlabel="log10 Re_cell at handoff" ylabel="log10 tail error" />
              ) : (
                <p className={`text-xs ${muted}`}>boundary data not found</p>
              )}
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
