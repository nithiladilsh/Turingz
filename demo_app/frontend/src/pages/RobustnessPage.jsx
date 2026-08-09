import { useEffect, useRef, useState } from "react";
import { Card, Stat, Banner } from "../components/ui.jsx";
import { LineChart } from "../components/Charts.jsx";
import { getMeta, buildIC, runRobustness } from "../api.js";
import { ROB_XS, ROB_FRAMES, ROB_STATS as S } from "../robustnessData.js";
// committed reliability numbers (one source of truth for the in-window errors)
import rel from "../../../../results/eval/reliability_summary.json";
import {
  Waves, Clock, Activity, TrendingUp, ShieldAlert, Scale,
  HelpCircle, ArrowRight, Play,
} from "lucide-react";

// in-window errors computed live from the committed reliability_summary.json
const REL_INW = {
  PINN: (rel.reliability_unseen.PINN.in_window * 100).toFixed(1),      // 1.5
  DeepONet: (rel.reliability_unseen.DeepONet.in_window * 100).toFixed(0), // 29
};

function DriftChart({ frame }) {
  const W = 580, H = 300, padX = 18, padT = 18, padB = 28;
  const sx = (x) => padX + ((x + 1) / 2) * (W - 2 * padX);
  const sy = (v) => H - padB - ((v + 1.4) / 2.8) * (H - padT - padB);
  const path = (arr) =>
    arr.map((v, i) => `${i ? "L" : "M"}${sx(ROB_XS[i]).toFixed(1)} ${sy(v).toFixed(1)}`).join(" ");
  let band = `M${sx(ROB_XS[0]).toFixed(1)} ${sy(frame.exact[0]).toFixed(1)}`;
  ROB_XS.forEach((x, i) => (band += `L${sx(x).toFixed(1)} ${sy(frame.exact[i]).toFixed(1)}`));
  for (let i = ROB_XS.length - 1; i >= 0; i--)
    band += `L${sx(ROB_XS[i]).toFixed(1)} ${sy(frame.fno[i]).toFixed(1)}`;
  band += "Z";
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {[-1, -0.5, 0, 0.5, 1].map((v) => (
        <line key={v} x1={sx(-1)} x2={sx(1)} y1={sy(v)} y2={sy(v)}
          stroke="var(--chart-grid)" strokeWidth="1" />
      ))}
      <path d={band} fill="#fb7185" fillOpacity="0.25" />
      <path d={path(frame.exact)} fill="none" stroke="#94a3b8" strokeWidth="2" strokeDasharray="5 4" />
      <path d={path(frame.fno)} fill="none" stroke="#e11d48" strokeWidth="3" strokeLinecap="round" />
      <text x={sx(-1) + 4} y={padT + 4} fontSize="11" fill="#fb7185" fontWeight="700">
        shaded = the FNO&apos;s real error
      </text>
    </svg>
  );
}

function Metric({ icon: Icon, tone, tag, value, label }) {
  const c = {
    rose: ["bg-rose-50 dark:bg-rose-500/10 border-rose-100 dark:border-rose-500/20",
      "bg-rose-100 dark:bg-rose-500/20 text-rose-600 dark:text-rose-400",
      "text-rose-600 dark:text-rose-400"],
    amber: ["bg-amber-50 dark:bg-amber-500/10 border-amber-100 dark:border-amber-500/20",
      "bg-amber-100 dark:bg-amber-500/20 text-amber-600 dark:text-amber-400",
      "text-amber-600 dark:text-amber-400"],
    emerald: ["bg-emerald-50 dark:bg-emerald-500/10 border-emerald-100 dark:border-emerald-500/20",
      "bg-emerald-100 dark:bg-emerald-500/20 text-emerald-600 dark:text-emerald-400",
      "text-emerald-600 dark:text-emerald-400"],
  }[tone];
  return (
    <div className={`rounded-2xl border p-4 hover:-translate-y-1 transition-transform ${c[0]}`}>
      <div className="flex items-center justify-between">
        <div className={`w-9 h-9 rounded-xl grid place-items-center ${c[1]}`}><Icon size={18} /></div>
        {tag && <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">{tag}</span>}
      </div>
      <div className={`text-3xl font-extrabold mt-3 ${c[2]}`}>{value}</div>
      <div className="text-xs text-slate-600 dark:text-slate-300 mt-1 leading-snug">{label}</div>
    </div>
  );
}

function Step({ n, icon: Icon, tone, title, sub }) {
  const c = { indigo: "bg-indigo-600", violet: "bg-violet-600", fuchsia: "bg-fuchsia-600" }[tone];
  return (
    <div className="flex-1 min-w-[150px] rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50/60 dark:bg-slate-700/30 p-3 flex items-center gap-3">
      <div className={`relative w-10 h-10 rounded-xl grid place-items-center text-white shrink-0 ${c}`}>
        <Icon size={18} />
        <span className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-white dark:bg-slate-800 text-[11px] font-bold grid place-items-center text-slate-700 dark:text-slate-200 border border-slate-200 dark:border-slate-600">{n}</span>
      </div>
      <div>
        <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">{title}</div>
        <div className="text-[11px] text-slate-500 dark:text-slate-400 leading-tight">{sub}</div>
      </div>
    </div>
  );
}

const MODELS = ["FNO", "DeepONet"];
const PRESETS = [
  ["in_dist", "familiar wave", "sin(πx) — inside the training family"],
  ["high_freq", "wigglier wave", "sin(6πx) — a frequency never seen (trained on modes 1–4)"],
  ["gaussian", "localized bump", "a shape unlike any training sinusoid"],
  ["custom", "build your own", "more modes = further out of distribution"],
];
const inactiveBtn = "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700";

export default function RobustnessPage() {
  /* ---- auto-loop over the committed FNO frames ---- */
  const [i, setI] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setI((k) => (k + 1) % ROB_FRAMES.length), 120);
    return () => clearInterval(id);
  }, []);
  const frame = ROB_FRAMES[i];
  const extrap = frame.t > 1.0;
  const upto = ROB_FRAMES.slice(0, i + 1);

  /* ---- live "run it yourself" state ---- */
  const [meta, setMeta] = useState(null);
  const [model, setModel] = useState("FNO");
  const [preset, setPreset] = useState("high_freq");
  const [modes, setModes] = useState(2);
  const [amplitude, setAmplitude] = useState(1.0);
  const [ic, setIc] = useState(null);
  const [lf, setLf] = useState(null);
  const [hist, setHist] = useState([]);
  const [summary, setSummary] = useState(null);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState(null);
  const [tab, setTab] = useState("findings");
  const wsRef = useRef(null);

  useEffect(() => { getMeta().then(setMeta).catch(() => {}); }, []);
  useEffect(() => {
    if (!meta || preset !== "custom") return;
    buildIC(modes, amplitude).then((d) => setIc(d.ic)).catch(() => {});
  }, [meta, preset, modes, amplitude]);

  function run() {
    if (wsRef.current) wsRef.current.close();
    setHist([]); setLf(null); setSummary(null); setErr(null); setRunning(true);
    const payload = preset === "custom" ? { model, ic } : { model, preset };
    wsRef.current = runRobustness(payload,
      (f) => { setLf(f); setHist((h) => [...h, f]); },
      (s) => setSummary(s),
      () => setRunning(false),
      (e) => { setErr(e); setRunning(false); });
  }

  const x = meta?.x || [];
  const muted = "text-slate-500 dark:text-slate-400";
  const lyr = lf ? [Math.min(...lf.true, ...lf.u, -1.1), Math.max(...lf.true, ...lf.u, 1.1)] : [-1.1, 1.1];

  return (
    <div className="space-y-7">
      {/* HEADER */}
      <div>
        <span className="text-xs font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">ML model analysis</span>
        <h1 className="text-3xl font-extrabold text-slate-800 dark:text-slate-100 mt-1">Robustness Analysis</h1>
        <p className="text-slate-500 dark:text-slate-400 mt-1">
          Each surrogate is trained only up to <b>t = 1</b>. We push it two ways it never saw —{" "}
          <span className="font-medium text-slate-700 dark:text-slate-200">unseen future times (extrapolation) and unseen wave shapes (out-of-distribution)</span>{" "}
          — and track the signal that catches the failure.
        </p>
      </div>

      {/* VIEW TABS — Findings | Try it live (matches Reliability & Cost) */}
      <div className="inline-flex rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-1">
        <button onClick={() => setTab("findings")}
          className={`px-4 py-1.5 rounded-lg text-sm font-semibold transition ${tab === "findings"
            ? "bg-indigo-600 text-white" : "text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"}`}>
          Findings
        </button>
        <button onClick={() => setTab("live")}
          className={`px-4 py-1.5 rounded-lg text-sm font-semibold transition ${tab === "live"
            ? "bg-indigo-600 text-white" : "text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"}`}>
          Try it live
        </button>
      </div>

      {tab === "findings" && (<>
      {/* SCOREBOARD — same 3-column card format as Reliability / Cost */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { m: "FNO", color: "#059669", badge: ["most robust", "emerald"], inW: `${S.inWindow}%`, extrap: `${S.extrap}%`, ood: `>${S.oodHighFreq}%` },
          { m: "PINN", color: "#d97706", badge: ["fails early", "amber"], inW: `${REL_INW.PINN}%`, extrap: `${S.pinnExtrap}%`, ood: "per-IC" },
          { m: "DeepONet", color: "#e11d48", badge: ["worst", "rose"], inW: `${REL_INW.DeepONet}%`, extrap: `${S.deeponetExtrap}%`, ood: `>${S.deeponetOOD}%` },
        ].map((d) => {
          const bt = {
            emerald: "text-emerald-700 bg-emerald-100 dark:bg-emerald-500/20 dark:text-emerald-300",
            amber: "text-amber-700 bg-amber-100 dark:bg-amber-500/20 dark:text-amber-300",
            rose: "text-rose-700 bg-rose-100 dark:bg-rose-500/20 dark:text-rose-300",
          }[d.badge[1]];
          return (
            <div key={d.m} className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full" style={{ background: d.color }} />
                  <span className="text-sm font-bold text-slate-800 dark:text-slate-100">{d.m}</span>
                </div>
                <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full ${bt}`}>{d.badge[0]}</span>
              </div>
              <div className="grid grid-cols-3 gap-2 mt-3">
                <div>
                  <div className="text-[10px] text-slate-400 dark:text-slate-500">in-window · held-out</div>
                  <div className="text-lg font-bold text-slate-700 dark:text-slate-200">{d.inW}</div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-400 dark:text-slate-500">extrapolation</div>
                  <div className="text-lg font-bold" style={{ color: d.color }}>{d.extrap}</div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-400 dark:text-slate-500">OOD</div>
                  <div className="text-lg font-bold text-slate-700 dark:text-slate-200">{d.ood}</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* COMPARISON CHART + SIDE PANEL (mirrors Reliability's findings layout) */}
      <div className="grid grid-cols-[1.7fr_1fr] gap-5 items-start">
        <div className="relative rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4 overflow-hidden">
          <div className="absolute -top-8 -right-8 w-40 h-40 rounded-full bg-rose-100/50 dark:bg-rose-500/10 blur-2xl" />
          <div className="relative">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">Even the most robust model drifts — FNO past t = 1</div>
                <div className="mt-1 flex gap-1.5"><span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300">REAL MODEL OUTPUT</span><span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-amber-100 dark:bg-amber-500/20 text-amber-700 dark:text-amber-300">TRUTH USED ONLY TO SCORE</span></div>
              </div>
              <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-1 rounded-full ${extrap
                ? "bg-rose-100 dark:bg-rose-500/20 text-rose-600 dark:text-rose-400 animate-pulse"
                : "bg-emerald-100 dark:bg-emerald-500/20 text-emerald-600 dark:text-emerald-400"}`}>
                {extrap ? "extrapolating" : "training window"}
              </span>
            </div>
            <DriftChart frame={frame} />
            <div className="flex items-center justify-between text-[11px] text-slate-400 dark:text-slate-500">
              <div className="flex gap-4">
                <span className="flex items-center gap-1"><span className="w-3 h-1 rounded-full bg-rose-500" /> FNO</span>
                <span className="flex items-center gap-1"><span className="w-3 h-1 rounded-full bg-slate-400" /> exact</span>
                <span className="flex items-center gap-1"><span className="w-3 h-2 rounded bg-rose-300" /> error</span>
              </div>
              <div className="flex gap-3">
                <span>t = <span className="font-bold text-slate-700 dark:text-slate-200">{frame.t.toFixed(2)}</span></span>
                <span className="text-rose-500">error <span className="font-bold">{(frame.err * 100).toFixed(0)}%</span></span>
                <span className="text-blue-500">spectral <span className="font-bold">{frame.sd.toFixed(2)}</span></span>
              </div>
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-5">
          <Card title="Extrapolation error" subtitle="past the training horizon — the whole family fails">
            <div className="space-y-3 mt-1">
              {[["FNO", S.extrap, "#059669"], ["PINN", S.pinnExtrap, "#d97706"], ["DeepONet", S.deeponetExtrap, "#e11d48"]].map(([m, v, c]) => (
                <div key={m}>
                  <div className="flex justify-between text-xs mb-1"><span className="font-semibold" style={{ color: c }}>{m}</span><span className="text-slate-500 dark:text-slate-400">{v}%</span></div>
                  <div className="h-2 rounded-full bg-slate-100 dark:bg-slate-700"><div className="h-2 rounded-full" style={{ width: `${Math.min(100, v / 0.7)}%`, background: c }} /></div>
                </div>
              ))}
            </div>
          </Card>
          <Card title="On unfamiliar inputs (OOD)" subtitle="a frequency it never trained on — sin(6πx)">
            <div className="space-y-3 mt-1">
              {[["FNO", S.oodHighFreq, "#059669"], ["DeepONet", S.deeponetOOD, "#e11d48"]].map(([m, v, c]) => (
                <div key={m}>
                  <div className="flex justify-between text-xs mb-1"><span className="font-semibold" style={{ color: c }}>{m}</span><span className="text-rose-600 dark:text-rose-400 font-medium">&gt;{v}%</span></div>
                  <div className="h-2 rounded-full bg-slate-100 dark:bg-slate-700"><div className="h-2 rounded-full" style={{ width: `${Math.min(100, v / 1.3)}%`, background: c }} /></div>
                </div>
              ))}
              <p className="text-[11px] text-slate-400 dark:text-slate-500">PINN is trained per wave, so it can&apos;t take an unfamiliar input at all.</p>
            </div>
          </Card>
        </div>
      </div>

      {/* PROGRESSIVE CURVES */}
      <div className="grid grid-cols-2 gap-4">
        <Card title="Error vs time" subtitle="The FNO drifts past the training horizon, and the error is measured">
          <LineChart
            series={[{ x: upto.map((f) => f.t), y: upto.map((f) => f.err), color: "#e11d48", width: 2.5 }]}
            xr={[0, 2]} yr={[0, Math.max(0.35, ...ROB_FRAMES.map((f) => f.err)) * 1.05]}
            vline={1.0} hline={0.1} h={175} xlabel="t" ylabel="relative L2 error" />
          <p className={`text-[11px] mt-1 ${muted}`}>dotted line = the 10% error mark · FNO crosses it at t ≈ {S.horizon}</p>
        </Card>
        <Card title="Spectral distance vs time" subtitle="The same drift, seen in frequency space">
          <LineChart
            series={[{ x: upto.map((f) => f.t), y: upto.map((f) => f.sd), color: "#2563eb", width: 2.5 }]}
            xr={[0, 2]} yr={[0, Math.max(0.35, ...ROB_FRAMES.map((f) => f.sd)) * 1.05]}
            vline={1.0} h={175} xlabel="t" ylabel="spectral distance" />
          <p className={`text-[11px] mt-1 ${muted}`}>we tried this in analysis — could the ML output&apos;s own sharpness warn of failure without the true answer? It looked promising, but it&apos;s not part of the live switch yet (Coupling page)</p>
        </Card>
      </div>

      {/* VERDICT */}
      <div className="rounded-2xl p-5 bg-gradient-to-r from-rose-50 via-white to-white dark:from-rose-500/10 dark:via-slate-800 dark:to-slate-800 border border-rose-200 dark:border-rose-500/30">
        <div className="text-sm font-semibold text-rose-800 dark:text-rose-300">Conclusion</div>
        <p className="text-sm text-slate-700 dark:text-slate-200 mt-1">
          The ML solvers are <span className="font-medium">fast but not robust</span>, they fail beyond the
          training horizon and on unfamiliar inputs, and the failure is <span className="font-medium">measured, not assumed</span>.
          This is exactly the failure the <span className="font-semibold">trust module detects</span> and my{" "}
          <span className="font-semibold">coupling module corrects</span>. We also tried this in analysis, whether the ML&apos;s own
          sharpness could warn of failure without the true answer. It looked promising, but it&apos;s not part of the live switch yet.
        </p>
      </div>
      </>)}

      {tab === "live" && (<>
      {/* RUN IT YOURSELF */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
        <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-1 flex items-center gap-2">
          <Play size={14} /> Run it yourself — live <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-emerald-100 dark:bg-emerald-500/20 text-emerald-700 dark:text-emerald-300">LIVE BACKEND</span>
        </div>
        <p className={`text-xs ${muted} mb-4`}>
          Pick a model and an input; the backend runs the real model and streams the failure as it happens.
          (PINN is trained per wave, so OOD inputs don&apos;t apply to it.)
        </p>
        {err && <div className="mb-3 text-sm text-rose-600 dark:text-rose-300 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 rounded-lg px-3 py-2">{err}</div>}
        <div className="grid grid-cols-[300px_1fr] gap-5">
          <div className="space-y-3">
            <div className="flex gap-2">
              {MODELS.map((m) => (
                <button key={m} onClick={() => setModel(m)}
                  className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium border ${model === m
                    ? "bg-indigo-600 text-white border-indigo-600" : inactiveBtn}`}>{m}</button>
              ))}
            </div>
            <p className="text-[11px] text-slate-500 dark:text-slate-400">
              PINN isn&apos;t offered here: each PINN is fitted to one fixed wave, so it can&apos;t take a new
              unfamiliar input to fail on. Its extrapolation number is on the scoreboard above.
            </p>
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
              <div className="space-y-3 text-sm">
                <label className={`block ${muted}`}>
                  modes: {modes} {modes > 4 && <span className="text-amber-600 dark:text-amber-400">(beyond trained band)</span>}
                  <input type="range" min="1" max="8" value={modes} onChange={(e) => setModes(+e.target.value)} className="w-full" />
                </label>
                <label className={`block ${muted}`}>
                  amplitude: {amplitude.toFixed(2)}
                  <input type="range" min="0.2" max="1.5" step="0.05" value={amplitude} onChange={(e) => setAmplitude(+e.target.value)} className="w-full" />
                </label>
              </div>
            )}
            <button onClick={run} disabled={running || (preset === "custom" && !ic)}
              className="w-full px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium text-sm transition">
              {running ? "Running…" : "Run the model"}
            </button>
            {summary && (
              <div className="grid grid-cols-2 gap-2">
                <Stat label="in-window (t ≤ 1)" value={`${(summary.in_window_err * 100).toFixed(1)}%`} tone="green" />
                <Stat label="extrapolating (t > 1)" value={`${(summary.extrap_err * 100).toFixed(1)}%`} tone="red" />
                <Stat label="reliable horizon" value={`t = ${summary.reliable_horizon.toFixed(2)}`} tone="indigo" />
                <Stat label="verdict" value={summary.extrap_err < 0.1 ? "survives" : "fails"} />
              </div>
            )}
          </div>
          <div className="space-y-3">
            <Card title={lf ? `t = ${lf.t.toFixed(2)}${lf.t > 1 ? " — extrapolating" : " — in training window"}` : "run to start"}>
              <LineChart
                series={[
                  { x, y: lf ? lf.true : [], color: "#94a3b8", dashed: true },
                  { x, y: lf ? lf.u : [], color: "#e11d48", width: 2.5 },
                ]}
                xr={[-1, 1]} yr={lyr} h={190} xlabel="x" ylabel="u(x, t)" />
            </Card>
            <Card title="error (red) and spectral distance (blue)">
              <LineChart
                series={[
                  { x: hist.map((f) => f.t), y: hist.map((f) => f.err), color: "#e11d48", width: 2 },
                  { x: hist.map((f) => f.t), y: hist.map((f) => f.sd), color: "#2563eb", width: 2 },
                ]}
                xr={[0, 2]} yr={[0, Math.max(0.35, ...hist.map((f) => f.err), ...hist.map((f) => f.sd))]}
                vline={1.0} hline={0.1} h={160} xlabel="t" />
            </Card>
          </div>
        </div>
      </div>
      </>)}
    </div>
  );
}
