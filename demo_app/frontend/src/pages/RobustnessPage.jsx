import { useEffect, useRef, useState } from "react";
import { Card, Stat, Banner } from "../components/ui.jsx";
import { LineChart } from "../components/Charts.jsx";
import { getMeta, buildIC, runRobustness } from "../api.js";
import { ROB_XS, ROB_FRAMES, ROB_STATS as S } from "../robustnessData.js";
import {
  Waves, Clock, Activity, TrendingUp, ShieldAlert, Scale,
  HelpCircle, ArrowRight, Play,
} from "lucide-react";

/* ---- auto-looping FNO vs exact with the error shaded (real committed
   predictions, held-out wave — no exaggeration, the gap is the real error) ---- */
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
        <h1 className="text-3xl font-extrabold text-slate-800 dark:text-slate-100 mt-1">Robustness — where the models break</h1>
        <p className="text-slate-500 dark:text-slate-400 mt-1 max-w-2xl">
          The ML solvers are fast — but{" "}
          <span className="font-medium text-slate-700 dark:text-slate-200">can they survive the future, and the unfamiliar?</span>{" "}
          Here is the stress test, the failure, and the signal that tracks it.
        </p>
      </div>

      {/* HOW IT WORKS */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
        <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-4">
          The stress test — three steps
        </div>
        <div className="flex items-stretch gap-2 flex-wrap">
          <Step n="1" icon={Waves} tone="indigo" title="Feed it a wave" sub="familiar — or one it never trained on" />
          <ArrowRight size={18} className="self-center shrink-0 text-slate-300 dark:text-slate-600" />
          <Step n="2" icon={Clock} tone="violet" title="Predict the future" sub="query past the training horizon t = 1" />
          <ArrowRight size={18} className="self-center shrink-0 text-slate-300 dark:text-slate-600" />
          <Step n="3" icon={Activity} tone="fuchsia" title="Track the failure" sub="my spectral signal watches frequency space" />
        </div>
      </div>

      {/* HERO ANIMATION (real committed FNO predictions) */}
      <div className="grid grid-cols-[1fr_230px] gap-5 items-stretch">
        <div className="relative rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4 overflow-hidden">
          <div className="absolute -top-8 -right-8 w-40 h-40 rounded-full bg-rose-100/50 dark:bg-rose-500/10 blur-2xl" />
          <div className="relative">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">The FNO drifts off the truth after t = 1</div>
                <div className="mt-1 flex gap-1.5"><span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300">COMMITTED RESULT</span><span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-amber-100 dark:bg-amber-500/20 text-amber-700 dark:text-amber-300">EVALUATION-ONLY TRUTH</span></div>
                <div className="text-xs text-slate-500 dark:text-slate-400">
                  a real held-out wave, the committed FNO prediction — the shaded gap is the actual error
                </div>
              </div>
              <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-1 rounded-full ${extrap
                ? "bg-rose-100 dark:bg-rose-500/20 text-rose-600 dark:text-rose-400 animate-pulse"
                : "bg-emerald-100 dark:bg-emerald-500/20 text-emerald-600 dark:text-emerald-400"}`}>
                {extrap ? "extrapolating" : "training window"}
              </span>
            </div>
            <DriftChart frame={frame} />
            <div className="flex items-center justify-center gap-5 text-[11px] text-slate-400 dark:text-slate-500">
              <span className="flex items-center gap-1"><span className="w-3 h-1 rounded-full bg-rose-500" /> FNO</span>
              <span className="flex items-center gap-1"><span className="w-3 h-1 rounded-full bg-slate-400" /> exact (Cole–Hopf)</span>
              <span className="flex items-center gap-1"><span className="w-3 h-2 rounded bg-rose-300" /> error</span>
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4 text-center">
            <div className="text-xs text-slate-500 dark:text-slate-400">time</div>
            <div className="text-2xl font-bold text-slate-800 dark:text-slate-100">t = {frame.t.toFixed(2)}</div>
          </div>
          <div className="rounded-2xl border border-rose-200 dark:border-rose-500/30 bg-rose-50 dark:bg-rose-500/10 p-4 text-center">
            <div className="text-xs text-rose-500 dark:text-rose-400">error vs exact</div>
            <div className="text-4xl font-extrabold text-rose-600 dark:text-rose-400 mt-1">{(frame.err * 100).toFixed(0)}%</div>
            <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">grows sharply once t &gt; 1</div>
          </div>
          <div className="flex-1 rounded-2xl border border-blue-200 dark:border-blue-500/30 bg-blue-50 dark:bg-blue-500/10 p-4 text-center grid place-items-center">
            <div>
              <div className="text-xs text-blue-500 dark:text-blue-400">spectral distance</div>
              <div className="text-4xl font-extrabold text-blue-600 dark:text-blue-400 mt-1">{frame.sd.toFixed(2)}</div>
              <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">offline diagnostic vs Cole–Hopf truth — the frequency-space view of the failure</div>
            </div>
          </div>
        </div>
      </div>

      {/* PROGRESSIVE CURVES */}
      <div className="grid grid-cols-2 gap-4">
        <Card title="Error vs time" subtitle="drawn live with the animation — the red vertical line is the training horizon">
          <LineChart
            series={[{ x: upto.map((f) => f.t), y: upto.map((f) => f.err), color: "#e11d48", width: 2.5 }]}
            xr={[0, 2]} yr={[0, Math.max(0.35, ...ROB_FRAMES.map((f) => f.err)) * 1.05]}
            vline={1.0} hline={0.1} h={175} xlabel="t" ylabel="relative L2 error" />
          <p className={`text-[11px] mt-1 ${muted}`}>dotted = 10% failure band · FNO crosses it at t ≈ {S.horizon}</p>
        </Card>
        <Card title="Spectral distance vs time" subtitle="the frequency-space view of the same failure">
          <LineChart
            series={[{ x: upto.map((f) => f.t), y: upto.map((f) => f.sd), color: "#2563eb", width: 2.5 }]}
            xr={[0, 2]} yr={[0, Math.max(0.35, ...ROB_FRAMES.map((f) => f.sd)) * 1.05]}
            vline={1.0} h={175} xlabel="t" ylabel="spectral distance" />
          <p className={`text-[11px] mt-1 ${muted}`}>a related high-wavenumber state feature was evaluated offline as a candidate reference-free diagnostic; it is not currently enforced at runtime (Coupling page)</p>
        </Card>
      </div>

      {/* FINDINGS */}
      <div>
        <h2 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3 flex items-center gap-2">
          <HelpCircle size={15} /> The findings · measured on held-out waves
        </h2>
        <div className="grid grid-cols-4 gap-4">
          <Metric icon={ShieldAlert} tone="emerald" tag="in-window" value={`${S.inWindow}%`}
            label="accurate where it trained (t ≤ 1) — the speed is real" />
          <Metric icon={TrendingUp} tone="rose" tag="future" value={`${S.extrap}%`}
            label={`extrapolation error — ~25× worse; reliable only to t ≈ ${S.horizon}`} />
          <Metric icon={Waves} tone="rose" tag="unfamiliar" value={`>${S.oodHighFreq}%`}
            label="on sin(6πx), beyond the trained band, the output is meaningless" />
          <Metric icon={Scale} tone="amber" tag="all models" value={`${S.deeponetExtrap}%`}
            label={`DeepONet extrapolation (PINN ${S.pinnExtrap}%) — failure is the family's, not one model's`} />
        </div>
      </div>

      {/* VERDICT */}
      <div className="rounded-2xl p-5 bg-gradient-to-r from-rose-50 via-white to-white dark:from-rose-500/10 dark:via-slate-800 dark:to-slate-800 border border-rose-200 dark:border-rose-500/30">
        <div className="text-sm font-semibold text-rose-800 dark:text-rose-300">Answer</div>
        <p className="text-sm text-slate-700 dark:text-slate-200 mt-1">
          The ML solvers are <span className="font-medium">fast but not robust</span> — they fail beyond the
          training horizon and on unfamiliar inputs, and the failure is <span className="font-medium">measured, not assumed</span>.
          This is exactly the failure the <span className="font-semibold">trust module detects</span> and my{" "}
          <span className="font-semibold">coupling module corrects</span>. A related high-wavenumber state feature was
          evaluated offline as a candidate reference-free diagnostic; it is not currently enforced at runtime.
        </p>
      </div>

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
                  <input type="range" min="1" max="4" value={modes} onChange={(e) => setModes(+e.target.value)} className="w-full" />
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
    </div>
  );
}
