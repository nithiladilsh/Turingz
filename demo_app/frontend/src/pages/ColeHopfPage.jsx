import { useEffect, useRef, useState } from "react";
import { Card, Stat, Banner } from "../components/ui.jsx";
import { LineChart } from "../components/Charts.jsx";
import { getMeta, buildIC, runColeHopf } from "../api.js";
import { CH_XS, CH_FRAMES, CH_STATS as S } from "../colehopfData.js";
import {
  Wand2, Thermometer, Undo2, ArrowRight, Timer, ShieldCheck, Database,
  Anchor, HelpCircle, CheckCircle2, FlaskConical, Play,
} from "lucide-react";

/* ---- auto-looping exact solution steepening into the viscous shock ---- */
function ShockChart({ frame }) {
  const W = 580, H = 300, padX = 18, padT = 22, padB = 28;
  const clamp = (v) => Math.max(-1.9, Math.min(1.9, v));
  const sx = (x) => padX + ((x + 1) / 2) * (W - 2 * padX);
  const sy = (v) => H - padB - ((clamp(v) + 2) / 4) * (H - padT - padB);
  const path = (arr) =>
    arr.map((v, i) => `${i ? "L" : "M"}${sx(CH_XS[i]).toFixed(1)} ${sy(v).toFixed(1)}`).join(" ");
  let area = `M${sx(CH_XS[0]).toFixed(1)} ${sy(0).toFixed(1)}`;
  CH_XS.forEach((x, i) => (area += `L${sx(x).toFixed(1)} ${sy(frame.u[i]).toFixed(1)}`));
  area += `L${sx(CH_XS[CH_XS.length - 1]).toFixed(1)} ${sy(0).toFixed(1)}Z`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {[-1.5, -0.75, 0, 0.75, 1.5].map((v) => (
        <line key={v} x1={sx(-1)} x2={sx(1)} y1={sy(v)} y2={sy(v)} stroke="var(--chart-grid)" strokeWidth="1" />
      ))}
      <path d={area} fill="#4f46e5" fillOpacity="0.10" />
      <path d={path(frame.u)} fill="none" stroke="#4f46e5" strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />
      <text x={sx(-1) + 4} y={padT} fontSize="11" fill="#059669" fontWeight="700">exact — no time-stepping error at all</text>
    </svg>
  );
}

function Metric({ icon: Icon, tone, tag, value, label }) {
  const c = {
    emerald: ["bg-emerald-50 dark:bg-emerald-500/10 border-emerald-100 dark:border-emerald-500/20", "bg-emerald-100 dark:bg-emerald-500/20 text-emerald-600 dark:text-emerald-400", "text-emerald-600 dark:text-emerald-400"],
    indigo: ["bg-indigo-50 dark:bg-indigo-500/10 border-indigo-100 dark:border-indigo-500/20", "bg-indigo-100 dark:bg-indigo-500/20 text-indigo-600 dark:text-indigo-400", "text-indigo-600 dark:text-indigo-400"],
    violet: ["bg-violet-50 dark:bg-violet-500/10 border-violet-100 dark:border-violet-500/20", "bg-violet-100 dark:bg-violet-500/20 text-violet-600 dark:text-violet-400", "text-violet-600 dark:text-violet-400"],
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

export default function ColeHopfPage() {
  /* ---- auto-loop ---- */
  const [i, setI] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setI((k) => (k + 1) % CH_FRAMES.length), 160);
    return () => clearInterval(id);
  }, []);
  const frame = CH_FRAMES[i];

  /* ---- live cross-verification state ---- */
  const [meta, setMeta] = useState(null);
  const [modes, setModes] = useState(4);
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
    if (!meta) return;
    buildIC(modes, amplitude).then((d) => setIc(d.ic)).catch(() => {});
  }, [meta, modes, amplitude]);

  function run() {
    if (wsRef.current) wsRef.current.close();
    setHist([]); setLf(null); setSummary(null); setErr(null); setRunning(true);
    wsRef.current = runColeHopf({ ic },
      (f) => { setLf(f); setHist((h) => [...h, f]); },
      (s) => setSummary(s),
      () => setRunning(false),
      (e) => { setErr(e); setRunning(false); });
  }

  const x = meta?.x || [];
  const muted = "text-slate-500 dark:text-slate-400";
  const lyr = lf ? [Math.min(...lf.ch, -1.1), Math.max(...lf.ch, 1.1)] : [-1.1, 1.1];

  return (
    <div className="space-y-7">
      {/* ============ TOP — same vibe as FDM / Spectral ============ */}

      {/* HEADER */}
      <div>
        <span className="text-xs font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">Numerical solvers</span>
        <h1 className="text-3xl font-extrabold text-slate-800 dark:text-slate-100 mt-1">Cole–Hopf Transform</h1>
        <p className="text-slate-500 dark:text-slate-400 mt-1 max-w-2xl">
          Every result in this project is scored against one answer. So we asked —{" "}
          <span className="font-medium text-slate-700 dark:text-slate-200">can we manufacture a perfect answer, and prove it&apos;s perfect?</span>{" "}
          Here is the trick, the solution, and the proof.
        </p>
      </div>

      {/* HOW IT WORKS */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
        <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-4">The Cole–Hopf trick — three steps</div>
        <div className="flex items-stretch gap-2 flex-wrap">
          <Step n="1" icon={Wand2} tone="indigo" title="Transform" sub="nonlinear Burgers becomes the linear heat equation" />
          <ArrowRight size={18} className="self-center shrink-0 text-slate-300 dark:text-slate-600" />
          <Step n="2" icon={Thermometer} tone="violet" title="Solve heat exactly" sub="one heat-kernel convolution — no time marching" />
          <ArrowRight size={18} className="self-center shrink-0 text-slate-300 dark:text-slate-600" />
          <Step n="3" icon={Undo2} tone="fuchsia" title="Transform back" sub="recover the Burgers solution, exact in time" />
        </div>
        <div className="mt-5 rounded-xl bg-slate-50 dark:bg-slate-700/40 p-4">
          <div className="text-center font-mono text-[15px] text-slate-700 dark:text-slate-200 leading-relaxed">
            <span className="text-indigo-600 dark:text-indigo-400 font-semibold"> u = −2ν φₓ/φ </span>
            turns
            <span className="text-rose-600 dark:text-rose-400 font-semibold"> uₜ + u uₓ = ν uₓₓ </span>
            into
            <span className="text-teal-600 dark:text-teal-400 font-semibold"> φₜ = ν φₓₓ </span>
          </div>
          <div className="flex items-center justify-center gap-6 mt-3 text-xs">
            <span className="flex items-center gap-1.5 text-slate-500 dark:text-slate-400"><span className="w-2.5 h-2.5 rounded bg-rose-500" /> <span className="font-semibold text-rose-600 dark:text-rose-400">nonlinear</span> — hard, shock-forming</span>
            <span className="flex items-center gap-1.5 text-slate-500 dark:text-slate-400"><span className="w-2.5 h-2.5 rounded bg-teal-500" /> <span className="font-semibold text-teal-600 dark:text-teal-400">linear heat</span> — solved exactly</span>
          </div>
        </div>
      </div>

      {/* COMPARISON (auto-loop) */}
      <div className="grid grid-cols-[1fr_230px] gap-5 items-stretch">
        <div className="relative rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4 overflow-hidden">
          <div className="absolute -top-8 -right-8 w-40 h-40 rounded-full bg-indigo-100/50 dark:bg-indigo-500/10 blur-2xl" />
          <div className="relative">
            <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">The exact answer, steepening into the viscous shock</div>
            <div className="text-xs text-slate-500 dark:text-slate-400">this trajectory is what every model trains on and is graded against</div>
            <ShockChart frame={frame} />
            <div className="flex items-center justify-center gap-5 text-[11px] text-slate-400 dark:text-slate-500">
              <span className="flex items-center gap-1"><span className="w-3 h-1 rounded-full bg-indigo-500" /> Cole–Hopf (exact)</span>
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4 text-center">
            <div className="text-xs text-slate-500 dark:text-slate-400">time</div>
            <div className="text-2xl font-bold text-slate-800 dark:text-slate-100">t = {frame.t.toFixed(2)}</div>
          </div>
          <div className="flex-1 rounded-2xl border border-emerald-200 dark:border-emerald-500/30 bg-emerald-50 dark:bg-emerald-500/10 p-4 text-center grid place-items-center">
            <div>
              <div className="text-xs text-emerald-600 dark:text-emerald-400">two independent solvers disagree by</div>
              <div className="text-4xl font-extrabold text-emerald-600 dark:text-emerald-400 mt-1">{frame.dis.toExponential(0)}</div>
              <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">effectively zero — proven, not assumed</div>
            </div>
          </div>
        </div>
      </div>

      {/* THE PROOF */}
      <div>
        <h2 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3 flex items-center gap-2"><HelpCircle size={15} /> The proof · why this is the source of truth</h2>
        <div className="grid grid-cols-4 gap-4">
          <Metric icon={Timer} tone="indigo" tag="Exactness" value="0" label="time-stepping error — the heat kernel gives the whole trajectory in one shot" />
          <Metric icon={ShieldCheck} tone="emerald" tag="Cross-check" value={`${S.agreePct}%`} label={`agreement with the independent spectral solver (max disagreement ${S.maxDis.toExponential(0)})`} />
          <Metric icon={Database} tone="violet" tag="We use it for" value="1000 waves" label="generated the training dataset and the held-out truth every model error is scored against" />
          <Metric icon={Anchor} tone="indigo" tag="In the hybrid" value="Anchor" label="the trustworthy state the coupling module re-anchors the drifting ML trajectory to" />
        </div>
      </div>

      {/* VERDICT */}
      <div className="rounded-2xl p-5 bg-gradient-to-r from-emerald-50 via-white to-white dark:from-emerald-500/10 dark:via-slate-800 dark:to-slate-800 border border-emerald-200 dark:border-emerald-500/30">
        <div className="text-sm font-semibold text-emerald-800 dark:text-emerald-300 flex items-center gap-2"><CheckCircle2 size={16} /> Answer</div>
        <p className="text-sm text-slate-700 dark:text-slate-200 mt-1">
          Yes — Cole–Hopf gives an answer that is <span className="font-medium">exact in time and cross-verified by an
          independent method</span>. It generated the dataset, it grades every model, and the hybrid re-anchors to it.
          If this reference were wrong, everything downstream would be —{" "}
          <span className="font-semibold">which is why it is proven, not assumed.</span>
        </p>
      </div>

      {/* ============ SCROLL DOWN — what only Cole–Hopf can do ============ */}

      <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
        <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-1 flex items-center gap-2">
          <FlaskConical size={14} /> The special part — cross-verify it yourself, live
        </div>
        <p className={`text-xs ${muted} mb-4`}>
          The proof above isn&apos;t a stored number. Build <span className="font-medium">any</span> wave and the backend solves it
          exactly with Cole–Hopf <span className="font-medium">and</span> independently with the spectral solver, on the spot —
          two different methods, one answer, for whatever input an examiner picks.
        </p>
        {err && <div className="mb-3 text-sm text-rose-600 dark:text-rose-300 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 rounded-lg px-3 py-2">{err}</div>}
        <div className="grid grid-cols-[300px_1fr] gap-5">
          <div className="space-y-3">
            <div className="space-y-3 text-sm">
              <label className={`block ${muted}`}>
                modes: {modes}
                <input type="range" min="1" max="8" value={modes} onChange={(e) => setModes(+e.target.value)} className="w-full" />
              </label>
              <label className={`block ${muted}`}>
                amplitude: {amplitude.toFixed(2)}
                <input type="range" min="0.2" max="1.5" step="0.05" value={amplitude} onChange={(e) => setAmplitude(+e.target.value)} className="w-full" />
              </label>
            </div>
            <button onClick={run} disabled={running || !ic}
              className="w-full px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium text-sm transition flex items-center justify-center gap-2">
              <Play size={15} /> {running ? "Solving…" : "Solve exactly + cross-verify"}
            </button>
            {summary && (
              <div className="space-y-2">
                <Stat label="mean disagreement (this wave)" value={summary.mean_disagreement.toExponential(1)} tone="green" />
                <Stat label="max disagreement" value={summary.max_disagreement.toExponential(1)} tone="indigo" />
                <Banner ok={summary.max_disagreement < 1e-3}
                  text="Two independent methods agree — trustworthy for this wave too." />
              </div>
            )}
          </div>
          <div className="space-y-3">
            <Card title={lf ? `t = ${lf.t.toFixed(2)}` : "run to start"}>
              <LineChart
                series={[
                  { x, y: lf ? lf.ch : [], color: "#4f46e5", width: 2.5 },
                  { x, y: lf ? lf.sp : [], color: "#e11d48", dashed: true },
                ]}
                xr={[-1, 1]} yr={lyr} h={185} xlabel="x" ylabel="u(x, t)" />
              <div className="flex gap-4 mt-1 text-xs">
                <span className="text-indigo-500 font-medium">— Cole–Hopf (exact)</span>
                <span className="text-rose-500">— pseudo-spectral (independent, dashed)</span>
              </div>
            </Card>
            <Card title="disagreement over time — stays at the discretisation floor">
              <LineChart
                series={[{ x: hist.map((f) => f.t), y: hist.map((f) => f.disagreement), color: "#059669", width: 2 }]}
                xr={[0, 2]} yr={[0, Math.max(1e-4, ...hist.map((f) => f.disagreement)) * 1.2]}
                h={140} xlabel="t" />
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
