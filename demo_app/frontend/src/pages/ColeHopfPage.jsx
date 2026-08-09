import { useEffect, useState } from "react";
import { CH_XS, CH_FRAMES, CH_STATS as S } from "../colehopfData.js";
import { eToSup } from "../components/ui.jsx";
import {
  Wand2, Thermometer, Undo2, ShieldCheck, Scale, Timer, Target,
  HelpCircle, ArrowRight,CheckCircle2
} from "lucide-react";

/* ---- auto-looping exact solution steepening into the viscous shock ---- */
function ShockChart({ frame }) {
  const W = 580, H = 300, padX = 18, padT = 18, padB = 28;
  const sx = (x) => padX + ((x + 1) / 2) * (W - 2 * padX);
  const sy = (v) => H - padB - ((v + 1.4) / 2.8) * (H - padT - padB);
  const path = (arr) =>
    arr.map((v, i) => `${i ? "L" : "M"}${sx(CH_XS[i]).toFixed(1)} ${sy(v).toFixed(1)}`).join(" ");
  let area = `M${sx(CH_XS[0]).toFixed(1)} ${sy(0).toFixed(1)}`;
  CH_XS.forEach((x, i) => (area += `L${sx(x).toFixed(1)} ${sy(frame.u[i]).toFixed(1)}`));
  area += `L${sx(CH_XS[CH_XS.length - 1]).toFixed(1)} ${sy(0).toFixed(1)}Z`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {[-1, -0.5, 0, 0.5, 1].map((v) => (
        <line key={v} x1={sx(-1)} x2={sx(1)} y1={sy(v)} y2={sy(v)}
          stroke="var(--chart-grid)" strokeWidth="1" />
      ))}
      <path d={area} fill="#4f46e5" fillOpacity="0.12" />
      <path d={path(frame.u)} fill="none" stroke="#4f46e5" strokeWidth="3" strokeLinecap="round" />
      <text x={sx(-1) + 4} y={padT + 4} fontSize="11" fill="#818cf8" fontWeight="700">
        no step-by-step build-up of error
      </text>
    </svg>
  );
}

function Metric({ icon: Icon, tone, tag, value, label }) {
  const c = {
    rose: ["bg-rose-50 dark:bg-rose-500/10 border-rose-100 dark:border-rose-500/20",
      "bg-rose-100 dark:bg-rose-500/20 text-rose-600 dark:text-rose-400",
      "text-rose-600 dark:text-rose-400"],
    indigo: ["bg-indigo-50 dark:bg-indigo-500/10 border-indigo-100 dark:border-indigo-500/20",
      "bg-indigo-100 dark:bg-indigo-500/20 text-indigo-600 dark:text-indigo-400",
      "text-indigo-600 dark:text-indigo-400"],
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

export default function ColeHopfPage() {
  const [i, setI] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setI((k) => (k + 1) % CH_FRAMES.length), 120);
    return () => clearInterval(id);
  }, []);
  const frame = CH_FRAMES[i];
  const shockPct = Math.min(100, Math.round((frame.sharp / S.sharpMax) * 100));

  return (
    <div className="space-y-7">
      {/* HEADER */}
      <div>
        <span className="text-xs font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">Numerical solvers</span>
        <h1 className="text-3xl font-extrabold text-slate-800 dark:text-slate-100 mt-1">Cole–Hopf, the exact reference</h1>
        <span className="inline-block mt-2 text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-amber-100 dark:bg-amber-500/20 text-amber-700 dark:text-amber-300">
          Evaluation reference, not the runtime corrector
        </span>
        <p className="text-slate-500 dark:text-slate-400 mt-1">
          Every number in this project is measured against one reference solution. So we asked:{" "}
          <span className="font-medium text-slate-700 dark:text-slate-200">can we build one answer we fully trust, and show why we can trust it?</span>{" "}
          Here is the method, the test, and the answer.
        </p>
      </div>

      {/* HOW IT WORKS */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
        <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-4">
          How Cole–Hopf works, three steps
        </div>
        <div className="flex items-stretch gap-2 flex-wrap">
          <Step n="1" icon={Wand2} tone="indigo" title="Transform" sub="the nonlinear Burgers equation becomes the linear heat equation" />
          <ArrowRight size={18} className="self-center shrink-0 text-slate-300 dark:text-slate-600" />
          <Step n="2" icon={Thermometer} tone="violet" title="Solve heat exactly" sub="one heat-kernel convolution, no time marching" />
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
            <span className="flex items-center gap-1.5 text-slate-500 dark:text-slate-400">
              <span className="w-2.5 h-2.5 rounded bg-rose-500" />
              <span className="font-semibold text-rose-600 dark:text-rose-400">nonlinear</span>, hard, shock-forming
            </span>
            <span className="flex items-center gap-1.5 text-slate-500 dark:text-slate-400">
              <span className="w-2.5 h-2.5 rounded bg-teal-500" />
              <span className="font-semibold text-teal-600 dark:text-teal-400">linear heat</span>, solved exactly
            </span>
          </div>
        </div>
      </div>

      {/* COMPARISON (auto-loop) */}
      <div className="grid grid-cols-[1fr_230px] gap-5 items-stretch">
        <div className="relative rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4 overflow-hidden">
          <div className="absolute -top-8 -right-8 w-40 h-40 rounded-full bg-indigo-100/50 dark:bg-indigo-500/10 blur-2xl" />
          <div className="relative">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">The exact solution steepening into the viscous shock</div>
                <div className="text-xs text-slate-500 dark:text-slate-400">
                  a real held-out wave, the trajectory every model trains on and is measured against
                </div>
              </div>
              <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-1 rounded-full ${shockPct > 60
                ? "bg-indigo-100 dark:bg-indigo-500/20 text-indigo-600 dark:text-indigo-400 animate-pulse"
                : "bg-slate-100 dark:bg-slate-700 text-slate-500 dark:text-slate-400"}`}>
                {shockPct > 60 ? "shock formed" : "steepening"}
              </span>
            </div>
            <ShockChart frame={frame} />
            <div className="flex items-center justify-center gap-5 text-[11px] text-slate-400 dark:text-slate-500">
              <span className="flex items-center gap-1"><span className="w-3 h-1 rounded-full bg-indigo-500" /> Cole–Hopf (exact)</span>
              <span>the reference every solver is measured against</span>
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
              <div className="text-xs text-emerald-600 dark:text-emerald-400">time-stepping error</div>
              <div className="text-5xl font-extrabold text-emerald-600 dark:text-emerald-400 mt-1">0</div>
              <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">exact in time, one-shot heat kernel, nothing accumulates</div>
            </div>
          </div>
        </div>
      </div>

      {/* THE EVIDENCE */}
      <div>
        <h2 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3 flex items-center gap-2">
          <HelpCircle size={15} /> The evidence · why this is the source of truth
        </h2>
        <div className="grid grid-cols-4 gap-4">
          <Metric icon={Timer} tone="indigo" tag="Exactness" value="0"
            label="time-stepping error, the heat kernel gives the whole trajectory in one shot" />
          <Metric icon={ShieldCheck} tone="emerald" tag="Cross-check" value={`${S.agreePct}%`}
            label={`agreement with the independent spectral solver (max gap ${eToSup(S.maxDis)})`} />
          <Metric icon={Scale} tone="indigo" tag="Vs FDM" value={`${S.timesWorse}×`}
            label={`FDM is ${S.fdmVsExact}% off this reference, a cheap baseline, not a truth source`} />
          <Metric icon={Target} tone="indigo" tag="Role" value="Reference"
            label="generates the reference dataset and scores every model; the runtime hand-off continues with the pseudo-spectral solver" />
        </div>
      </div>

      {/* VERDICT */}
      <div className="rounded-2xl p-5 bg-gradient-to-r from-emerald-50 via-white to-white dark:from-emerald-500/10 dark:via-slate-800 dark:to-slate-800 border border-emerald-200 dark:border-emerald-500/30">
        <div className="text-sm font-semibold text-emerald-800 dark:text-emerald-300 flex items-center gap-2"><CheckCircle2 size={16} /> Answer</div>
          <p className="text-sm text-slate-700 dark:text-slate-200 mt-1">
          Cole–Hopf is the <span className="font-medium">the ground truth reference</span>. It provides the trusted solution used to measure the error of the ML, numerical and hybrid methods. Its implementation was cross-checked against the spectral solver for consistency.
        </p>
      </div>
    </div>
  );
}
