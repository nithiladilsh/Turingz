import { useEffect, useState } from "react";
import { SPECTRAL_XS as XS, SPECTRAL_FRAMES as FR, SPECTRAL_STATS as S } from "../spectralData.js";
import { Waves, Zap, Repeat, ArrowRight, Target, Scale, Gauge, Cpu, HelpCircle, CheckCircle2 } from "lucide-react";

function MatchChart({ frame }) {
  const W = 580, H = 300, padX = 18, padT = 22, padB = 28;
  const clamp = (v) => Math.max(-1.9, Math.min(1.9, v));
  const sx = (x) => padX + ((x + 1) / 2) * (W - 2 * padX);
  const sy = (v) => H - padB - ((clamp(v) + 2) / 4) * (H - padT - padB);
  const path = (arr) => arr.map((v, i) => `${i ? "L" : "M"}${sx(XS[i]).toFixed(1)} ${sy(v).toFixed(1)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {[-1.5, -0.75, 0, 0.75, 1.5].map((v) => (
        <line key={v} x1={sx(-1)} x2={sx(1)} y1={sy(v)} y2={sy(v)} stroke="var(--chart-grid)" strokeWidth="1" />
      ))}
      <path d={path(frame.exact)} fill="none" stroke="#94a3b8" strokeWidth="6" strokeLinejoin="round" strokeLinecap="round" opacity="0.55" />
      <path d={path(frame.spectral)} fill="none" stroke="#4f46e5" strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />
      <text x={sx(-1) + 4} y={padT} fontSize="11" fill="#059669" fontWeight="700">spectral sits on the exact answer</text>
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

export default function SpectralPage() {
  const [i, setI] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setI((k) => (k + 1) % FR.length), 160);
    return () => clearInterval(id);
  }, []);
  const frame = FR[i];
  const errShown = (frame.err * 100).toFixed(3);
  // use the precise measured ratio (fdm_evaluation_values.json's fdm_times_worse) rather than
  // recomputing from the two already-rounded display percentages, which would compound
  // rounding error (16.3/0.011 ~= 1482, not the real ~1513x).
  const vsFdm = S.vsFdm;

  return (
    <div className="space-y-7">
      {/* HEADER */}
      <div>
        <span className="text-xs font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">Numerical solvers</span>
        <h1 className="text-3xl font-extrabold text-slate-800 dark:text-slate-100 mt-1">Pseudo-Spectral Method</h1>
        <p className="text-slate-500 dark:text-slate-400 mt-1 max-w-2xl">
          We built the spectral solver and asked two questions —{" "}
          <span className="font-medium text-slate-700 dark:text-slate-200">is it accurate enough to trust, and can it be the hybrid's correction engine?</span>{" "}
          Here is the method, the test, and the answer.
        </p>
      </div>

      {/* HOW IT WORKS */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
        <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-4">How the spectral method works — three steps</div>
        <div className="flex items-stretch gap-2 flex-wrap">
          <Step n="1" icon={Waves} tone="indigo" title="To frequencies" sub="FFT the wave into sine components" />
          <ArrowRight size={18} className="self-center shrink-0 text-slate-300 dark:text-slate-600" />
          <Step n="2" icon={Zap} tone="violet" title="Advance exactly" sub="diffusion solved exactly per frequency (ETDRK4), 2/3 de-aliased" />
          <ArrowRight size={18} className="self-center shrink-0 text-slate-300 dark:text-slate-600" />
          <Step n="3" icon={Repeat} tone="fuchsia" title="Back to space" sub="inverse FFT to the physical wave" />
        </div>
        <div className="mt-5 rounded-xl bg-slate-50 dark:bg-slate-700/40 p-4">
          <div className="text-center font-mono text-[15px] text-slate-700 dark:text-slate-200 leading-relaxed">
            ûₖⁿ⁺¹ =
            <span className="text-indigo-600 dark:text-indigo-400 font-semibold"> e^(−ν k² Δt) ûₖ </span>
            +
            <span className="text-teal-600 dark:text-teal-400 font-semibold"> (exponential-integrator nonlinear term) </span>
          </div>
          <div className="flex items-center justify-center gap-6 mt-3 text-xs">
            <span className="flex items-center gap-1.5 text-slate-500 dark:text-slate-400"><span className="w-2.5 h-2.5 rounded bg-indigo-500" /> <span className="font-semibold text-indigo-600 dark:text-indigo-400">exact diffusion</span> — no numerical blurring</span>
            <span className="flex items-center gap-1.5 text-slate-500 dark:text-slate-400"><span className="w-2.5 h-2.5 rounded bg-teal-500" /> <span className="font-semibold text-teal-600 dark:text-teal-400">de-aliasing</span> — keeps sharp shocks stable</span>
          </div>
        </div>
      </div>

      {/* COMPARISON (auto-loop) */}
      <div className="grid grid-cols-[1fr_230px] gap-5 items-stretch">
        <div className="relative rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4 overflow-hidden">
          <div className="absolute -top-8 -right-8 w-40 h-40 rounded-full bg-emerald-100/50 dark:bg-emerald-500/10 blur-2xl" />
          <div className="relative">
            <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">Spectral keeps the shock sharp — and exact</div>
            <div className="text-xs text-slate-500 dark:text-slate-400">shown at true scale: the spectral curve lies on top of the exact answer, no visible gap</div>
            <MatchChart frame={frame} />
            <div className="flex items-center justify-center gap-5 text-[11px] text-slate-400 dark:text-slate-500">
              <span className="flex items-center gap-1"><span className="w-3 h-1 rounded-full bg-indigo-500" /> spectral</span>
              <span className="flex items-center gap-1"><span className="w-3 h-1.5 rounded-full bg-slate-400" /> exact (Cole-Hopf)</span>
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
              <div className="text-xs text-emerald-600 dark:text-emerald-400">error vs exact</div>
              <div className="text-5xl font-extrabold text-emerald-600 dark:text-emerald-400 mt-1">{errShown}%</div>
              <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">stays tiny even as the shock sharpens</div>
            </div>
          </div>
        </div>
      </div>

      {/* THE EXPERIMENT */}
      <div>
        <h2 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3 flex items-center gap-2"><HelpCircle size={15} /> The test · is spectral accurate enough to trust?</h2>
        <div className="grid grid-cols-4 gap-4">
          <Metric icon={Target} tone="emerald" tag="Accuracy" value={`${S.accVsExactPct}%`} label="whole-trajectory error vs the exact Cole-Hopf answer — essentially exact" />
          <Metric icon={Scale} tone="indigo" tag="Vs FDM" value={`${vsFdm}×`} label={`more accurate than finite-difference (${S.fdmVsExactPct}% vs ${S.accVsExactPct}%)`} />
          <Metric icon={Gauge} tone="emerald" tag="Extrapolation" value={`${S.extrapPct}%`} label="error past the training window — it holds, it does not drift" />
          <Metric icon={Cpu} tone="violet" tag="Role" value="Engine" label="the numerical solver the hybrid switches to when ML drifts (torch-free)" />
        </div>
      </div>

      {/* VERDICT */}
      <div className="rounded-2xl p-5 bg-gradient-to-r from-emerald-50 via-white to-white dark:from-emerald-500/10 dark:via-slate-800 dark:to-slate-800 border border-emerald-200 dark:border-emerald-500/30">
        <div className="text-sm font-semibold text-emerald-800 dark:text-emerald-300 flex items-center gap-2"><CheckCircle2 size={16} /> Answer</div>
        <p className="text-sm text-slate-700 dark:text-slate-200 mt-1">
          Spectral is <span className="font-medium">reference-grade</span> — it matches Cole-Hopf to a hundredth of a percent, stays accurate beyond the training window, and it is the
          <span className="font-semibold"> numerical engine the hybrid switches to</span> when the ML drifts. Accurate enough to trust, fast enough to deploy.
        </p>
      </div>
    </div>
  );
}
