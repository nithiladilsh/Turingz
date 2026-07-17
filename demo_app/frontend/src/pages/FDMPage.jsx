import { useEffect, useState } from "react";
import { FDM_XS, FDM_FRAMES, FDM_STATS as S } from "../fdmData.js";
import {
  TrendingUp,
  Waves,
  ShieldCheck,
  Scale,
  HelpCircle,
  Grid3x3,
  Share2,
  Clock,
  ArrowRight,
} from "lucide-react";

/* ---- auto-looping FDM vs exact, with the error shaded ----
   GAP exaggerates the FDM–exact difference for on-screen clarity only
   (the reported error % and the evidence numbers are the real, unexaggerated values). */
const GAP = 2.6;
function BlurChart({ frame }) {
  const W = 580,
    H = 300,
    padX = 18,
    padT = 18,
    padB = 28;
  const clamp = (v) => Math.max(-1.9, Math.min(1.9, v));
  const fdmY = frame.fdm.map((v, i) =>
    clamp(frame.exact[i] + GAP * (v - frame.exact[i])),
  );
  const sx = (x) => padX + ((x + 1) / 2) * (W - 2 * padX);
  const sy = (v) => H - padB - ((v + 2) / 4) * (H - padT - padB);
  const path = (arr) =>
    arr
      .map(
        (v, i) =>
          `${i ? "L" : "M"}${sx(FDM_XS[i]).toFixed(1)} ${sy(v).toFixed(1)}`,
      )
      .join(" ");
  let band = `M${sx(FDM_XS[0]).toFixed(1)} ${sy(frame.exact[0]).toFixed(1)}`;
  FDM_XS.forEach(
    (x, i) => (band += `L${sx(x).toFixed(1)} ${sy(frame.exact[i]).toFixed(1)}`),
  );
  for (let i = FDM_XS.length - 1; i >= 0; i--)
    band += `L${sx(FDM_XS[i]).toFixed(1)} ${sy(fdmY[i]).toFixed(1)}`;
  band += "Z";
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {[-1.5, -0.75, 0, 0.75, 1.5].map((v) => (
        <line
          key={v}
          x1={sx(-1)}
          x2={sx(1)}
          y1={sy(v)}
          y2={sy(v)}
          stroke="var(--chart-grid)"
          strokeWidth="1"
        />
      ))}
      <path d={band} fill="#fb7185" fillOpacity="0.2" />
      <path
        d={path(frame.exact)}
        fill="none"
        stroke="#94a3b8"
        strokeWidth="2"
        strokeDasharray="5 4"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      <path
        d={path(fdmY)}
        fill="none"
        stroke="#4f46e5"
        strokeWidth="3"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      <text
        x={sx(-1) + 4}
        y={padT + 4}
        fontSize="11"
        fill="#fb7185"
        fontWeight="700"
      >
        shaded = FDM's error
      </text>
    </svg>
  );
}

function Metric({ icon: Icon, tone, tag, value, label }) {
  const c = {
    rose: [
      "bg-rose-50 dark:bg-rose-500/10 border-rose-100 dark:border-rose-500/20",
      "bg-rose-100 dark:bg-rose-500/20 text-rose-600 dark:text-rose-400",
      "text-rose-600 dark:text-rose-400",
    ],
    amber: [
      "bg-amber-50 dark:bg-amber-500/10 border-amber-100 dark:border-amber-500/20",
      "bg-amber-100 dark:bg-amber-500/20 text-amber-600 dark:text-amber-400",
      "text-amber-600 dark:text-amber-400",
    ],
    emerald: [
      "bg-emerald-50 dark:bg-emerald-500/10 border-emerald-100 dark:border-emerald-500/20",
      "bg-emerald-100 dark:bg-emerald-500/20 text-emerald-600 dark:text-emerald-400",
      "text-emerald-600 dark:text-emerald-400",
    ],
  }[tone];
  return (
    <div
      className={`rounded-2xl border p-4 hover:-translate-y-1 transition-transform ${c[0]}`}
    >
      <div className="flex items-center justify-between">
        <div className={`w-9 h-9 rounded-xl grid place-items-center ${c[1]}`}>
          <Icon size={18} />
        </div>
        {tag && (
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">
            {tag}
          </span>
        )}
      </div>
      <div className={`text-3xl font-extrabold mt-3 ${c[2]}`}>{value}</div>
      <div className="text-xs text-slate-600 dark:text-slate-300 mt-1 leading-snug">
        {label}
      </div>
    </div>
  );
}

function Step({ n, icon: Icon, tone, title, sub }) {
  const c = {
    indigo: "bg-indigo-600",
    violet: "bg-violet-600",
    fuchsia: "bg-fuchsia-600",
  }[tone];
  return (
    <div className="flex-1 min-w-[150px] rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50/60 dark:bg-slate-700/30 p-3 flex items-center gap-3">
      <div
        className={`relative w-10 h-10 rounded-xl grid place-items-center text-white shrink-0 ${c}`}
      >
        <Icon size={18} />
        <span className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-white dark:bg-slate-800 text-[11px] font-bold grid place-items-center text-slate-700 dark:text-slate-200 border border-slate-200 dark:border-slate-600">
          {n}
        </span>
      </div>
      <div>
        <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">
          {title}
        </div>
        <div className="text-[11px] text-slate-500 dark:text-slate-400 leading-tight">
          {sub}
        </div>
      </div>
    </div>
  );
}

export default function FDMPage() {
  const [i, setI] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setI((k) => (k + 1) % FDM_FRAMES.length), 160);
    return () => clearInterval(id);
  }, []);
  const frame = FDM_FRAMES[i];
  const errPct = Math.round(frame.err * 100);

  return (
    <div className="space-y-7">
      {/* HEADER */}
      <div>
        <span className="text-xs font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">
          Numerical solvers
        </span>
        <h1 className="text-3xl font-extrabold text-slate-800 dark:text-slate-100 mt-1">
          Finite-Difference Method
        </h1>
        <p className="text-slate-500 dark:text-slate-400 mt-1">
          We built FDM and asked one question —{" "}
          <span className="font-medium text-slate-700 dark:text-slate-200">
            can it be our source of truth?
          </span>{" "}
          Here is the method, the experiment, and the answer.
        </p>
      </div>

      {/* HOW IT WORKS */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
        <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-4">
          How FDM works — three steps
        </div>
        <div className="flex items-stretch gap-2 flex-wrap">
          <Step
            n="1"
            icon={Grid3x3}
            tone="indigo"
            title="Grid"
            sub="split the domain into points"
          />
          <ArrowRight
            size={18}
            className="self-center shrink-0 text-slate-300 dark:text-slate-600"
          />
          <Step
            n="2"
            icon={Share2}
            tone="violet"
            title="Neighbours"
            sub="update each point from its left & right"
          />
          <ArrowRight
            size={18}
            className="self-center shrink-0 text-slate-300 dark:text-slate-600"
          />
          <Step
            n="3"
            icon={Clock}
            tone="fuchsia"
            title="March"
            sub="step forward in small time steps"
          />
        </div>

        <div className="mt-5 rounded-xl bg-slate-50 dark:bg-slate-700/40 p-4">
          <div className="text-center font-mono text-[15px] text-slate-700 dark:text-slate-200 leading-relaxed">
            uᵢⁿ⁺¹ = uᵢⁿ + Δt (
            <span className="text-indigo-600 dark:text-indigo-400 font-semibold">
              {" "}
              −uᵢ (uᵢ−uᵢ₋₁)/Δx{" "}
            </span>
            +
            <span className="text-teal-600 dark:text-teal-400 font-semibold">
              {" "}
              ν (uᵢ₊₁−2uᵢ+uᵢ₋₁)/Δx²{" "}
            </span>
            )
          </div>
          <div className="flex items-center justify-center gap-6 mt-3 text-xs">
            <span className="flex items-center gap-1.5 text-slate-500 dark:text-slate-400">
              <span className="w-2.5 h-2.5 rounded bg-indigo-500" />{" "}
              <span className="font-semibold text-indigo-600 dark:text-indigo-400">
                upwind
              </span>{" "}
              — the moving shock
            </span>
            <span className="flex items-center gap-1.5 text-slate-500 dark:text-slate-400">
              <span className="w-2.5 h-2.5 rounded bg-teal-500" />{" "}
              <span className="font-semibold text-teal-600 dark:text-teal-400">
                central
              </span>{" "}
              — the smoothing
            </span>
          </div>
        </div>
      </div>

      {/* COMPARISON (auto-loop) */}
      <div className="grid grid-cols-[1fr_230px] gap-5 items-stretch">
        <div className="relative rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4 overflow-hidden">
          <div className="absolute -top-8 -right-8 w-40 h-40 rounded-full bg-indigo-100/50 dark:bg-indigo-500/10 blur-2xl" />
          <div className="relative">
            <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">
              FDM blurs the sharp shock
            </div>
            <div className="text-xs text-slate-500 dark:text-slate-400">
              the exact answer stays crisp; FDM rounds it off — the gap is FDM's
              error
            </div>
            <BlurChart frame={frame} />
            <div className="flex items-center justify-center gap-5 text-[11px] text-slate-400 dark:text-slate-500">
              <span className="flex items-center gap-1">
                <span className="w-3 h-1 rounded-full bg-indigo-500" /> FDM
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-1 rounded-full bg-slate-400" /> exact
                (Cole-Hopf)
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-2 rounded bg-rose-300" /> error
              </span>
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4 text-center">
            <div className="text-xs text-slate-500 dark:text-slate-400">
              time
            </div>
            <div className="text-2xl font-bold text-slate-800 dark:text-slate-100">
              t = {frame.t.toFixed(2)}
            </div>
          </div>
          <div className="flex-1 rounded-2xl border border-rose-200 dark:border-rose-500/30 bg-rose-50 dark:bg-rose-500/10 p-4 text-center grid place-items-center">
            <div>
              <div className="text-xs text-rose-500 dark:text-rose-400">
                FDM error vs exact
              </div>
              <div className="text-5xl font-extrabold text-rose-600 dark:text-rose-400 mt-1">
                {errPct}%
              </div>
              <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">
                grows as the shock sharpens
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* THE EXPERIMENT — findings */}
      <div>
        <h2 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3 flex items-center gap-2">
          <HelpCircle size={15} /> The experiment · can FDM be the reference?
        </h2>
        <div className="grid grid-cols-4 gap-4">
          <Metric
            icon={TrendingUp}
            tone="rose"
            tag="Test 1"
            value={`${S.selfConv512}%`}
            label="still off its own finer grid at 512 — it hasn't converged"
          />
          <Metric
            icon={Waves}
            tone="amber"
            tag="Test 2"
            value={`${S.fakeSmoothing512}%`}
            label={`fake smoothing vs real physics — blurs the shock (needs ~${S.nxFor10} pts)`}
          />
          <Metric
            icon={ShieldCheck}
            tone="emerald"
            tag="Test 3"
            value="Stable"
            label={`mass conserved · energy never rises (${S.energyRising} times)`}
          />
          <Metric
            icon={Scale}
            tone="rose"
            tag="Cross-check"
            value={`${S.timesWorse}×`}
            label={`worse than exact: FDM ${S.fdmVsExact}% vs spectral ${S.spectralVsExact}%`}
          />
        </div>
      </div>

      {/* VERDICT */}
      <div className="rounded-2xl p-5 bg-gradient-to-r from-emerald-50 via-white to-white dark:from-emerald-500/10 dark:via-slate-800 dark:to-slate-800 border border-emerald-200 dark:border-emerald-500/30">
        <div className="text-sm font-semibold text-emerald-800 dark:text-emerald-300">
          Answer
        </div>
        <p className="text-sm text-slate-700 dark:text-slate-200 mt-1">
          FDM is <span className="font-medium">valid, stable and cheap</span> —
          but not accurate enough to be the source of truth. It stays as an
          independent baseline, and
          <span className="font-semibold">
            {" "}
            Cole-Hopf (confirmed by the spectral method) is the reference.
          </span>
        </p>
      </div>
    </div>
  );
}
