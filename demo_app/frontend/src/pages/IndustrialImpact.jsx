import { useState } from "react";
import { Zap, GitBranch, ShieldCheck, ArrowRight, Gauge, Timer, SlidersHorizontal } from "lucide-react";

/* Industrial-impact framing: what the integrated system IS as a software artefact,
   why it matters outside this benchmark, and where the validated scope ends.
   Content is static (committed results); nothing here calls the backend. */

const STAGES = [
  {
    id: "fast", n: "01", eyebrow: "Fast path", icon: Zap, tone: "rose",
    title: "The ML surrogate predicts",
    text: "The learned model produces rapid PDE forecasts while its output remains acceptable — roughly two orders of magnitude cheaper than marching the numerical solver.",
    event: "ML surrogate accepted",
    detail: "Low-cost prediction path active",
  },
  {
    id: "decision", n: "02", eyebrow: "Runtime decision", icon: SlidersHorizontal, tone: "amber",
    title: "Trust and cost are assessed",
    text: "The monitor scores reliability without ground truth; the controller weighs that against the requested accuracy target and the compute budget.",
    event: "Intervention requested",
    detail: "Trust or accuracy policy activates the fallback",
  },
  {
    id: "failover", n: "03", eyebrow: "Verified failover", icon: ShieldCheck, tone: "indigo",
    title: "State moves to the trusted solver",
    text: "The coupling module transfers the live ML state into the verified pseudo-spectral continuation — continuous by construction, and proven to reproduce the production solver.",
    event: "Numerical continuation executed",
    detail: "Continuous hybrid trajectory returned",
  },
];

const BENEFITS = [
  { n: "01", icon: Timer, title: "Speed while trusted",
    text: "Run the inexpensive surrogate instead of the numerical solver continuously." },
  { n: "02", icon: ShieldCheck, title: "Reliability when needed",
    text: "Escalate through a verified state transfer once the fast path is no longer acceptable." },
  { n: "03", icon: Gauge, title: "Explicit cost control",
    text: "Accuracy and numerical work become measurable, requestable runtime trade-offs." },
];

const TONES = {
  rose: ["border-rose-200 dark:border-rose-500/30", "bg-rose-50 dark:bg-rose-500/10",
    "text-rose-600 dark:text-rose-400", "bg-rose-600"],
  amber: ["border-amber-200 dark:border-amber-500/30", "bg-amber-50 dark:bg-amber-500/10",
    "text-amber-600 dark:text-amber-400", "bg-amber-500"],
  indigo: ["border-indigo-200 dark:border-indigo-500/30", "bg-indigo-50 dark:bg-indigo-500/10",
    "text-indigo-600 dark:text-indigo-400", "bg-indigo-600"],
};

export default function IndustrialImpact() {
  const [i, setI] = useState(0);
  const s = STAGES[i];
  const T = TONES[s.tone];
  const StageIcon = s.icon;

  return (
    <div className="space-y-7">
      {/* HEADER */}
      <div>
        <div className="flex items-center gap-3 flex-wrap">
          <span className="text-xs font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">
            From PDE benchmark to software reliability
          </span>
          <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-amber-100 dark:bg-amber-500/20 text-amber-700 dark:text-amber-300">
            Validated on 1-D Burgers
          </span>
          <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300">
            Committed results
          </span>
        </div>
        <h1 className="text-3xl font-extrabold text-slate-800 dark:text-slate-100 mt-1">
          Fast prediction. <span className="text-indigo-600 dark:text-indigo-400">Verified intervention.</span>
        </h1>
        <p className="text-slate-500 dark:text-slate-400 mt-1 max-w-3xl">
          The engineered output of this project is not a solver comparison. It is a{" "}
          <span className="font-medium text-slate-700 dark:text-slate-200">trust-aware hybrid inference runtime</span>:
          it monitors a fast ML surrogate, decides when numerical work is justified against an accuracy target,
          and transfers the live state to a trusted numerical solver.
        </p>
      </div>

      {/* INTERACTIVE RUNTIME FLOW */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
        <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-4">
          The runtime, stage by stage — click a stage
        </div>

        <div className="flex gap-2 flex-wrap">
          {STAGES.map((st, k) => (
            <button key={st.id} onClick={() => setI(k)}
              className={`flex-1 min-w-[170px] text-left px-3 py-2.5 rounded-xl border transition ${k === i
                ? `${TONES[st.tone][1]} ${TONES[st.tone][0]}`
                : "bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700/40"}`}>
              <div className="flex items-center gap-2">
                <span className={`text-[10px] font-bold ${k === i ? TONES[st.tone][2] : "text-slate-400 dark:text-slate-500"}`}>{st.n}</span>
                <span className={`text-[10px] font-bold uppercase tracking-wider ${k === i ? TONES[st.tone][2] : "text-slate-400 dark:text-slate-500"}`}>
                  {st.eyebrow}
                </span>
              </div>
              <div className="text-sm font-semibold text-slate-800 dark:text-slate-100 mt-0.5">{st.title}</div>
            </button>
          ))}
        </div>

        <div className="grid grid-cols-[1fr_300px] gap-5 mt-5 max-lg:grid-cols-1">
          {/* stage copy + event */}
          <div>
            <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">
              Stage {s.n} / 03
            </div>
            <div className="flex items-center gap-2 mt-1">
              <div className={`w-9 h-9 rounded-xl grid place-items-center ${T[1]} ${T[2]}`}><StageIcon size={18} /></div>
              <h2 className="text-lg font-bold text-slate-800 dark:text-slate-100">{s.title}</h2>
            </div>
            <p className="text-sm text-slate-600 dark:text-slate-300 mt-2 max-w-xl">{s.text}</p>

            <div className={`mt-4 rounded-xl border ${T[0]} ${T[1]} px-4 py-3 flex items-start gap-3`}>
              <span className={`mt-1.5 w-2 h-2 rounded-full ${T[3]} animate-pulse shrink-0`} />
              <div>
                <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">Runtime event</div>
                <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">{s.event}</div>
                <div className="text-xs text-slate-500 dark:text-slate-400">{s.detail}</div>
              </div>
            </div>
          </div>

          {/* cascade diagram */}
          <div className="space-y-2">
            <div className={`rounded-xl border p-3 transition ${i === 0
              ? "border-rose-300 dark:border-rose-500/40 bg-rose-50 dark:bg-rose-500/10"
              : "border-slate-200 dark:border-slate-700 bg-slate-50/60 dark:bg-slate-700/30"}`}>
              <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">Fast engine</div>
              <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">ML surrogate</div>
              <div className="text-[11px] text-slate-500 dark:text-slate-400">rapid field prediction</div>
            </div>
            <div className={`rounded-xl border border-dashed p-2 text-center transition ${i === 1
              ? "border-amber-400 dark:border-amber-500/50 bg-amber-50 dark:bg-amber-500/10"
              : "border-slate-300 dark:border-slate-600"}`}>
              <div className="text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">Trust + cost policy</div>
              <ArrowRight size={14} className="mx-auto mt-0.5 rotate-90 text-slate-400 dark:text-slate-500" />
            </div>
            <div className={`rounded-xl border p-3 transition ${i === 2
              ? "border-indigo-300 dark:border-indigo-500/40 bg-indigo-50 dark:bg-indigo-500/10"
              : "border-slate-200 dark:border-slate-700 bg-slate-50/60 dark:bg-slate-700/30"}`}>
              <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">Trusted engine</div>
              <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">Pseudo-spectral solver</div>
              <div className="text-[11px] text-slate-500 dark:text-slate-400">verified restart from the ML state</div>
            </div>
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-3">
              <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">Output</div>
              <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">Hybrid forecast + intervention trace</div>
            </div>
          </div>
        </div>
      </div>

      {/* EVIDENCE STRIP */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
        <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide">
          Primary evidence · n = 20 held-out waves
        </div>
        <div className="text-sm font-semibold text-slate-800 dark:text-slate-100 mt-0.5 mb-3">
          Accuracy recovered after intervention
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex-1 min-w-[150px] rounded-xl bg-rose-50 dark:bg-rose-500/10 border border-rose-100 dark:border-rose-500/20 px-4 py-3 text-center">
            <div className="text-[10px] uppercase tracking-wide text-rose-500">Pure ML</div>
            <div className="text-3xl font-extrabold text-rose-600 dark:text-rose-400">13.44%</div>
            <div className="text-[11px] text-slate-500 dark:text-slate-400">mean future error</div>
          </div>
          <ArrowRight size={20} className="text-slate-300 dark:text-slate-600 shrink-0" />
          <div className="flex-1 min-w-[150px] rounded-xl bg-indigo-50 dark:bg-indigo-500/10 border-2 border-indigo-200 dark:border-indigo-500/30 px-4 py-3 text-center">
            <div className="text-[10px] uppercase tracking-wide text-indigo-500">Hybrid</div>
            <div className="text-3xl font-extrabold text-indigo-600 dark:text-indigo-400">0.98%</div>
            <div className="text-[11px] text-slate-500 dark:text-slate-400">mean future error</div>
          </div>
          <div className="flex-1 min-w-[150px] rounded-xl bg-emerald-50 dark:bg-emerald-500/10 border border-emerald-100 dark:border-emerald-500/20 px-4 py-3 text-center">
            <div className="text-[10px] uppercase tracking-wide text-emerald-600">Improved</div>
            <div className="text-3xl font-extrabold text-emerald-600 dark:text-emerald-400">20 / 20</div>
            <div className="text-[11px] text-slate-500 dark:text-slate-400">evaluated cases</div>
          </div>
          <div className="flex-1 min-w-[150px] rounded-xl bg-slate-50 dark:bg-slate-700/40 border border-slate-100 dark:border-slate-600/40 px-4 py-3 text-center">
            <div className="text-[10px] uppercase tracking-wide text-slate-400">Fallback fidelity</div>
            <div className="text-3xl font-extrabold text-slate-700 dark:text-slate-200">0.0</div>
            <div className="text-[11px] text-slate-500 dark:text-slate-400">rel. diff vs production solver</div>
          </div>
        </div>
      </div>

      {/* INDUSTRIAL VALUE */}
      <div>
        <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-1">
          Industrial value
        </div>
        <h2 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-3">
          A reliability layer for physics-ML services
        </h2>
        <div className="grid grid-cols-3 gap-4 max-lg:grid-cols-1">
          {BENEFITS.map((b) => {
            const Icon = b.icon;
            return (
              <div key={b.n} className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4 hover:-translate-y-1 transition-transform">
                <div className="flex items-center justify-between">
                  <div className="w-9 h-9 rounded-xl grid place-items-center bg-indigo-100 dark:bg-indigo-500/20 text-indigo-600 dark:text-indigo-400">
                    <Icon size={18} />
                  </div>
                  <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">{b.n}</span>
                </div>
                <div className="text-sm font-semibold text-slate-800 dark:text-slate-100 mt-3">{b.title}</div>
                <p className="text-xs text-slate-600 dark:text-slate-300 mt-1 leading-snug">{b.text}</p>
              </div>
            );
          })}
        </div>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-3 max-w-3xl">
          The same shape as a <span className="font-medium">model cascade</span> in production ML serving — cheap model
          first, escalate when confidence drops — with two things that pattern usually lacks: an escalation path
          <span className="font-medium"> verified equivalent</span> to the trusted engine, and an
          <span className="font-medium"> accuracy budget</span> as a user input rather than a hand-tuned threshold.
        </p>
      </div>

      {/* SCOPE BOUNDARY */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
        <div className="grid grid-cols-[1fr_auto_1fr_auto] gap-5 items-center max-lg:grid-cols-1">
          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-emerald-100 dark:bg-emerald-500/20 text-emerald-700 dark:text-emerald-300">
              Validated now
            </span>
            <div className="text-sm font-semibold text-slate-800 dark:text-slate-100 mt-2">Controlled nonlinear PDE benchmark</div>
            <p className="text-xs text-slate-600 dark:text-slate-300 mt-1">
              Integrated monitor, controller and verified ML-to-numerical hand-off, evaluated against Cole–Hopf
              reference truth.
            </p>
          </div>
          <ArrowRight size={22} className="text-slate-300 dark:text-slate-600 mx-auto" />
          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300">
              Intended direction
            </span>
            <div className="text-sm font-semibold text-slate-800 dark:text-slate-100 mt-2">Fluid and thermal digital twins</div>
            <p className="text-xs text-slate-600 dark:text-slate-300 mt-1">
              Real-time engineering forecasting where full numerical simulation is costly and silent model failure
              is unacceptable.
            </p>
          </div>
          <div className="rounded-xl border border-amber-200 dark:border-amber-500/30 bg-amber-50 dark:bg-amber-500/10 px-3 py-2 text-center">
            <div className="text-[10px] font-bold uppercase tracking-wider text-amber-700 dark:text-amber-300 leading-tight">
              Industrial<br />transfer is<br />future work
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
