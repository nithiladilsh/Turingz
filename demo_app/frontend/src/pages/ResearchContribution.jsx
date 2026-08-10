import { useEffect, useState } from "react";
import {
  Target, Lightbulb, Globe2, Users, Gauge, Link2, SlidersHorizontal,
  AlertTriangle, CheckCircle2, ArrowRight, Wrench,
} from "lucide-react";

const cardCls = "bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700";

const H2 = ({ icon: Icon, children }) => (
  <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3">
    {Icon && <Icon size={15} />} {children}
  </h2>
);

function Reveal({ children, i = 0, className = "" }) {
  const [on, setOn] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setOn(true), 80 + i * 90);
    return () => clearTimeout(t);
  }, [i]);
  return (
    <div
      className={`transition-all duration-500 ease-out ${on ? "opacity-100 translate-y-0" : "opacity-0 translate-y-3"} ${className}`}
    >
      {children}
    </div>
  );
}

function FlowStrip() {
  const [on, setOn] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setOn(true), 150);
    return () => clearTimeout(t);
  }, []);
  const steps = [
    { icon: Lightbulb, label: "Novelty", tone: "bg-indigo-500" },
    { icon: Globe2, label: "Contribution", tone: "bg-emerald-500" },
    { icon: Users, label: "Who did what", tone: "bg-fuchsia-500" },
  ];
  return (
    <div className={`${cardCls} px-8 py-6`}>
      <div className="relative">
        <div className="absolute top-6 left-[8%] right-[8%] h-0.5 rounded-full bg-slate-100 dark:bg-slate-700" />
        <div
          className="absolute top-6 left-[8%] h-0.5 rounded-full bg-gradient-to-r from-rose-400 via-indigo-400 to-fuchsia-400 transition-all duration-[1600ms] ease-out"
          style={{ width: on ? "84%" : "0%" }}
        />
        <div className="relative flex justify-between">
          {steps.map((s, i) => (
            <div key={s.label} className="flex flex-col items-center gap-2 w-[100px]">
              <div
                className={`w-12 h-12 rounded-2xl grid place-items-center text-white shadow-sm transition-all duration-500 ${s.tone}`}
                style={{ transitionDelay: `${i * 150}ms`, transform: on ? "scale(1)" : "scale(0.4)", opacity: on ? 1 : 0 }}
              >
                <s.icon size={20} />
              </div>
              <div className="text-xs font-semibold text-slate-600 dark:text-slate-300 text-center">{s.label}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function NoveltyCard({ icon: Icon, tone, title, body, i }) {
  return (
    <Reveal i={i}>
      <div className={`${cardCls} p-4 flex gap-3 h-full hover:-translate-y-1 hover:shadow-md transition-all`}>
        <div className={`w-9 h-9 rounded-xl grid place-items-center text-white shrink-0 ${tone}`}><Icon size={17} /></div>
        <div>
          <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">{title}</div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 leading-snug">{body}</p>
        </div>
      </div>
    </Reveal>
  );
}

function MiniBlock({ icon: Icon, tone, label, children }) {
  return (
    <div className="rounded-xl bg-slate-50 dark:bg-slate-900/40 border border-slate-200 dark:border-slate-700 p-2.5">
      <div className={`inline-flex items-center gap-1.5 text-[9.5px] font-bold uppercase tracking-wide ${tone}`}>
        <Icon size={11} /> {label}
      </div>
      <p className="text-xs text-slate-600 dark:text-slate-300 mt-1 leading-snug">{children}</p>
    </div>
  );
}

function PersonCard({ icon: Icon, tone, name, id, module, scope, gap, novelty, contribution, itDomain, limitations, i }) {
  return (
    <Reveal i={i}>
      <div className={`${cardCls} p-4 hover:shadow-md transition-shadow`}>
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-xl grid place-items-center text-white shrink-0 ${tone}`}><Icon size={18} /></div>
          <div>
            <div className="text-sm font-bold text-slate-800 dark:text-slate-100">{name}</div>
            <div className="text-xs text-slate-500 dark:text-slate-400">{id} &middot; {module}</div>
          </div>
        </div>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-2.5 leading-snug">{scope}</p>

        <div className="grid grid-cols-2 gap-2 mt-3">
          <MiniBlock icon={Target} tone="text-rose-600 dark:text-rose-400" label="Gap">{gap}</MiniBlock>
          <MiniBlock icon={Lightbulb} tone="text-indigo-600 dark:text-indigo-400" label="Novelty">{novelty}</MiniBlock>
          <MiniBlock icon={Wrench} tone="text-violet-600 dark:text-violet-400" label="Contribution">{contribution}</MiniBlock>
          <MiniBlock icon={Globe2} tone="text-emerald-600 dark:text-emerald-400" label="IT domain">{itDomain}</MiniBlock>
        </div>
        <div className="mt-2">
          <MiniBlock icon={AlertTriangle} tone="text-amber-600 dark:text-amber-400" label="Limitation">{limitations}</MiniBlock>
        </div>
      </div>
    </Reveal>
  );
}

export default function ResearchContribution({ go }) {
  return (
    <div className="space-y-6">
      <div>
        <span className="text-xs font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">Project</span>
        <h1 className="text-3xl font-extrabold text-slate-800 dark:text-slate-100 mt-1">Research &amp; Contribution</h1>
        <p className="text-slate-500 dark:text-slate-400 mt-1 max-w-3xl">
          What's new about this system, and who built what. The numbers behind every claim are on the pages that follow.
        </p>
      </div>

      <FlowStrip />

      <div>
        <H2 icon={Lightbulb}>What's new</H2>
        <div className="grid grid-cols-2 gap-3">
          <NoveltyCard i={0} icon={Gauge} tone="bg-indigo-600" title="Trust without a reference answer"
            body="Fails a prediction using only signals from the prediction itself, calibrated per model." />
          <NoveltyCard i={1} icon={Link2} tone="bg-violet-600" title="A verified hand-off"
            body="Re-anchors into the numerical solver with zero jump — characterised, not assumed." />
          <NoveltyCard i={2} icon={SlidersHorizontal} tone="bg-fuchsia-600" title="Accuracy you can ask for"
            body="One requested target becomes one runtime spending decision, not a hard-coded rule." />
          <NoveltyCard i={3} icon={CheckCircle2} tone="bg-emerald-600" title="Evaluated as one system"
            body="All three modules integrated and benchmarked against pure ML and pure numerical baselines." />
        </div>
      </div>

      <Reveal i={2}>
        <div className={`${cardCls} p-5`}>
          <div className="flex items-center gap-2 text-xs font-semibold text-indigo-600 dark:text-indigo-400 uppercase tracking-wide">
            <Globe2 size={15} /> Beyond this benchmark
          </div>
          <p className="text-sm text-slate-700 dark:text-slate-200 mt-2">
            The equation is a controlled testbed; the pattern — know when to trust a fast model, fail over cleanly,
            make "safe" a requested number — applies to any ML-in-production system.
          </p>
        </div>
      </Reveal>

      <div>
        <H2 icon={Users}>Individual scope</H2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <PersonCard i={0} icon={Gauge} tone="bg-indigo-600" name="Sandeepa D.S." id="214039V" module="Module 1 &middot; Trust"
            scope="Built the FDM baseline, confirmed the Cole&ndash;Hopf reference as ground truth, and built the runtime trust monitor."
            gap="No live trust signal without a held-out reference to check against."
            novelty="Trust score fused from physics-based signals, calibrated per model."
            contribution="FDM baseline, surrogate degradation study, runtime trust monitor."
            itDomain="Reference-free health-checking for ML in production."
            limitations="Flags a bad surrogate — can't fix it." />
          <PersonCard i={1} icon={Link2} tone="bg-violet-600" name="Dharmapala R.D." id="214050V" module="Module 2 &middot; Coupling"
            scope="Led the ML-to-numerical hand-off and built the reference solver used for ground truth."
            gap="Hand-offs are assumed safe in prior hybrids, not verified."
            novelty="Hand-off treated as a numerical operation, verified for continuity."
            contribution="Restart-capable coupling adapter; switch-time and restart-fidelity experiments."
            itDomain="Safe state hand-off between heterogeneous services."
            limitations="Verified at one resolution, one benchmark equation." />
          <PersonCard i={2} icon={SlidersHorizontal} tone="bg-fuchsia-600" name="Mendis B.N.D." id="214133E" module="Module 3 &middot; Cost Control"
            scope="Led the cost-aware controller and the numerical scheme used as verifier and corrector."
            gap="No controller took a requested accuracy target and scheduled effort to meet it."
            novelty="Accuracy target &rarr; a calibrated runtime spending decision."
            contribution="Controller, full cost-accuracy trade-off, traced the accuracy ceiling to the trust signal."
            itDomain="Cost-aware adaptive resource allocation on demand."
            limitations="Accuracy ceiling is set by the trust signal, not the controller." />
        </div>
      </div>

      <Reveal i={4}>
        <button
          onClick={() => go && go("hybrid")}
          className="w-full text-left rounded-2xl p-5 bg-gradient-to-r from-indigo-50 via-white to-white dark:from-indigo-500/10 dark:via-slate-800 dark:to-slate-800 border border-indigo-200 dark:border-indigo-500/30 flex items-center justify-between gap-4 hover:-translate-y-0.5 hover:shadow-md transition-all cursor-pointer"
        >
          <p className="text-sm text-slate-700 dark:text-slate-200">
            Next: the full pipeline, running live.
          </p>
          <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-indigo-600 dark:text-indigo-400 shrink-0">
            Open the hybrid engine <ArrowRight size={14} />
          </span>
        </button>
      </Reveal>
    </div>
  );
}
