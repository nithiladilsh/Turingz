import { useEffect, useState } from "react";
import {
  Target, Lightbulb, Globe2, Users, Gauge, Link2, SlidersHorizontal,
  AlertCircle, AlertTriangle, CheckCircle2, ArrowRight, Wrench, ChevronDown,
} from "lucide-react";

const cardCls = "bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700";

const H2 = ({ icon: Icon, children }) => (
  <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3">
    {Icon && <Icon size={15} />} {children}
  </h2>
);

/* fades + slides children up on mount, staggered by index — the only thing making
   this page feel alive instead of a static wall of text */
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

/* connecting pipeline strip: Problem -> Gap -> Novelty -> Contribution -> Team,
   with a line that draws itself in on mount */
function FlowStrip() {
  const [on, setOn] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setOn(true), 150);
    return () => clearTimeout(t);
  }, []);
  const steps = [
    { icon: Target, label: "Problem", tone: "bg-rose-500" },
    { icon: AlertCircle, label: "Gap", tone: "bg-amber-500" },
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

function ProblemCard({ n, title, body, i }) {
  return (
    <Reveal i={i}>
      <div className={`${cardCls} p-5 h-full hover:-translate-y-1 hover:shadow-md transition-all`}>
        <div className="flex items-center gap-2">
          <span className="w-6 h-6 rounded-full bg-rose-100 dark:bg-rose-500/20 text-rose-600 dark:text-rose-400 text-xs font-bold grid place-items-center">{n}</span>
          <span className="text-sm font-semibold text-slate-800 dark:text-slate-100">{title}</span>
        </div>
        <p className="text-sm text-slate-600 dark:text-slate-300 mt-2.5 leading-relaxed">{body}</p>
      </div>
    </Reveal>
  );
}

function NoveltyCard({ icon: Icon, tone, title, body, i }) {
  return (
    <Reveal i={i}>
      <div className={`${cardCls} p-5 flex gap-3 h-full hover:-translate-y-1 hover:shadow-md transition-all`}>
        <div className={`w-9 h-9 rounded-xl grid place-items-center text-white shrink-0 ${tone}`}><Icon size={17} /></div>
        <div>
          <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">{title}</div>
          <p className="text-sm text-slate-600 dark:text-slate-300 mt-1 leading-relaxed">{body}</p>
        </div>
      </div>
    </Reveal>
  );
}

/* small labelled block used inside each person's card for gap / novelty / contribution / IT-domain */
function MiniBlock({ icon: Icon, tone, label, children }) {
  return (
    <div className="rounded-xl bg-slate-50 dark:bg-slate-900/40 border border-slate-200 dark:border-slate-700 p-3">
      <div className={`inline-flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wide ${tone}`}>
        <Icon size={12} /> {label}
      </div>
      <p className="text-[12.5px] text-slate-600 dark:text-slate-300 mt-1.5 leading-snug">{children}</p>
    </div>
  );
}

function PersonCard({ icon: Icon, tone, name, id, module, scope, gap, novelty, contribution, itDomain, limitations, learned, i }) {
  const [open, setOpen] = useState(false);
  return (
    <Reveal i={i}>
      <div className={`${cardCls} p-5 hover:shadow-md transition-shadow`}>
        <div className="flex items-center gap-3">
          <div className={`w-11 h-11 rounded-xl grid place-items-center text-white shrink-0 ${tone}`}><Icon size={20} /></div>
          <div>
            <div className="text-sm font-bold text-slate-800 dark:text-slate-100">{name}</div>
            <div className="text-xs text-slate-500 dark:text-slate-400">{id} &middot; {module}</div>
          </div>
        </div>
        <p className="text-sm text-slate-600 dark:text-slate-300 mt-3 leading-relaxed">{scope}</p>

        <div className="grid grid-cols-2 gap-2.5 mt-4">
          <MiniBlock icon={Target} tone="text-rose-600 dark:text-rose-400" label="Gap addressed">{gap}</MiniBlock>
          <MiniBlock icon={Lightbulb} tone="text-indigo-600 dark:text-indigo-400" label="Novelty">{novelty}</MiniBlock>
          <MiniBlock icon={Wrench} tone="text-violet-600 dark:text-violet-400" label="Technical contribution">{contribution}</MiniBlock>
          <MiniBlock icon={Globe2} tone="text-emerald-600 dark:text-emerald-400" label="IT domain contribution">{itDomain}</MiniBlock>
          <div className="col-span-2">
            <MiniBlock icon={AlertTriangle} tone="text-amber-600 dark:text-amber-400" label="Known limitation">{limitations}</MiniBlock>
          </div>
        </div>

        <button
          onClick={() => setOpen((o) => !o)}
          className="w-full flex items-center justify-between mt-3 pt-3 border-t border-dashed border-slate-200 dark:border-slate-700 text-xs font-semibold text-slate-500 dark:text-slate-400 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
        >
          What I learned
          <ChevronDown size={14} className={`transition-transform ${open ? "rotate-180" : ""}`} />
        </button>
        <div className={`grid overflow-hidden transition-all duration-300 ${open ? "grid-rows-[1fr] opacity-100 mt-2" : "grid-rows-[0fr] opacity-0"}`}>
          <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed overflow-hidden">{learned}</p>
        </div>
      </div>
    </Reveal>
  );
}

export default function ResearchContribution({ go }) {
  return (
    <div className="space-y-8">
      {/* HEADER */}
      <div>
        <span className="text-xs font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">Project</span>
        <h1 className="text-3xl font-extrabold text-slate-800 dark:text-slate-100 mt-1">Research &amp; Contribution</h1>
        <p className="text-slate-500 dark:text-slate-400 mt-1 max-w-3xl">
          Why this project exists, what gap it fills, what is actually new about it, and who built what.
          The numbers behind every claim here are measured and shown on the pages that follow &mdash;
          this page is about the framing, not the figures.
        </p>
      </div>

      <FlowStrip />

      {/* THE PROBLEM */}
      <div>
        <H2 icon={Target}>The research problem</H2>
        <p className="text-sm text-slate-600 dark:text-slate-300 max-w-4xl mb-4 leading-relaxed">
          Partial differential equations sit behind most physical simulation &mdash; fluid flow, heat transfer, wave propagation
          &mdash; and solving them well matters across science and engineering. Machine learning has become an attractive way to
          solve them far faster than classical numerical methods, but speed alone raises a question classical methods never had
          to answer: when can a fast prediction actually be trusted? Three specific problems follow from that question.
        </p>
        <div className="grid grid-cols-3 gap-4">
          <ProblemCard i={0} n="1" title="No way to know, live, when to trust a prediction"
            body="A machine learning model keeps producing smooth, confident-looking output even after it has started failing. There is no correct answer available at the moment of prediction to check it against, so a user has no way to tell a good prediction from a confidently wrong one." />
          <ProblemCard i={1} n="2" title="No safe way to hand control back to a numerical solver"
            body="Once a failure is suspected, simply restarting a numerical solver from whatever state the model produced can introduce a jump or instability into the solution if the hand-off is not done carefully. Whether and when that hand-off is actually worth doing has not been well characterised." />
          <ProblemCard i={2} n="3" title="Cost and accuracy are rarely studied together"
            body="Some models are accurate but expensive to run; others are cheap but unreliable outside the conditions they were trained on. There is no principled way for a user to simply ask for a target accuracy and be told, honestly, what that costs." />
        </div>
      </div>

      {/* THE GAP */}
      <Reveal i={3}>
        <div className="bg-rose-50 dark:bg-rose-500/10 rounded-2xl border border-rose-100 dark:border-rose-500/20 p-6">
          <div className="flex items-center gap-2 text-xs font-semibold text-rose-500 dark:text-rose-400 uppercase tracking-wide">
            <AlertCircle size={15} /> The gap
          </div>
          <p className="text-sm text-slate-700 dark:text-slate-200 mt-2 leading-relaxed max-w-4xl">
            Existing work addresses pieces of this in isolation. Some methods estimate uncertainty but never hand off to a numerical
            solver. Some combine a learned model with a numerical one, but fix that combination in advance rather than deciding it
            while the system is running. Some react to a failure signal, but give the user no way to request a specific accuracy or
            to see what meeting it actually costs. No prior approach brings a live, reference-free trust signal, a verified hand-off
            that preserves the solution's continuity, and a controller driven by a user-requested accuracy target together into one
            system &mdash; and evaluates the whole thing end to end against both a pure machine-learning approach and a pure numerical one.
          </p>
        </div>
      </Reveal>

      {/* NOVELTY / CONTRIBUTION */}
      <div>
        <H2 icon={Lightbulb}>What is new here</H2>
        <div className="grid grid-cols-2 gap-4">
          <NoveltyCard i={0} icon={Gauge} tone="bg-indigo-600" title="A trust signal that never needs the correct answer"
            body="The system judges a running prediction using only signals available from the prediction itself, combined and calibrated per model, so it can tell live when a specific model is starting to fail &mdash; without ever comparing against ground truth during deployment." />
          <NoveltyCard i={1} icon={Link2} tone="bg-violet-600" title="A verified hand-off between two different kinds of solver"
            body="When the machine-learning prediction is no longer trusted, the numerical solver picks up exactly where the prediction left off, without introducing a discontinuity or instability into the solution &mdash; and the conditions under which that hand-off is actually worth doing are characterised, not assumed." />
          <NoveltyCard i={2} icon={SlidersHorizontal} tone="bg-fuchsia-600" title="Accuracy as something you can ask for"
            body="Rather than a fixed rule for when to invoke the numerical solver, the controller takes a target accuracy from the user and turns it into a spending decision at runtime, so the trade-off between speed and correctness is chosen deliberately instead of being left to a hard-coded threshold." />
          <NoveltyCard i={3} icon={CheckCircle2} tone="bg-emerald-600" title="The whole system, not just its parts"
            body="Each module is verified on its own and then integrated and evaluated as a single hybrid solver, benchmarked against a pure machine-learning approach and a pure numerical approach under the same conditions &mdash; including on inputs the system was not trained on." />
        </div>
      </div>

      {/* IT DOMAIN CONTRIBUTION (overall) */}
      <Reveal i={2}>
        <div className={`${cardCls} p-6`}>
          <div className="flex items-center gap-2 text-xs font-semibold text-indigo-600 dark:text-indigo-400 uppercase tracking-wide">
            <Globe2 size={15} /> Contribution to the IT domain &mdash; as a whole research effort
          </div>
          <p className="text-sm text-slate-700 dark:text-slate-200 mt-2 leading-relaxed max-w-4xl">
            The specific equation used here is a controlled benchmark, chosen because it is well understood and hard enough to expose
            failure. But the pattern the project develops &mdash; a system that knows when to trust a fast learned component, hands off
            cleanly to a slower but reliable one when it should not be trusted, and lets the speed-versus-correctness trade-off be
            requested rather than hard-coded &mdash; is a general one. It speaks directly to problems the wider IT and software
            engineering field is already facing as machine learning moves into production: how to keep an ML-driven system reliable
            when it silently drifts outside the conditions it was built for, how to fall back safely without a human in the loop, and
            how to make the cost of "safe" a number a system can be held accountable to rather than a guess. Treating <span className="font-medium">when a
            model can be trusted</span> as a first-class engineering problem, with a measured answer, is the contribution this project
            offers beyond the one equation it is demonstrated on &mdash; as a whole, the three modules together read as a template for
            trustworthy, cost-aware AI system design, not just as three separate results.
          </p>
        </div>
      </Reveal>

      {/* INDIVIDUAL SCOPE */}
      <div>
        <H2 icon={Users}>Individual scope</H2>
        <div className="space-y-4">
          <PersonCard i={0} icon={Gauge} tone="bg-indigo-600" name="Sandeepa D.S." id="214039V" module="Module 1 &middot; Reliability &amp; Trust"
            scope="Built and stress-tested the FDM baseline solver, confirmed the team's Cole&ndash;Hopf reference (built by Dharmapala) as the trustworthy ground truth, tested how far each machine-learning model could genuinely be trusted, and built the runtime trust system that watches a model while it runs."
            gap="No method offered a calibrated trust signal that works during a live prediction without extra models or a held-out reference answer to compare against."
            novelty="A trust score fused from several physics-based warning signs and calibrated per model, so it can flag failure without ever seeing the correct answer."
            contribution="Built and evaluated the FDM baseline against the team's Cole&ndash;Hopf reference to confirm it as ground truth, characterised how each surrogate degrades beyond its training range, and implemented the runtime monitor and coarse-reference check that triggers the hand-off."
            itDomain="A blueprint for runtime health-checking of any ML model in production where no ground truth is available at inference time — directly transferable to ML monitoring and observability tooling."
            limitations="The trust signal cannot make a surrogate more robust than it already is: if a model fails outright on inputs far from what it was trained on, the system flags it but inherits the failure rather than fixing it, and its thresholds are calibrated for this benchmark rather than shown to transfer elsewhere."
            learned="Learned how to judge a numerical method by stability, convergence and conservation rather than by appearance, how the three surrogate families behave inside and outside their training range, and &mdash; above all &mdash; how to estimate and calibrate trust in a model without a reference answer." />
          <PersonCard i={1} icon={Link2} tone="bg-violet-600" name="Dharmapala R.D." id="214050V" module="Module 2 &middot; Coupling"
            scope="Led the coupling mechanism that transfers a predicted state from the machine-learning model to the numerical solver for continued evolution, and built the reference solver used to generate the project's ground truth."
            gap="Selective numerical correction with return of control had not been analysed for restart fidelity, and no prior work characterised when a mid-trajectory hand-off is actually worth taking."
            novelty="A verified, continuity-preserving re-anchoring mechanism that lets a live prediction hand over to a numerical solver with no discontinuity in the trajectory."
            contribution="Implemented and verified the restart-capable coupling adapter, and ran the systematic experiments — switch-time sweeps, restart-fidelity stress tests — that establish when a hand-off actually helps."
            itDomain="A general pattern for safe state hand-off between heterogeneous system components — e.g. failing over from a fast, approximate service to a slow, exact one without breaking the continuity of an in-flight operation."
            limitations="The switch-time and restart-fidelity studies were evaluated on 20 of the 100 held-out test initial conditions at a single spatial resolution (extensible to the full 100 with no retraining), so the exact point where a hand-off stops being worth it should be read as a property of this benchmark, not yet shown to hold on other equations."
            learned="Learned that the apparent simplicity of a hand-off conceals a real verification problem &mdash; a solver restarted mid-trajectory must be shown, not assumed, to behave like the trusted production scheme &mdash; and gained experience separating a numerical solver's own error from error inherited from the state it was handed." />
          <PersonCard i={2} icon={SlidersHorizontal} tone="bg-fuchsia-600" name="Mendis B.N.D." id="214133E" module="Module 3 &middot; Cost-Aware Control &amp; Deployment"
            scope="Led the cost-aware controller and deployment layer, and implemented the numerical scheme used as both the team's verifier and the hybrid's corrector."
            gap="No controller accepted a user-specified accuracy target and scheduled numerical effort to meet it at a reduced, measured cost, and no deployed hybrid reported a measured cost-versus-accuracy trade-off."
            novelty="The accuracy-budget idea: turning a requested accuracy target into a calibrated runtime spending decision, so 'how accurate' becomes something a user can ask for instead of a fixed threshold buried in code."
            contribution="Built the controller, measured the full cost-accuracy trade-off against pure machine-learning and pure numerical baselines, and traced the accuracy limit back to the trust signal rather than the controller itself."
            itDomain="A reusable pattern for cost-aware adaptive resource allocation in any system trading compute cost against quality of service on demand — relevant to cloud cost optimisation and adaptive service-level systems generally."
            limitations="The accuracy the controller can promise is capped by how sensitive the trust signal is, not by the controller itself: past a certain point, asking for a tighter target stops changing the outcome, because that ceiling belongs to the signal it acts on, not to the scheduling logic."
            learned="Learned that the value of an adaptive controller is not the switching rule itself, but making accuracy something a user can request and the system can honestly deliver or refuse." />
        </div>
      </div>

      {/* FOOTER NOTE */}
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
