import { useEffect, useState } from "react";
import {
  Zap,
  ShieldCheck,
  GitMerge,
  Gauge,
  Link2,
  SlidersHorizontal,
  ArrowRight,
} from "lucide-react";

/* ---- animated concept chart: ML diverges, hybrid stays accurate ---- */
function ConceptChart() {
  const [p, setP] = useState(0);
  useEffect(() => {
    let raf, start;
    const loop = (ts) => {
      if (!start) start = ts;
      setP(Math.min(1, ((ts - start) / 3500) % 1.25));
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);
  const W = 360,
    H = 210,
    pad = 26,
    xs = 0.55;
  const sx = (x) => pad + x * (W - 2 * pad);
  const sy = (e) => H - pad - Math.min(1, e) * (H - 2 * pad);
  const mlErr = (x) => Math.min(1, 0.04 + 5 * Math.max(0, x - 0.45) ** 2);
  const hyErr = (x) => (x < xs ? mlErr(x) : 0.06);
  const line = (f) => {
    const pts = [];
    for (let x = 0; x <= p; x += 0.02) pts.push([sx(x), sy(f(x))]);
    return pts
      .map((q, i) => `${i ? "L" : "M"}${q[0].toFixed(1)} ${q[1].toFixed(1)}`)
      .join(" ");
  };
  const head = (f) => [sx(Math.min(p, 1)), sy(f(Math.min(p, 1)))];
  const switched = p >= xs;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      <rect
        x="0"
        y="0"
        width={sx(0.5)}
        height={H}
        fill="#eef2ff"
        opacity="0.5"
      />
      <text x={sx(0.25)} y={16} textAnchor="middle" fontSize="9" fill="#818cf8">
        trained window
      </text>
      <line
        x1={pad}
        x2={W - pad}
        y1={sy(0.02)}
        y2={sy(0.02)}
        stroke="#e2e8f0"
      />
      {switched && (
        <line
          x1={sx(xs)}
          x2={sx(xs)}
          y1={pad}
          y2={H - pad}
          stroke="#e11d48"
          strokeWidth="1.5"
          strokeDasharray="4 3"
        />
      )}
      <path
        d={line((x) => 0.02)}
        fill="none"
        stroke="#64748b"
        strokeWidth="2"
      />
      <path d={line(mlErr)} fill="none" stroke="#e11d48" strokeWidth="2.5" />
      <path d={line(hyErr)} fill="none" stroke="#059669" strokeWidth="3" />
      {[
        ["#64748b", () => 0.02],
        ["#e11d48", mlErr],
        ["#059669", hyErr],
      ].map(([c, f], i) => {
        const [hx, hy] = head(f);
        return <circle key={i} cx={hx} cy={hy} r="3.5" fill={c} />;
      })}
      <text
        x={W - pad}
        y={sy(mlErr(1)) - 6}
        textAnchor="end"
        fontSize="10"
        fill="#e11d48"
        fontWeight="600"
      >
        pure ML
      </text>
      <text
        x={W - pad}
        y={sy(0.02) - 6}
        textAnchor="end"
        fontSize="10"
        fill="#64748b"
        fontWeight="600"
      >
        numerical
      </text>
      <text
        x={sx(0.62)}
        y={sy(0.06) + 16}
        fontSize="10"
        fill="#059669"
        fontWeight="600"
      >
        hybrid
      </text>
    </svg>
  );
}

function Stat({ n, label }) {
  return (
    <div className="bg-white rounded-2xl border border-slate-200 px-4 py-3 text-center">
      <div className="text-2xl font-bold text-indigo-600">{n}</div>
      <div className="text-xs text-slate-500 mt-0.5">{label}</div>
    </div>
  );
}

function Family({ icon: Icon, color, title, tag }) {
  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-4 flex items-start gap-3 hover:shadow-md transition-shadow">
      <div className={`w-10 h-10 rounded-xl grid place-items-center ${color}`}>
        <Icon size={20} />
      </div>
      <div>
        <div className="text-sm font-semibold text-slate-800">{title}</div>
        <div className="text-xs text-slate-500 mt-0.5">{tag}</div>
      </div>
    </div>
  );
}

function ModuleCard({ n, icon: Icon, name, owner, one, tone }) {
  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-4 hover:shadow-md hover:-translate-y-0.5 transition-all">
      <div className="flex items-center gap-2.5">
        <div
          className={`w-9 h-9 rounded-xl grid place-items-center text-white ${tone}`}
        >
          <Icon size={18} />
        </div>
        <div>
          <div className="text-sm font-semibold text-slate-800">
            {n} · {name}
          </div>
          <div className="text-xs text-slate-500">{owner}</div>
        </div>
      </div>
      <p className="text-xs text-slate-500 mt-3">{one}</p>
    </div>
  );
}

function Node({ label, tone }) {
  const c = {
    infra: "bg-slate-100 text-slate-600",
    m: "bg-indigo-600 text-white",
    out: "bg-emerald-500 text-white",
  }[tone];
  return (
    <div className={`rounded-xl px-3 py-2 text-xs font-semibold ${c}`}>
      {label}
    </div>
  );
}

export default function Overview({ go }) {
  return (
    <div className="space-y-8">
      {/* RESEARCH TOPIC — first thing examiners see */}
      <div className="text-center border-b border-slate-200 pb-6">
        <div className="inline-flex items-center gap-2 text-xs font-medium text-indigo-700 bg-indigo-50 px-3 py-1 rounded-full">
          Team Turingz · University of Moratuwa
        </div>
        <div className="text-xl md:text-3xl font-extrabold text-slate-800 leading-[1.12] mt-4 max-w-10xl mx-auto tracking-tight">
          Hybrid Machine Learning and Numerical Methods for
          <span className="text-indigo-600"> PDE Extrapolation</span>
        </div>
        <p className="text-slate-500 mt-3 text-base">
          A trust-gated, cost-aware hybrid solver for long-horizon prediction
        </p>
      </div>

      {/* HERO */}
      <div className="grid grid-cols-[1fr_400px] gap-6 items-center">
        <div>
          <h2 className="text-2xl font-bold text-slate-800 leading-tight">
            Fast where it can be trusted.
            <br />
            <span className="text-indigo-600">Exact where it cannot.</span>
          </h2>
          <p className="text-slate-600 mt-3 max-w-lg">
            The solver runs a fast ML model and switches to an accurate
            numerical solver the moment the ML is about to fail.
          </p>
          <button
            onClick={() => go("trust")}
            className="mt-5 inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-indigo-600 text-white text-sm font-semibold hover:bg-indigo-700 transition-colors"
          >
            See it live <ArrowRight size={16} />
          </button>
        </div>
        <div className="bg-white rounded-2xl border border-slate-200 p-4 shadow-sm">
          <ConceptChart />
          <div className="text-xs text-slate-400 text-center mt-1">
            error over time — ML diverges, the hybrid stays accurate
          </div>
        </div>
      </div>

      {/* STATS */}
      <div className="grid grid-cols-4 gap-3">
        <Stat n="3" label="numerical solvers" />
        <Stat n="3" label="ML surrogates" />
        <Stat n="3" label="method modules" />
        <Stat n="1" label="hybrid engine" />
      </div>

      {/* FAMILIES */}
      <div>
        <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-3">
          Two worlds, one solver
        </h2>
        <div className="grid grid-cols-3 gap-4">
          <Family
            icon={Zap}
            color="bg-amber-100 text-amber-600"
            title="ML solvers"
            tag="Fast · fail silently later"
          />
          <Family
            icon={ShieldCheck}
            color="bg-slate-100 text-slate-600"
            title="Numerical solvers"
            tag="Accurate · slow"
          />
          <Family
            icon={GitMerge}
            color="bg-emerald-100 text-emerald-600"
            title="Our hybrid"
            tag="Fast and accurate"
          />
        </div>
      </div>

      {/* PIPELINE */}
      <div>
        <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-3">
          How it works
        </h2>
        <div className="bg-white rounded-2xl border border-slate-200 p-5 flex flex-wrap items-center justify-center gap-2">
          <Node label="Shared foundation" tone="infra" />
          <ArrowRight size={16} className="text-slate-300" />
          <Node label="1 · Trust" tone="m" />
          <ArrowRight size={16} className="text-slate-300" />
          <Node label="2 · Coupling" tone="m" />
          <ArrowRight size={16} className="text-slate-300" />
          <Node label="3 · Control" tone="m" />
          <ArrowRight size={16} className="text-slate-300" />
          <Node label="Hybrid output" tone="out" />
        </div>
      </div>

      {/* MODULES */}
      <div>
        <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-3">
          Three research modules
        </h2>
        <div className="grid grid-cols-3 gap-4">
          <ModuleCard
            n="1"
            icon={Gauge}
            name="Trust"
            owner="Sandeepa D.S."
            tone="bg-indigo-600"
            one="Knows when to stop trusting the ML model — with no true answer."
          />
          <ModuleCard
            n="2"
            icon={Link2}
            name="Coupling"
            owner="Dharmapala R.D."
            tone="bg-violet-600"
            one="Hands over to the numerical solver smoothly, at minimum cost."
          />
          <ModuleCard
            n="3"
            icon={SlidersHorizontal}
            name="Cost control"
            owner="Mendis B.N.D."
            tone="bg-fuchsia-600"
            one="Spends numerical effort only where it pays off, for a target accuracy."
          />
        </div>
      </div>

      {/* FOUNDATION chips */}
      <div>
        <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-3">
          Built on
        </h2>
        <div className="flex flex-wrap gap-2">
          {[
            "1D viscous Burgers benchmark",
            "FDM",
            "Cole–Hopf (exact)",
            "Spectral",
            "PINN",
            "FNO",
            "DeepONet",
            "Shared evaluation framework",
          ].map((c) => (
            <span
              key={c}
              className="text-xs font-medium text-slate-600 bg-white border border-slate-200 px-3 py-1.5 rounded-full"
            >
              {c}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
