import { useEffect, useState } from "react";
import {
  Zap, ShieldCheck, GitMerge, Gauge, Link2, SlidersHorizontal, ArrowRight,
  AlertTriangle, ArrowLeftRight, Rocket, Layers, CheckCircle2, Table2,
} from "lucide-react";
import DatasetModal from "../components/DatasetModal.jsx";

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
  const W = 380, H = 230, padX = 14, padT = 26, padB = 20, xs = 0.6, fail = 0.1;
  const sx = (x) => padX + x * (W - 2 * padX);
  const sy = (e) => H - padB - Math.min(1, e) * (H - padT - padB);
  const mlErr = (x) => Math.min(1, 0.03 + 4.4 * Math.max(0, x - 0.4) ** 2);
  const hyErr = (x) => (x < xs ? mlErr(x) : 0.05);
  const stroke = (f) => {
    const pts = [];
    for (let x = 0; x <= p; x += 0.02) pts.push(`${x ? "L" : "M"}${sx(x).toFixed(1)} ${sy(f(x)).toFixed(1)}`);
    return pts.join(" ");
  };
  const area = (f) => {
    let d = `M${sx(0).toFixed(1)} ${sy(0).toFixed(1)} `;
    for (let x = 0; x <= p; x += 0.02) d += `L${sx(x).toFixed(1)} ${sy(f(x)).toFixed(1)} `;
    return d + `L${sx(Math.min(p, 1)).toFixed(1)} ${sy(0).toFixed(1)} Z`;
  };
  const switched = p >= xs;
  const head = (f) => <circle cx={sx(Math.min(p, 1))} cy={sy(f(Math.min(p, 1)))} r="4" fill="#fff" stroke={f === mlErr ? "#e11d48" : f === hyErr ? "#059669" : "#64748b"} strokeWidth="2.5" />;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      <defs>
        <linearGradient id="mlG" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#e11d48" stopOpacity="0.28" />
          <stop offset="100%" stopColor="#e11d48" stopOpacity="0" />
        </linearGradient>
        <linearGradient id="hyG" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#059669" stopOpacity="0.22" />
          <stop offset="100%" stopColor="#059669" stopOpacity="0" />
        </linearGradient>
      </defs>
      {/* trained window */}
      <rect x={sx(0)} y={padT} width={sx(0.5) - sx(0)} height={H - padT - padB} fill="#6366f1" fillOpacity="0.06" />
      <text x={sx(0.25)} y={padT - 8} textAnchor="middle" fontSize="9" fill="#818cf8" fontWeight="600">trained window</text>
      {/* gridlines */}
      {[0.25, 0.5, 0.75].map((g) => <line key={g} x1={sx(0)} x2={sx(1)} y1={sy(g)} y2={sy(g)} stroke="var(--chart-grid)" strokeWidth="1" />)}
      {/* fail threshold */}
      <line x1={sx(0)} x2={sx(1)} y1={sy(fail)} y2={sy(fail)} stroke="#f59e0b" strokeWidth="1" strokeDasharray="3 3" />
      <text x={sx(0)+2} y={sy(fail) - 4} fontSize="8.5" fill="#d97706" fontWeight="600">fail line (10%)</text>
      {/* areas */}
      <path d={area(mlErr)} fill="url(#mlG)" />
      <path d={area(hyErr)} fill="url(#hyG)" />
      {/* switch marker */}
      {switched && <>
        <line x1={sx(xs)} x2={sx(xs)} y1={padT} y2={H - padB} stroke="#e11d48" strokeWidth="1.3" strokeDasharray="4 3" />
        <rect x={sx(xs) - 22} y={padT - 2} width="44" height="15" rx="7" fill="#e11d48" />
        <text x={sx(xs)} y={padT + 8.5} textAnchor="middle" fontSize="8.5" fill="#fff" fontWeight="700">switch</text>
      </>}
      {/* lines */}
      <path d={stroke(() => 0.02)} fill="none" stroke="#94a3b8" strokeWidth="2" strokeLinecap="round" />
      <path d={stroke(mlErr)} fill="none" stroke="#e11d48" strokeWidth="2.5" strokeLinecap="round" />
      <path d={stroke(hyErr)} fill="none" stroke="#059669" strokeWidth="3" strokeLinecap="round" />
      {head(() => 0.02)}{head(mlErr)}{head(hyErr)}
      {/* labels */}
      <text x={sx(1) - 2} y={sy(mlErr(1)) + 3} textAnchor="end" fontSize="10" fill="#e11d48" fontWeight="700">pure ML</text>
      <text x={sx(1) - 2} y={sy(0.02) - 7} textAnchor="end" fontSize="9.5" fill="#94a3b8" fontWeight="600">numerical</text>
      <text x={sx(0.72)} y={sy(0.05) + 15} fontSize="10" fill="#059669" fontWeight="700">hybrid</text>
    </svg>
  );
}

const cardCls = "bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700";

function Group({ n, title, items }) {
  return (
    <div className={`${cardCls} p-4 hover:shadow-md transition-shadow`}>
      <div className="flex items-baseline gap-2">
        <span className="text-xl font-bold text-indigo-600 dark:text-indigo-400">{n}</span>
        <span className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide">{title}</span>
      </div>
      <div className="flex flex-wrap gap-1.5 mt-2.5">
        {items.map((i) => (
          <span key={i} className="text-xs font-medium text-indigo-700 dark:text-indigo-300 bg-indigo-50 dark:bg-indigo-500/15 px-2 py-1 rounded-lg">{i}</span>
        ))}
      </div>
    </div>
  );
}

function Family({ icon: Icon, color, title, tag }) {
  return (
    <div className={`${cardCls} p-4 flex items-start gap-3 hover:shadow-md transition-shadow`}>
      <div className={`w-10 h-10 rounded-xl grid place-items-center ${color}`}><Icon size={20} /></div>
      <div>
        <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">{title}</div>
        <div className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{tag}</div>
      </div>
    </div>
  );
}

function ModuleCard({ n, icon: Icon, name, owner, one, tone }) {
  return (
    <div className={`${cardCls} p-4 hover:shadow-md hover:-translate-y-0.5 transition-all`}>
      <div className="flex items-center gap-2.5">
        <div className={`w-9 h-9 rounded-xl grid place-items-center text-white ${tone}`}><Icon size={18} /></div>
        <div>
          <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">{n} · {name}</div>
          <div className="text-xs text-slate-500 dark:text-slate-400">{owner}</div>
        </div>
      </div>
      <p className="text-xs text-slate-500 dark:text-slate-400 mt-3">{one}</p>
    </div>
  );
}

function StatTile({ value, label }) {
  return (
    <div className={`${cardCls} p-4 text-center`}>
      <div className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">{value}</div>
      <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">{label}</div>
    </div>
  );
}

function Stage({ icon: Icon, chip, step, title, sub }) {
  return (
    <div className="flex flex-col items-center text-center w-[112px]">
      <div className={`relative w-14 h-14 rounded-2xl grid place-items-center ring-4 shadow-sm ${chip}`}>
        <Icon size={22} />
        {step && (
          <span className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-white dark:bg-slate-800 text-[11px] font-bold grid place-items-center text-slate-700 dark:text-slate-200 border border-slate-200 dark:border-slate-600">
            {step}
          </span>
        )}
      </div>
      <div className="mt-2.5 text-sm font-semibold text-slate-800 dark:text-slate-100">{title}</div>
      <div className="text-[11px] text-slate-500 dark:text-slate-400">{sub}</div>
    </div>
  );
}

const H2 = ({ children }) => (
  <h2 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3">{children}</h2>
);

export default function Overview({ go }) {
  const [showDataset, setShowDataset] = useState(false);
  return (
    <div className="space-y-8">
      <DatasetModal open={showDataset} onClose={() => setShowDataset(false)} />
      {/* RESEARCH TOPIC */}
      <div className="text-center rounded-3xl bg-gradient-to-b from-indigo-50 dark:from-indigo-500/10 to-transparent border border-indigo-100 dark:border-indigo-500/20 px-6 pt-7 pb-8">
        <div className="inline-flex items-center gap-2 text-xs font-medium text-indigo-700 dark:text-indigo-300 bg-indigo-50 dark:bg-indigo-500/15 px-3 py-1 rounded-full">
          Team Turingz · University of Moratuwa
        </div>
        <div className="text-2xl md:text-4xl font-extrabold text-slate-800 dark:text-slate-100 leading-[1.12] mt-4 max-w-4xl mx-auto tracking-tight">
          Hybrid Machine Learning and Numerical Methods for
          <span className="text-indigo-600 dark:text-indigo-400"> PDE Extrapolation</span>
        </div>
        <p className="text-slate-500 dark:text-slate-400 mt-3 text-base">
          A trust-gated, cost-aware hybrid solver for long-horizon prediction
        </p>
      </div>

      {/* HERO */}
      <div className="grid grid-cols-[1fr_420px] gap-8 items-center">
        <div>
          <div className="inline-flex items-center gap-1.5 text-[11px] font-semibold text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">
            <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" /> The idea
          </div>
          <h2 className="text-3xl font-extrabold text-slate-800 dark:text-slate-100 leading-tight mt-2">
            Fast where it can be trusted.<br />
            <span className="bg-gradient-to-r from-indigo-600 to-fuchsia-600 dark:from-indigo-400 dark:to-fuchsia-400 bg-clip-text text-transparent">
              Exact where it cannot.
            </span>
          </h2>
          <p className="text-slate-600 dark:text-slate-300 mt-4 max-w-md leading-relaxed">
            The solver runs a fast ML model and switches to an accurate numerical solver the moment the ML is about to fail.
          </p>
          <div className="flex flex-wrap gap-2 mt-4">
            <span className="text-xs font-medium px-2.5 py-1 rounded-full bg-rose-50 dark:bg-rose-500/10 text-rose-600 dark:text-rose-400">ML diverges</span>
            <span className="text-xs font-medium px-2.5 py-1 rounded-full bg-emerald-50 dark:bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">hybrid stays accurate</span>
          </div>
          <button onClick={() => go("hybrid")}
            className="mt-6 inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-indigo-600 text-white text-sm font-semibold shadow-sm shadow-indigo-500/30 hover:bg-indigo-700 hover:-translate-y-0.5 transition-all">
            See it live <ArrowRight size={16} />
          </button>
        </div>
        <div className="relative">
          <div className="absolute -inset-4 rounded-3xl blur-2xl opacity-60 bg-gradient-to-tr from-indigo-300/40 via-fuchsia-200/30 to-emerald-200/40 dark:from-indigo-500/20 dark:via-fuchsia-500/10 dark:to-emerald-500/15" />
          <div className={`relative ${cardCls} p-4 shadow-lg`}>
            <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1 px-1">Error over time</div>
            <ConceptChart />
            <div className="flex items-center justify-center gap-4 mt-1 text-[11px] text-slate-500 dark:text-slate-400">
              <span className="flex items-center gap-1"><span className="w-2.5 h-1 rounded-full bg-rose-500" /> pure ML</span>
              <span className="flex items-center gap-1"><span className="w-2.5 h-1 rounded-full bg-slate-400" /> numerical</span>
              <span className="flex items-center gap-1"><span className="w-2.5 h-1 rounded-full bg-emerald-600" /> hybrid</span>
            </div>
          </div>
        </div>
      </div>

      {/* BENCHMARK + GAP */}
      <div className="grid grid-cols-[320px_1fr] gap-4">
        <div className={`${cardCls} p-5 flex flex-col justify-center`}>
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide">Benchmark equation</div>
          <div
            className="text-2xl text-slate-800 dark:text-slate-100 mt-2"
            style={{ fontFamily: "Cambria, Georgia, serif" }}
          >
            <math
              xmlns="http://www.w3.org/1998/Math/MathML"
              display="inline"
            >
              <mrow>
                {/* ∂u / ∂t */}
                <mfrac>
                  <mrow>
                    <mo>∂</mo>
                    <mi>u</mi>
                  </mrow>
                  <mrow>
                    <mo>∂</mo>
                    <mi>t</mi>
                  </mrow>
                </mfrac>

                {/* + */}
                <mo>+</mo>

                {/* u · */}
                <mi>u</mi>
                <mo>·</mo>

                {/* ∂u / ∂x */}
                <mfrac>
                  <mrow>
                    <mo>∂</mo>
                    <mi>u</mi>
                  </mrow>
                  <mrow>
                    <mo>∂</mo>
                    <mi>x</mi>
                  </mrow>
                </mfrac>

                {/* = */}
                <mo>=</mo>

                {/* ν · */}
                <mi>ν</mi>
                <mo>·</mo>

                {/* ∂²u / ∂x² */}
                <mfrac>
                  <mrow>
                    <msup>
                      <mo>∂</mo>
                      <mn>2</mn>
                    </msup>
                    <mi>u</mi>
                  </mrow>
                  <mrow>
                    <mo>∂</mo>
                    <msup>
                      <mi>x</mi>
                      <mn>2</mn>
                    </msup>
                  </mrow>
                </mfrac>
              </mrow>
            </math>
          </div>
          <div className="text-xs text-slate-500 dark:text-slate-400 mt-2">
            1D viscous Burgers — forms a moving <span className="font-medium text-slate-700 dark:text-slate-200">shock</span>, ideal for stress-testing extrapolation.
          </div>
        </div>
        <div className="bg-rose-50 dark:bg-rose-500/10 rounded-2xl border border-rose-100 dark:border-rose-500/20 p-5 flex flex-col justify-center">
          <div className="text-xs font-semibold text-rose-500 dark:text-rose-400 uppercase tracking-wide">The gap we exploit</div>
          <p className="text-sm text-slate-700 dark:text-slate-200 mt-2">
            Beyond their training window, ML solvers drift and <span className="font-semibold text-rose-600 dark:text-rose-400">fail silently</span> — and at
            deployment there is <span className="font-semibold">no true answer</span> to catch it. The hybrid must decide, on its own, when to trust ML and when to compute.
          </p>
        </div>
      </div>

      {/* THE DATASET */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <H2>The dataset</H2>
          <button onClick={() => setShowDataset(true)}
            className="inline-flex items-center gap-1.5 text-xs font-semibold cursor-pointer text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-500/15 px-3 py-1.5 rounded-lg hover:bg-indigo-100 dark:hover:bg-indigo-500/25 transition">
            <Table2 size={14} /> View dataset
          </button>
        </div>
        <div className="grid grid-cols-6 gap-3">
          <StatTile value="1000" label="initial conditions" />
          <StatTile value="200" label="timesteps each" />
          <StatTile value="512" label="spatial points" />
          <StatTile value="102.4M" label="data points total" />
          <StatTile value="[-1, 1]" label="space domain (x)" />
          <StatTile value="[0, 2]" label="time domain (t)" />
        </div>
      </div>

      {/* WHAT WE USED */}
      <div className="grid grid-cols-4 gap-3">
        <Group n="3" title="Numerical solvers" items={["FDM", "Cole–Hopf", "Spectral"]} />
        <Group n="3" title="ML surrogates" items={["PINN", "FNO", "DeepONet"]} />
        <Group n="3" title="Method modules" items={["Trust", "Coupling", "Cost control"]} />
        <Group n="1" title="Hybrid engine" items={["Tunable runtime"]} />
      </div>

      {/* FAMILIES */}
      <div>
        <H2>Two worlds, one solver</H2>
        <div className="grid grid-cols-3 gap-4">
          <Family icon={Zap} color="bg-amber-100 dark:bg-amber-500/20 text-amber-600 dark:text-amber-400" title="ML solvers" tag="Fast · fail silently later" />
          <Family icon={ShieldCheck} color="bg-slate-100 dark:bg-slate-600 text-slate-600 dark:text-slate-200" title="Numerical solvers" tag="Accurate · slow" />
          <Family icon={GitMerge} color="bg-emerald-100 dark:bg-emerald-500/20 text-emerald-600 dark:text-emerald-400" title="Our hybrid" tag="Fast and accurate" />
        </div>
      </div>

      {/* PIPELINE */}
      <div>
        <H2>How it works</H2>
        <div className={`${cardCls} px-8 py-7`}>
          <div className="relative">
            <div className="absolute top-7 left-[9%] right-[9%] h-0.5 rounded-full opacity-70 bg-gradient-to-r from-slate-300 via-violet-400 to-emerald-400 dark:from-slate-600 dark:via-violet-500 dark:to-emerald-500" />
            <div className="relative flex justify-between">
              <Stage icon={Layers} title="Foundation" sub="data · solvers · ML"
                chip="bg-slate-100 text-slate-600 ring-slate-100 dark:bg-slate-700 dark:text-slate-200 dark:ring-slate-700" />
              <Stage icon={Gauge} step="1" title="Trust" sub="when to switch"
                chip="bg-indigo-600 text-white ring-indigo-100 dark:ring-indigo-500/30" />
              <Stage icon={Link2} step="2" title="Coupling" sub="how to correct"
                chip="bg-violet-600 text-white ring-violet-100 dark:ring-violet-500/30" />
              <Stage icon={SlidersHorizontal} step="3" title="Control" sub="how much effort"
                chip="bg-fuchsia-600 text-white ring-fuchsia-100 dark:ring-fuchsia-500/30" />
              <Stage icon={CheckCircle2} title="Output" sub="fast + accurate"
                chip="bg-emerald-500 text-white ring-emerald-100 dark:ring-emerald-500/30" />
            </div>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 text-center mt-6 max-w-2xl mx-auto">
            The ML model runs while <span className="font-medium text-indigo-600 dark:text-indigo-400">Trust</span> watches it. When trust drops,
            <span className="font-medium text-violet-600 dark:text-violet-400"> Coupling</span> injects numerical correction and
            <span className="font-medium text-fuchsia-600 dark:text-fuchsia-400"> Control</span> keeps the cost minimal — producing one fast, accurate solution.
          </p>
        </div>
      </div>

      {/* MODULES */}
      <div>
        <H2>Three research modules</H2>
        <div className="grid grid-cols-3 gap-4">
          <ModuleCard n="1" icon={Gauge} name="Trust" owner="Sandeepa D.S." tone="bg-indigo-600"
            one="Knows when to stop trusting the ML model — with no true answer." />
          <ModuleCard n="2" icon={Link2} name="Coupling" owner="Dharmapala R.D." tone="bg-violet-600"
            one="Hands over to the numerical solver smoothly, at minimum cost." />
          <ModuleCard n="3" icon={SlidersHorizontal} name="Cost control" owner="Mendis B.N.D." tone="bg-fuchsia-600"
            one="Spends numerical effort only where it pays off, for a target accuracy." />
        </div>
      </div>

      {/* WHAT WE DEMONSTRATE */}
      <div>
        <H2>What the live demo shows</H2>
        <div className="grid grid-cols-3 gap-4">
          <Family icon={AlertTriangle} color="bg-rose-100 dark:bg-rose-500/20 text-rose-600 dark:text-rose-400" title="Catches silent failures" tag="no true answer needed" />
          <Family icon={ArrowLeftRight} color="bg-indigo-100 dark:bg-indigo-500/20 text-indigo-600 dark:text-indigo-400" title="Switches at the right moment" tag="trust flips the engine live" />
          <Family icon={Rocket} color="bg-emerald-100 dark:bg-emerald-500/20 text-emerald-600 dark:text-emerald-400" title="Fast + accurate long-horizon" tag="cheaper than pure numerical" />
        </div>
      </div>

      {/* FOUNDATION chips */}
      <div>
        <H2>Built on</H2>
        <div className="flex flex-wrap gap-2">
          {["1D viscous Burgers benchmark", "FDM", "Cole–Hopf (exact)", "Spectral", "PINN", "FNO", "DeepONet", "Shared evaluation framework"].map((c) => (
            <span key={c} className="text-xs font-medium text-slate-600 dark:text-slate-300 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 px-3 py-1.5 rounded-full">{c}</span>
          ))}
        </div>
      </div>
    </div>
  );
}
