import { useEffect, useRef, useState } from "react";
import { Card, Stat, Banner } from "../components/ui.jsx";
import { LineChart, Gauge } from "../components/Charts.jsx";
import { getMeta, buildIC, pinnIC, runTrust } from "../api.js";

const MODELS = ["FNO", "PINN", "DeepONet"];

const inactiveBtn = "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700";

// the four reference-free signals fused into the trust score
const SIGNAL_ROWS = [
  { key: "residual", label: "physics residual", desc: "breaks the PDE", color: "#4f46e5" },
  { key: "energy", label: "energy drift", desc: "gains energy", color: "#0d9488" },
  { key: "roughness", label: "roughness", desc: "goes oscillatory", color: "#e11d48" },
  { key: "momentum", label: "momentum drift", desc: "mass shifts", color: "#d97706" },
];

// one signal row: faint bar = its calibrated share of the score (always > 0 for
// the signals that matter), bright fill = how much it is contributing right now.
function SignalBar({ label, desc, color, weight, level }) {
  const share = Math.round(weight * 100);
  const active = Math.round(level * 100);
  return (
    <div className="flex items-center gap-3">
      <div className="w-32 shrink-0">
        <div className="text-xs font-medium text-slate-700 dark:text-slate-200">{label}</div>
        <div className="text-[10px] text-slate-400 dark:text-slate-500">{desc}</div>
      </div>
      <div className="flex-1 h-3 rounded-full bg-slate-100 dark:bg-slate-700/60 relative overflow-hidden">
        <div className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${share}%`, background: color, opacity: 0.22 }} />
        <div className="absolute inset-y-0 left-0 rounded-full transition-all duration-200" style={{ width: `${Math.round(weight * level * 100)}%`, background: color }} />
      </div>
      <div className="w-16 shrink-0 text-right">
        <div className="text-xs font-semibold" style={{ color }}>{share}%</div>
        <div className="text-[10px] text-slate-400 dark:text-slate-500">share</div>
      </div>
    </div>
  );
}

// the trust-score method, shown step by step in the "How it's built" tab
const BUILD_STEPS = [
  { n: 1, color: "#6366f1", title: "Read four physics signals",
    what: "From the prediction alone, compute four signals that each catch a different kind of failure.",
    chips: [
      { label: "physics residual", color: "#6366f1" },
      { label: "energy drift", color: "#14b8a6" },
      { label: "roughness", color: "#f43f5e" },
      { label: "momentum drift", color: "#f59e0b" },
    ],
    why: "The Burgers equation and conservation laws are a free, always-available truth the prediction must obey — no true answer needed." },
  { n: 2, color: "#14b8a6", title: "Normalise",
    what: "Put every signal on a common scale as a z-score.",
    detail: "z = (signal − mean) / spread",
    why: "Raw signals live on wildly different scales (residual ≈ 0.05, roughness ≈ 0.00001), so they can't be compared or added directly." },
  { n: 3, color: "#f59e0b", title: "Weight",
    what: "Weight each signal by how well it tracked the true error in training; clip negatives to 0; scale so they sum to 1.",
    detail: "wᵢ = max(0, corr(signalᵢ, true error)) ,   Σ w = 1",
    why: "Data-driven, not hand-picked — informative signals dominate, useless ones drop to 0. These are the 'shares' shown live." },
  { n: 4, color: "#8b5cf6", title: "Fuse",
    what: "Combine the weighted signals into one number.",
    detail: "fused = w₁z₁ + w₂z₂ + w₃z₃ + w₄z₄",
    why: "One number can be judged with one threshold; a linear sum stays interpretable and has nothing to overfit." },
  { n: 5, color: "#3b82f6", title: "Calibrate → trust 0–1",
    what: "Map the fused number through a fitted logistic curve to a 0–1 trust score.",
    detail: "trust = 1 − sigmoid(a · fused + b)",
    why: "Turns an arbitrary number into an interpretable 'probability it's still fine', so one cutoff behaves consistently." },
  { n: 6, color: "#10b981", title: "Switch",
    what: "When trust stays below the cutoff for K steps in a row, hand over to the numerical solver.",
    detail: "switch if trust < cutoff for K steps",
    why: "Requiring K steps avoids false alarms; the cost is tuned so a late switch is penalised more than an early one." },
];

function HowBuilt({ cut, K }) {
  return (
    <div className="mt-5 space-y-6">
      {/* soft header banner */}
      <div className="rounded-2xl p-6 md:p-7 bg-gradient-to-r from-indigo-50 via-indigo-50 to-violet-50 dark:from-indigo-500/10 dark:via-indigo-500/10 dark:to-violet-500/10 border border-indigo-100 dark:border-indigo-500/25">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">Reference-free · no true answer is ever used</div>
        <h2 className="text-2xl font-bold mt-1 text-slate-800 dark:text-slate-100">How the trust score is built</h2>
        <p className="text-sm text-slate-600 dark:text-slate-300 mt-2 leading-relaxed">
          The four physics signals are turned into one 0–1 trust score by a short pipeline. The dials it uses — the averages, the weights, and the logistic curve — are learned once, offline, on data where the true error was known, then simply applied live from the prediction alone.
        </p>
      </div>

      {/* pipeline at a glance */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-5 py-4">
        <div className="flex flex-wrap items-center gap-y-2">
          {BUILD_STEPS.map((s, i) => (
            <div key={s.n} className="flex items-center">
              <div className="flex items-center gap-2">
                <span className="w-6 h-6 rounded-lg text-[11px] font-bold flex items-center justify-center shrink-0" style={{ background: s.color + "26", color: s.color }}>{s.n}</span>
                <span className="text-xs font-medium text-slate-600 dark:text-slate-300 whitespace-nowrap">{s.title.replace(" → trust 0–1", "")}</span>
              </div>
              {i < BUILD_STEPS.length - 1 && <span className="mx-2.5 text-slate-300 dark:text-slate-600 text-xs">→</span>}
            </div>
          ))}
        </div>
      </div>

      {/* step cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {BUILD_STEPS.map((s) => (
          <div key={s.n}
            className="group rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5 shadow-sm hover:shadow-md hover:border-slate-300 dark:hover:border-slate-600 transition flex flex-col">
            <div className="flex items-center gap-3">
              <span className="w-9 h-9 rounded-xl text-sm font-bold flex items-center justify-center shrink-0" style={{ background: s.color + "26", color: s.color }}>{s.n}</span>
              <h3 className="text-sm font-bold text-slate-800 dark:text-slate-100 leading-tight">{s.title}</h3>
            </div>
            <p className="text-sm text-slate-600 dark:text-slate-300 mt-3">{s.what}</p>
            {s.chips ? (
              <div className="mt-3 flex flex-wrap gap-2">
                {s.chips.map((cp) => (
                  <span key={cp.label} className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-semibold"
                    style={{ color: cp.color, background: cp.color + "26", border: `1px solid ${cp.color}55` }}>
                    <span className="w-1.5 h-1.5 rounded-full" style={{ background: cp.color }} />
                    {cp.label}
                  </span>
                ))}
              </div>
            ) : (
              <div className="mt-3 rounded-xl px-3 py-2.5 font-mono text-[12.5px] text-slate-700 dark:text-slate-100 text-center overflow-x-auto bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-700">{s.detail}</div>
            )}
            <div className="mt-auto pt-3">
              <span className="text-[10px] font-bold uppercase tracking-wide" style={{ color: s.color }}>Why</span>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5 leading-relaxed">{s.why}</p>
            </div>
          </div>
        ))}
      </div>

      {/* highlighted notes */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="rounded-2xl border border-emerald-200 dark:border-emerald-500/30 bg-emerald-50 dark:bg-emerald-500/10 p-5">
          <div className="text-sm font-bold text-emerald-800 dark:text-emerald-300">No true answer at run time</div>
          <p className="text-sm text-slate-600 dark:text-slate-300 mt-1 leading-relaxed">The true error is used only once, offline, to set the weights and the logistic curve. After that, the score comes purely from the prediction.</p>
        </div>
        <div className="rounded-2xl border border-teal-200 dark:border-teal-500/30 bg-teal-50 dark:bg-teal-500/10 p-5">
          <div className="text-sm font-bold text-teal-800 dark:text-teal-300">FNO's safeguard</div>
          <p className="text-sm text-slate-600 dark:text-slate-300 mt-1 leading-relaxed">FNO can drift while still looking smooth, which these signals can miss. For FNO a cheap coarse solver runs alongside as a second opinion (the "cheap-reference" mode), and the score comes from how far FNO has drifted from it.</p>
        </div>
      </div>
    </div>
  );
}

// ---- evaluation data (measured on held-out waves against the exact answer) ----
const EVAL_WORST = [
  { label: "Only physics residual", v: 0.45 },
  { label: "Only energy drift", v: -0.03 },
  { label: "Only roughness", v: -0.16 },
  { label: "Fused (all four)", v: 0.66, hi: true },
];
const EVAL_CORR = [
  { m: "PINN", residual: 0.80, energy: 0.70, roughness: 0.32, momentum: 0.87 },
  { m: "FNO", residual: 0.73, energy: 0.10, roughness: -0.16, momentum: 0.80 },
  { m: "DeepONet", residual: 0.48, energy: -0.02, roughness: 0.83, momentum: 0.79 },
];
const EVAL_AUC = [
  { label: "PINN", v: 0.964 },
  { label: "FNO", v: 0.885 },
  { label: "Both pooled", v: 0.955, hi: true },
];
const EVAL_COARSE = [
  { label: "Physics signals only", v: 0.68 },
  { label: "+ cheap coarse reference", v: 0.999, hi: true },
];

function EvalBar({ label, value, hi }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const neg = value < 0;
  return (
    <div className="flex items-center gap-3">
      <div className="w-44 shrink-0 text-xs text-slate-600 dark:text-slate-300">{label}</div>
      <div className="flex-1 h-3 rounded-full bg-slate-100 dark:bg-slate-700/70 overflow-hidden">
        <div className="h-3 rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: hi ? "linear-gradient(90deg,#34d399,#059669)" : "linear-gradient(90deg,#a5b4fc,#6366f1)" }} />
      </div>
      <div className={`w-12 text-right text-sm font-bold ${neg ? "text-rose-500" : hi ? "text-emerald-600 dark:text-emerald-400" : "text-slate-600 dark:text-slate-300"}`}>{value.toFixed(2)}</div>
    </div>
  );
}

function StatTile({ value, label, hint, color }) {
  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4 shadow-sm">
      <div className="text-[28px] leading-none font-extrabold" style={{ color }}>{value}</div>
      <div className="text-xs font-semibold text-slate-700 dark:text-slate-200 mt-2">{label}</div>
      <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">{hint}</div>
    </div>
  );
}

function corrTone(v) {
  if (v >= 0.6) return "bg-emerald-100 text-emerald-800 dark:bg-emerald-500/20 dark:text-emerald-300";
  if (v >= 0.3) return "bg-amber-100 text-amber-800 dark:bg-amber-500/20 dark:text-amber-300";
  if (v >= 0) return "bg-slate-100 text-slate-500 dark:bg-slate-700 dark:text-slate-400";
  return "bg-rose-100 text-rose-700 dark:bg-rose-500/20 dark:text-rose-300";
}

function Evaluation() {
  return (
    <div className="mt-5 space-y-5">
      {/* header */}
      <div className="rounded-2xl p-6 md:p-7 bg-gradient-to-r from-indigo-50 via-indigo-50 to-violet-50 dark:from-indigo-500/10 dark:via-indigo-500/10 dark:to-violet-500/10 border border-indigo-100 dark:border-indigo-500/25">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">How the module is evaluated</div>
        <h2 className="text-2xl font-bold mt-1 text-slate-800 dark:text-slate-100">Does the trust score actually work?</h2>
        <p className="text-sm text-slate-600 dark:text-slate-300 mt-2 leading-relaxed">
          Every result below is measured on held-out waves against the exact answer. The module still uses no
          true answer while running — the exact answer is only used here, offline, to check whether it was right.
        </p>
      </div>

      {/* headline numbers */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatTile value="0.955" label="Detection quality (AUC)" hint="0.5 = guessing · 1 = perfect" color="#6366f1" />
        <StatTile value="0.66" label="Fused worst-case match" hint="robust across all three models" color="#14b8a6" />
        <StatTile value="0.999" label="Coarse catches the drift" hint="signals alone: 0.68" color="#8b5cf6" />
        <StatTile value="±0.28" label="Reliable-horizon error" hint="in time · correlation 0.85" color="#10b981" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* why fuse */}
        <Card title="Why we fuse the signals"
          subtitle="worst-case match with the true error across the three models (−1…1 · higher is better)">
          <div className="space-y-2.5 mt-1">
            {EVAL_WORST.map((r) => <EvalBar key={r.label} label={r.label} value={r.v} hi={r.hi} />)}
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-3">No single signal is dependable for every model — some even go negative. Fusing gives one calibrated score (never below <b>0.66</b>) that works without knowing which signal will fire.</p>
        </Card>

        {/* per-model correlation */}
        <Card title="Each model leans on a different signal"
          subtitle="how well each signal tracks the true error, per model">
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-slate-400 dark:text-slate-500">
                  <th className="text-left py-1 font-medium">Model</th>
                  <th className="py-1 font-medium">residual</th>
                  <th className="py-1 font-medium">energy</th>
                  <th className="py-1 font-medium">roughness</th>
                  <th className="py-1 font-medium">momentum</th>
                </tr>
              </thead>
              <tbody>
                {EVAL_CORR.map((r) => (
                  <tr key={r.m}>
                    <td className="py-1.5 font-semibold text-slate-700 dark:text-slate-200">{r.m}</td>
                    <td className="p-1"><div className={`rounded-md py-1 text-center font-semibold ${corrTone(r.residual)}`}>{r.residual.toFixed(2)}</div></td>
                    <td className="p-1"><div className={`rounded-md py-1 text-center font-semibold ${corrTone(r.energy)}`}>{r.energy.toFixed(2)}</div></td>
                    <td className="p-1"><div className={`rounded-md py-1 text-center font-semibold ${corrTone(r.roughness)}`}>{r.roughness.toFixed(2)}</div></td>
                    <td className="p-1"><div className={`rounded-md py-1 text-center font-semibold ${corrTone(r.momentum)}`}>{r.momentum.toFixed(2)}</div></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-3">Residual and momentum are the consistent backbone; roughness is DeepONet's key signal and energy is PINN's — so the learned weights adapt to each model.</p>
        </Card>

        {/* AUC */}
        <Card title="It separates good from failed predictions"
          subtitle="detection quality (ROC AUC): 0.5 = guessing, 1.0 = perfect">
          <div className="space-y-2.5 mt-1">
            {EVAL_AUC.map((r) => <EvalBar key={r.label} label={r.label} value={r.v} hi={r.hi} />)}
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-3">Pooled score is <b>0.955</b> — very reliable — with a low false-alarm rate (PINN 0%, FNO 11%, DeepONet 0%).</p>
        </Card>

        {/* coarse reference */}
        <Card title="The cheap reference closes the blind spot"
          subtitle="match with the true error on FNO's smooth-drift failure">
          <div className="space-y-2.5 mt-1">
            {EVAL_COARSE.map((r) => <EvalBar key={r.label} label={r.label} value={r.v} hi={r.hi} />)}
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-3">The physics signals miss FNO's smooth drift (0.68), but the cheap coarse solver catches it almost perfectly (<b>0.999</b>) and fires right at the failure.</p>
        </Card>
      </div>
    </div>
  );
}

export default function TrustPage() {
  const [meta, setMeta] = useState(null);
  const [model, setModel] = useState("FNO");
  const [modes, setModes] = useState(2);
  const [amplitude, setAmplitude] = useState(0.4);
  const [pinnIndex, setPinnIndex] = useState(0);
  const [fnoMode, setFnoMode] = useState("coarse");
  const [ic, setIc] = useState(null);
  const [frame, setFrame] = useState(null);
  const [hist, setHist] = useState([]);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState(null);
  const [tab, setTab] = useState("demo");
  const wsRef = useRef(null);

  useEffect(() => { getMeta().then(setMeta).catch(() => setErr("Backend not reachable. Start it with: uvicorn main:app")); }, []);

  useEffect(() => {
    if (!meta) return;
    if (model === "PINN") pinnIC(pinnIndex).then((d) => setIc(d.ic)).catch(() => {});
    else buildIC(modes, amplitude).then((d) => setIc(d.ic)).catch(() => {});
  }, [meta, model, modes, amplitude, pinnIndex]);

  function run() {
    if (wsRef.current) wsRef.current.close();
    setHist([]); setFrame(null); setErr(null); setRunning(true);
    const payload = model === "PINN"
      ? { model, pinn_index: pinnIndex }
      : { model, ic, mode: model === "FNO" ? fnoMode : "reference_free" };
    wsRef.current = runTrust(payload,
      (f) => { setFrame(f); setHist((h) => [...h, f]); },
      () => setRunning(false),
      (e) => { setErr(e); setRunning(false); });
  }

  const x = meta?.x || [];
  const p = meta?.params?.[model];
  // In cheap-reference mode the switch is governed by the coarse solver's own
  // rule (estimated error > 10% for 3 steps), which maps to trust < 0.5 — not
  // the reference-free calibration. Show the threshold that actually applies.
  const coarseMode = model === "FNO" && fnoMode === "coarse";
  const cut = coarseMode ? 0.5 : (p?.CUT ?? 0.5);
  const kSteps = coarseMode ? 3 : (p?.K ?? 4);
  const ood = model !== "PINN" && modes > 4;
  const muted = "text-slate-500 dark:text-slate-400";
  const faint = "text-slate-400 dark:text-slate-500";

  return (
    <div>
      <h1 className="text-2xl font-bold text-slate-800 dark:text-slate-100">Trust Score — live demo</h1>
      <p className="text-slate-600 dark:text-slate-300 mt-1 text-sm">
        Give a starting wave, run the ML model, and watch the trust score fall and the switch fire — all with no true answer used by the module.
      </p>
      {err && <div className="mt-3 text-sm text-rose-600 dark:text-rose-300 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 rounded-lg px-3 py-2">{err}</div>}

      {/* TABS */}
      <div className="mt-4 inline-flex rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-1">
        {[["demo", "Live demo"], ["built", "How it's built"], ["eval", "Evaluation"]].map(([id, label]) => (
          <button key={id} onClick={() => setTab(id)}
            className={`px-4 py-1.5 rounded-lg text-sm font-medium transition ${tab === id ? "bg-indigo-600 text-white" : "text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"}`}>{label}</button>
        ))}
      </div>

      {tab === "demo" && (
      <div className="mt-5 grid grid-cols-[320px_1fr] gap-5">
        {/* CONTROLS */}
        <div className="space-y-4">
          <Card title="1. Choose the ML model">
            <div className="flex gap-2">
              {MODELS.map((m) => (
                <button key={m} onClick={() => setModel(m)}
                  className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium border ${model === m
                    ? "bg-indigo-600 text-white border-indigo-600" : inactiveBtn}`}>
                  {m}
                </button>
              ))}
            </div>
          </Card>

          <Card title="2. Set the initial condition"
            subtitle={model === "PINN"
              ? "PINN is trained per wave — pick one of its trained inputs."
              : "Build a wave. More modes = sharper"}>
            {model === "PINN" ? (
              <select value={pinnIndex} onChange={(e) => setPinnIndex(+e.target.value)}
                className="w-full bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-2 text-sm">
                {Array.from({ length: meta?.n_pinn_ics || 0 }, (_, i) => (
                  <option key={i} value={i}>Trained wave #{i}</option>
                ))}
              </select>
            ) : (
              <div className="space-y-3">
                <div>
                  <div className={`flex justify-between text-xs ${muted}`}><span>Sine modes</span><span>{modes}</span></div>
                  <input type="range" min="1" max="4" value={modes} onChange={(e) => setModes(+e.target.value)} className="w-full" />
                  {ood && <div className="text-xs text-amber-600 dark:text-amber-400 mt-1">Above 4 modes = out-of-distribution (unfamiliar) input</div>}
                </div>
                <div>
                  <div className={`flex justify-between text-xs ${muted}`}><span>Amplitude</span><span>{amplitude.toFixed(1)}</span></div>
                  <input type="range" min="0.5" max="1.5" step="0.1" value={amplitude} onChange={(e) => setAmplitude(+e.target.value)} className="w-full" />
                </div>
              </div>
            )}
            {model === "FNO" && (
              <div className="mt-3">
                <div className={`text-xs mb-1 ${muted}`}>Trust mode</div>
                <div className="flex gap-2">
                  {[["coarse", "Cheap-reference"], ["reference_free", "Reference-free"]].map(([v, l]) => (
                    <button key={v} onClick={() => setFnoMode(v)}
                      className={`flex-1 px-2 py-1.5 rounded-lg text-xs border ${fnoMode === v
                        ? "bg-slate-800 dark:bg-slate-600 text-white border-slate-800 dark:border-slate-600" : inactiveBtn}`}>{l}</button>
                  ))}
                </div>
                {fnoMode === "reference_free" && (
                  <div className="text-xs text-amber-600 dark:text-amber-400 mt-1">
                    FNO drifts smoothly — reference-free signals can miss it, so it may not switch. Cheap-reference catches it.
                  </div>
                )}
              </div>
            )}
            {ic && (
              <div className="mt-3">
                <div className={`text-xs mb-1 ${faint}`}>Starting wave preview</div>
                <LineChart series={[{ x, y: ic, color: "#6366f1", width: 2 }]} xr={[-1, 1]} yr={[-1.6, 1.6]} h={130} xlabel="x" />
              </div>
            )}
          </Card>

          <button onClick={run} disabled={running || !ic}
            className="w-full px-4 py-2.5 rounded-lg bg-emerald-600 text-white text-sm font-semibold hover:bg-emerald-700 disabled:opacity-50">
            {running ? "Running…" : "▶ Run simulation"}
          </button>
        </div>

        {/* RESULTS */}
        <div className="space-y-4">
          <Banner ok={!frame || frame.ok}
            text={!frame ? "Ready — press run" : frame.ok
              ? `Using the ML model (fast) — t = ${frame.t.toFixed(2)}`
              : `Switched to the numerical solver at t = ${(frame.switch_t ?? 0).toFixed(2)}`} />

          <div className="grid grid-cols-2 gap-4">
            <Card title="Solution wave u(x, t)" subtitle={frame ? `time t = ${frame.t.toFixed(2)}` : "—"}>
              <LineChart h={190} xr={[-1, 1]} yr={[-1.6, 1.6]} xlabel="x"
                series={frame ? [
                  { x, y: frame.true, color: "#94a3b8", dashed: true, width: 1.5 },
                  { x, y: frame.u, color: frame.ok ? "#059669" : "#e11d48", width: 2.5 },
                ] : []} />
              <div className={`text-xs mt-1 ${faint}`}>green/red = ML prediction · grey dashed = true answer</div>
            </Card>

            <Card title="Trust score">
              <Gauge value={frame ? frame.trust : 1} />
              <div className="grid grid-cols-2 gap-2 mt-3">
                <Stat label="switch fired at" value={frame?.switch_t != null ? frame.switch_t.toFixed(2) : "—"} tone="red" />
                <Stat label="true error now" value={frame ? `${Math.round(frame.true_error * 100)}%` : "—"} />
              </div>
            </Card>
          </div>

          <Card title="Trust and true error over time" subtitle={coarseMode
            ? `cheap reference: switch fires when trust stays below ${cut} for ${kSteps} steps (est. error > 10%)`
            : `switch fires when trust stays below ${cut} for ${kSteps} steps`}>
            <LineChart h={200} xr={[0, 2]} yr={[0, 1]} hline={coarseMode ? null : cut} vline={frame?.switch_t ?? null} xlabel="time t"
              series={[
                { x: hist.map((f) => f.t), y: hist.map((f) => f.trust), color: "#4f46e5", width: 2.5 },
                { x: hist.map((f) => f.t), y: hist.map((f) => Math.min(1, f.true_error)), color: "#94a3b8", dashed: true, width: 1.5 },
              ]} />
            <div className={`text-xs mt-1 ${faint}`}>indigo = trust · grey dashed = true error{coarseMode ? "" : " · dashed line = cutoff"} · red line = switch</div>
          </Card>

          <Card title="How the trust score is built (live)">
            {frame && frame.weights ? (
              <div className="text-sm text-slate-600 dark:text-slate-300 space-y-3">
                <p className={`text-xs ${muted}`}>
                  The score fuses four reference-free signals. Each bar shows a signal's
                  <b> calibrated share</b> of the score (faint) and how much it's
                  <b> driving distrust right now</b> (bright fill).
                </p>
                <div className="space-y-2.5">
                  {SIGNAL_ROWS.map((s) => (
                    <SignalBar
                      key={s.key}
                      {...s}
                      weight={frame.weights[s.key] ?? 0}
                      level={frame.levels?.[s.key] ?? 0}
                    />
                  ))}
                </div>
                <p className={`text-[11px] ${muted}`}>
                  Shares are fixed per model by calibration — a share of 0 means that
                  signal wasn't informative for this model, so it isn't relied on.
                </p>
                <p>
                  {model === "FNO" && fnoMode === "coarse"
                    ? "Cheap-reference mode: a small coarse solver runs alongside FNO and the trust score comes from how far FNO has drifted from it."
                    : "The fused number is calibrated to a 0–1 trust score. When it stays below the cutoff for a few steps in a row, the switch latches on."}
                </p>
                <p className={muted}>cutoff = {cut} · patience K = {kSteps} · fail tolerance = 10% error.</p>
              </div>
            ) : <p className={`text-sm ${faint}`}>Run a simulation to see the live signal breakdown.</p>}
          </Card>
        </div>
      </div>
      )}

      {tab === "built" && <HowBuilt cut={p?.CUT ?? 0.5} K={p?.K ?? 4} />}

      {tab === "eval" && <Evaluation />}
    </div>
  );
}
