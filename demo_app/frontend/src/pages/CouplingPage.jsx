import { useEffect, useRef, useState } from "react";
import { Card, Stat, Banner, sci10, eToSup } from "../components/ui.jsx";
import { LineChart } from "../components/Charts.jsx";
import { getMeta, buildIC, pinnIC, realTestIC, runCoupling } from "../api.js";
import {
  RL_XS, RL_T, RL_FRAMES, RL_ERR_ML, RL_SWITCHES, RL_META,
} from "../couplingData.js";
import { Play, GitCommitHorizontal } from "lucide-react";
// committed source of truth: written by make_figures.py from the 100 held-out predictions
import sweep from "../../../../results/module2/figures/handoff_sweep_results.json";
// committed final-integration comparison (trust-triggered hard switch vs baselines)
import cmp from "../../../../results/module2/figures/trust_hardswitch_compare.json";
// committed model-agnostic transfer results (same coupling, FNO/PINN/DeepONet)
import transfer from "../../../../results/module2/figures/transfer_models_results.json";
// committed continuity/stability diagnostic (state jump, residuals across the switch)
import stab from "../../../../results/module2/figures/handoff_stability_diagnostic.json";

/* =====================================================================
   Module 2 · THE BATON PASS, a relay between two solvers.
   The page is built around one interaction: YOU drag the handoff point.
   ===================================================================== */

const MODELS = ["FNO", "DeepONet", "PINN"];
const inactiveBtn = "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700";

/* Headline numbers computed live from the committed handoff_sweep_results.json
   (make_figures.py, full 100-IC held-out set), nothing here is hand-typed. */
const _AGG_R0 = sweep.results.find((r) => Math.abs(r.t_s - 1.0) < 1e-9) || sweep.results[0];
const AGG = {
  fnoTail1: (_AGG_R0.fno_tail * 100).toFixed(1),                  // 13.4
  fnoTail2: (_AGG_R0.fno_tail * 100).toFixed(2),                  // 13.41
  hybTail2: (_AGG_R0.hybrid_tail * 100).toFixed(2),              // 0.86
  stateErr: (_AGG_R0.e_s * 100).toFixed(2),                      // 1.02 (ML state error at hand-off)
  reduction: (_AGG_R0.benefit * 100).toFixed(1),                 // 92.9
  icsImproved: _AGG_R0.ics_improved,                             // 100
  nIC: sweep.n_ic,                                               // 100
  boundary: Number(sweep.viability_rule.boundary_t_s).toFixed(2), // 1.49
};

/* Committed oracle (true-state restart) tail error at t_s = 1.0, from the same sweep. */
const _ORACLE = _AGG_R0.upper_bound_tail;                          // 1.8e-6
const ORACLE_STR = `${(_ORACLE * 1e6).toFixed(1)}×10⁻⁶`;          // "1.8×10⁻⁶"

/* The committed n=100 sweep, interpolated by hand-off time — so the Story tiles quote the
   aggregate (handoff_sweep_results.json) rather than the single representative wave. */
function committedAt(ts) {
  const R = sweep.results;                                         // rows at t_s = 1.0..1.8, ascending
  let i = 0;
  while (i < R.length - 2 && ts > R[i + 1].t_s) i++;
  const a = R[i], b = R[i + 1];
  const f = (ts - a.t_s) / (b.t_s - a.t_s);
  const lerp = (k) => a[k] + f * (b[k] - a[k]);
  return { mlTail: lerp("fno_tail"), hyTail: lerp("hybrid_tail"), work: lerp("numerical_fraction") };
}

/* "How it's built", the hand-off mechanism, each step with the design reason (grounded in the code).
   WORDING LOCK: say "verified" / "shows", never "proved" / "proven" / "guarantees" anywhere below.
   The viva Q&A explicitly answers "is this a mathematical proof?" with "no, verified empirically",
   so this file must not contradict that. If you're re-typing this after a revert, keep it this way. */
const COUPLING_BUILD = [
  {
    n: 1, color: "#4f46e5", title: "Run the ML solver",
    what: "Run the fast ML model (FNO / PINN / DeepONet) across the whole time window.",
    chips: [{ label: "ML prediction stream", color: "#4f46e5" }],
    why: "The ML is cheap, so it carries the wave while it can still be trusted. It only gets replaced once, at the switch."
  },
  {
    n: 2, color: "#0d9488", title: "Re-anchor at the switch",
    what: "Give the numerical solver the ML model's last reliable state, the step right before the switch, and start it from there.",
    detail: "solve_from(u_ML(t_s), i_start):  first numerical frame = u_ML(t_s)  →  jump = 0",
    why: "This makes the jump at the hand-off zero. No blending, no interpolation. The numerical solver's first frame is exactly the state it was handed, so there's no seam."
  },
  {
    n: 3, color: "#7c3aed", title: "Continue under the production scheme",
    what: "Keep advancing with the team's own trusted numerical solver, same grid and method as always, nothing special about the restart.",
    chips: [{ label: "verified restart", color: "#7c3aed" }, { label: "= production solver", color: "#7c3aed" }],
    why: "It's the method everyone already trusts. The restart version was checked bit-identical to it, so nothing about the numerics changes."
  },
  {
    n: 4, color: "#e11d48", title: "Switch once, never hand back",
    what: "At the first trust trigger, switch once and stay on the numerical solver for the rest of the run.",
    detail: "if trust fires: switch once, never hand back",
    why: "Switching back would re-introduce ML error. The numerical solver is plugged in, not hardcoded, so the same restart works with any trigger or backend."
  },
  {
    n: 5, color: "#059669", title: "Verify & decompose the error",
    what: "Run automated tests, plus a restart from the true state, to see how much error the hand-off adds versus how much it inherits from the ML.",
    detail: "E_coupling ≈ 10⁻⁶  ≪  E_inherited  (dominates)",
    why: "This shows the hand-off itself adds almost nothing. Almost all the remaining error comes from the ML state it started from, not from the switch."
  },
];

/* the 4 short reasons behind the design, kept as bullets, not paragraphs */
const WHY_CHOICES = [
  { t: "Same state", d: "prevents an artificial discontinuity at the switch" },
  { t: "Production numerical scheme preserved", d: "the fallback stays the one everyone already trusts" },
  { t: "One-way hard switch", d: "avoids re-introducing ML drift by switching back" },
  { t: "Coupling interface", d: "separates switching logic from model internals" },
];

/* Final-integration baselines, read live from trust_hardswitch_compare.json. */
const BASELINES = [
  { m: "Pure FNO, no hand-off", err: (cmp.means.pure_fno * 100).toFixed(1) + "%", work: "0%", c: "#e11d48" },
  { m: "Fixed switch @ t = 1.4", err: (cmp.means.fixed_1p4 * 100).toFixed(1) + "%", work: (cmp.fixed_workload * 100).toFixed(0) + "%", c: "#d97706" },
  { m: "Trust-triggered (real M1)", err: (cmp.means.trust * 100).toFixed(1) + "%", work: (cmp.trust_workload * 100).toFixed(0) + "%", c: "#059669" },
  { m: "Pure numerical (reference)", err: "reference", work: "100%", c: "#64748b" },
];

/* Model-agnostic table (hand-off at t_s = 1.0), read live from transfer_models_results.json. */
const MODEL_ROWS = ["FNO", "PINN", "DeepONet"].map((m) => {
  const r = transfer.models[m].find((x) => Math.abs(x.t_s - 1.0) < 1e-9) || transfer.models[m][0];
  return { m, es: (r.state_err * 100).toFixed(1), pml: (r.pureML_tail * 100).toFixed(1), hyb: (r.hybrid_tail * 100).toFixed(1), ben: (r.benefit * 100).toFixed(0) };
});

/* State-jump/residual diagnostic at t_s = 1.0, used in Evaluation Q3 (continuity check). */
const _STAB0 = stab.rows.find((r) => Math.abs(r.t_s - 1.0) < 1e-9) || stab.rows[0];

/* ---- the wave, carried by whoever owns it at the playhead ---- */
function RelayWave({ frame, hyField, switched }) {
  const W = 580, H = 260, padX = 16, padT = 14, padB = 24;
  const sx = (x) => padX + ((x + 1) / 2) * (W - 2 * padX);
  const sy = (v) => H - padB - ((v + 1.4) / 2.8) * (H - padT - padB);
  const path = (arr) =>
    arr.map((v, i) => `${i ? "L" : "M"}${sx(RL_XS[i]).toFixed(1)} ${sy(v).toFixed(1)}`).join(" ");
  const hy = hyField || frame.ml;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {[-1, 0, 1].map((v) => (
        <line key={v} x1={sx(-1)} x2={sx(1)} y1={sy(v)} y2={sy(v)} stroke="var(--chart-grid)" strokeWidth="1" />
      ))}
      <path d={path(frame.true)} fill="none" stroke="#94a3b8" strokeWidth="2" strokeDasharray="5 4" />
      <path d={path(frame.ml)} fill="none" stroke="#e11d48" strokeWidth="2"
        strokeOpacity={switched ? 0.45 : 1} strokeLinecap="round" />
      {switched && <path d={path(hy)} fill="none" stroke="#4f46e5" strokeWidth="3" strokeLinecap="round" />}
    </svg>
  );
}

function Badge({ kind }) {
  const c = {
    live: ["LIVE BACKEND", "bg-emerald-100 dark:bg-emerald-500/20 text-emerald-700 dark:text-emerald-300"],
    committed: ["COMMITTED RESULT", "bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300"],
    rep: ["REPRESENTATIVE WAVE", "bg-amber-100 dark:bg-amber-500/20 text-amber-700 dark:text-amber-300"],
    agg: ["AGGREGATE · n = 100 (FULL HELD-OUT SET)", "bg-indigo-100 dark:bg-indigo-500/20 text-indigo-700 dark:text-indigo-300"],
  }[kind];
  return <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full ${c[1]}`}>{c[0]}</span>;
}

/* committed restart-safety boundary (results/module2/figures/restart_safety_boundary.json) */
const SAFETY = [
  { re: 0.87, verified: 1.6e-12, careless: 3.17e-5 },
  { re: 1.78, verified: 7.8e-12, careless: 1.87e-3 },
  { re: 3.72, verified: 1.6e-11, careless: 1.98e-2 },
  { re: 5.81, verified: 1.3e-11, careless: 5.60e-2 },
  { re: 8.16, verified: 9.6e-12, careless: 1.08e-1 },
  { re: 12.87, verified: 5.5e-12, careless: 2.52e-1 },
  { re: 17.59, verified: 3.4e-12, careless: null, unstable: true },
];
const SAFETY_CROSS = 3.16;   // careless restart crosses the 1% target here

/* oracle decomposition, same continuation, restarted from ML state vs the true state (log bars) */
function OracleBars({ mlTail, hyTail, oracle }) {
  const W = 640, H = 156, padL = 176, padR = 60, padT = 14, padB = 26;
  const rows = [
    { label: "Pure ML, no hand-off", v: mlTail, color: "#e11d48" },
    { label: "Hybrid, restart from ML state", v: hyTail, color: "#4f46e5" },
    { label: "Oracle, restart from TRUE state", v: oracle, color: "#059669" },
  ];
  const L = Math.log10;
  const x0 = L(Math.max(oracle, 1e-7)) - 0.3, x1 = L(Math.max(mlTail, 1e-2)) + 0.3;
  const sx = (v) => padL + ((L(Math.max(v, 1e-7)) - x0) / (x1 - x0)) * (W - padL - padR);
  const bh = 20, gap = (H - padT - padB - rows.length * bh) / (rows.length - 1);
  const y = (i) => padT + i * (bh + gap);
  const dec = []; for (let k = Math.ceil(x0); k <= Math.floor(x1); k++) dec.push(k);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {dec.map((k) => (
        <g key={k}>
          <line x1={sx(10 ** k)} x2={sx(10 ** k)} y1={padT} y2={H - padB} stroke="var(--chart-grid)" strokeDasharray="2 4" />
          <text x={sx(10 ** k)} y={H - padB + 14} textAnchor="middle" fontSize="9" fill="var(--chart-axis)">{sci10(k)}</text>
        </g>
      ))}
      {rows.map((d, i) => (
        <g key={d.label}>
          <text x={padL - 8} y={y(i) + bh - 5} textAnchor="end" fontSize="10.5" fill="var(--chart-axis)">{d.label}</text>
          <rect x={padL} y={y(i)} width={Math.max(3, sx(d.v) - padL)} height={bh} rx="4" fill={d.color} opacity="0.9" />
          <text x={Math.max(sx(d.v) + 6, padL + 6)} y={y(i) + bh - 5} fontSize="10.5" fontWeight="700" fill={d.color}>
            {d.v >= 0.001 ? `${(d.v * 100).toFixed(d.v < 0.1 ? 2 : 1)}%` : `${(d.v * 100).toFixed(4)}%`}
          </text>
        </g>
      ))}
    </svg>
  );
}

/* restart-safety boundary, verified vs shortcut restart tail error vs cell Reynolds number (log-log) */
function SafetyChart({ data, cross }) {
  const W = 640, H = 236, padL = 54, padR = 20, padT = 16, padB = 40;
  const L = Math.log10;
  const res = data.map((d) => d.re);
  const xmin = L(Math.min(...res) * 0.8), xmax = L(Math.max(...res) * 1.2);
  const ymin = L(1e-12), ymax = L(0.4);
  const sx = (v) => padL + ((L(v) - xmin) / (xmax - xmin)) * (W - padL - padR);
  const sy = (v) => H - padB - ((L(Math.max(v, 1e-12)) - ymin) / (ymax - ymin)) * (H - padT - padB);
  const line = (key) => data.filter((d) => d[key] != null && d[key] > 0)
    .map((d, i) => `${i ? "L" : "M"}${sx(d.re).toFixed(1)} ${sy(d[key]).toFixed(1)}`).join(" ");
  const unstable = data.find((d) => d.unstable);
  const xdec = []; for (let k = Math.ceil(xmin); k <= Math.floor(xmax); k++) xdec.push(k);
  const ydec = []; for (let k = Math.ceil(ymin); k <= Math.floor(ymax); k += 2) ydec.push(k);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      <rect x={padL} y={padT} width={W - padL - padR} height={H - padT - padB} fill="var(--chart-surface)" stroke="var(--chart-grid)" />
      {ydec.map((k) => (
        <g key={"y" + k}>
          <line x1={padL} x2={W - padR} y1={sy(10 ** k)} y2={sy(10 ** k)} stroke="var(--chart-grid)" strokeDasharray="2 4" />
          <text x={padL - 6} y={sy(10 ** k) + 3} textAnchor="end" fontSize="9" fill="var(--chart-axis)">{sci10(k)}</text>
        </g>
      ))}
      {xdec.map((k) => (
        <text key={"x" + k} x={sx(10 ** k)} y={H - padB + 14} textAnchor="middle" fontSize="9" fill="var(--chart-axis)">{10 ** k}</text>
      ))}
      <line x1={padL} x2={W - padR} y1={sy(0.01)} y2={sy(0.01)} stroke="#f59e0b" strokeDasharray="4 3" />
      <text x={W - padR - 2} y={sy(0.01) - 3} textAnchor="end" fontSize="9" fill="#f59e0b">1% target</text>
      <line x1={sx(cross)} x2={sx(cross)} y1={padT} y2={H - padB} stroke="#e11d48" strokeWidth="1.2" strokeDasharray="3 3" />
      <text x={sx(cross) + 3} y={padT + 10} fontSize="9" fill="#e11d48">{`Re≈${cross.toFixed(1)}`}</text>
      <path d={line("careless")} fill="none" stroke="#e11d48" strokeWidth="2.5" strokeLinejoin="round" />
      <path d={line("verified")} fill="none" stroke="#059669" strokeWidth="2.5" strokeLinejoin="round" />
      {unstable && <text x={sx(unstable.re)} y={sy(0.33)} textAnchor="middle" fontSize="9" fontWeight="700" fill="#e11d48">UNSTABLE ✕</text>}
      <text x={(padL + W - padR) / 2} y={H - 4} textAnchor="middle" fontSize="10" fill="var(--chart-axis)">cell Reynolds number at hand-off (sharper waves →)</text>
    </svg>
  );
}

function FigCard({ src, title, note, metric, deduction, script, h = 320 }) {
  return (
    <figure className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4">
      <div className="flex items-center justify-center rounded-lg border border-slate-100 dark:border-slate-700 bg-white overflow-hidden" style={{ height: h }}>
        <img src={src} alt={title} loading="lazy" className="max-w-full max-h-full object-contain" />
      </div>
      <figcaption className="mt-3">
        <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">{title}</div>
        <div className="text-xs text-slate-500 dark:text-slate-400 mt-1 leading-snug">{note}</div>
        {metric && (
          <div className="mt-2 text-[11px] leading-snug text-teal-700 dark:text-teal-300 bg-teal-50/60 dark:bg-teal-500/10 border border-teal-100 dark:border-teal-500/20 rounded-lg px-2.5 py-1.5">
            <span className="font-bold">Metric: </span>{metric}
          </div>
        )}
        {deduction && (
          <div className="mt-2 text-[11px] leading-snug text-indigo-700 dark:text-indigo-300 bg-indigo-50/60 dark:bg-indigo-500/10 border border-indigo-100 dark:border-indigo-500/20 rounded-lg px-2.5 py-1.5">
            <span className="font-bold">→ Deduction: </span>{deduction}
          </div>
        )}
        {script && <div className="text-[10px] font-mono text-slate-400 dark:text-slate-500 mt-1.5">{script}</div>}
      </figcaption>
    </figure>
  );
}

const EVAL_ROBUST = [
  { src: "/module2_figures/fig8_ood_error_over_time.png", title: "Out-of-distribution wave", note: "A higher-frequency wave sin(6πx), beyond the trained band (modes 1–4). The hybrid still limits the damage after the hand-off, but cannot recover a state the ML has already lost.", metric: "Relative L2 error against the true Cole–Hopf solution, on an out-of-distribution input.", deduction: "The coupling is corrective, not reconstructive: it faithfully continues the state it receives, but cannot rebuild information the ML has already lost. This is exactly why switching early matters.", script: "ood_experiment.py" },
];

export default function CouplingPage() {
  const [tab, setTab] = useState("story");

  /* ---- the relay state ---- */
  const [tsIdx, setTsIdx] = useState(1);            // index into RL_SWITCHES (default t_s = 1.0)
  const [ph, setPh] = useState(0);                  // playhead frame
  const [playing, setPlaying] = useState(true);
  const [robustView, setRobustView] = useState("normal");   // Story §2 toggle: normal held-out wave vs severe OOD wave
  const [showNumerics, setShowNumerics] = useState(false);    // How it's built: "View numerical details" accordion
  const [showDeepVal, setShowDeepVal] = useState(false);      // Evaluation: "Deep technical validation" accordion
  const [showDesignChoices, setShowDesignChoices] = useState(false); // How it's built: "Why these choices" accordion
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setPh((k) => (k + 1) % RL_FRAMES.length), 110);
    return () => clearInterval(id);
  }, [playing]);

  const sw = RL_SWITCHES[tsIdx];
  const cm = committedAt(sw.ts);   // committed n=100 aggregate at this hand-off time
  const frame = RL_FRAMES[ph];
  const tNow = frame.t;
  const switched = tNow >= sw.ts;
  const hyField = switched ? sw.hyF[String(ph)] : null;
  const verdict = sw.ts <= 1.3 ? "meets 10% accuracy criterion" : sw.ts <= RL_META.boundary ? "diminishing benefit" : "exceeds 10% error criterion";
  const vTone = { "meets 10% accuracy criterion": "emerald", "diminishing benefit": "amber", "exceeds 10% error criterion": "rose" }[verdict];

  /* errHy is the full-length hybrid error curve (ML before the switch by construction) */
  const hyErrCurve = sw.errHy;

  /* ---- live run state (real backend, real M2Coupling) ---- */
  const [meta, setMeta] = useState(null);
  const [model, setModel] = useState("FNO");
  const [modes, setModes] = useState(2);
  const [amplitude, setAmplitude] = useState(1.0);
  const [pinnIndex, setPinnIndex] = useState(0);
  // "real" = one of the 10 official held-out test ICs the aggregate numbers were
  // measured on (default, so the live run is grounded in the evaluated data, like
  // the Cost Control demo). "synthetic" = a fresh random shape for exploration.
  const [source, setSource] = useState("real");
  // Test IC #904: its pure-ML tail (~14%) matches FNO's published mean extrapolation
  // error, so it's the representative default rather than the easiest wave.
  const [ridx, setRidx] = useState(4);
  const [switchMode, setSwitchMode] = useState("manual");
  const [lts, setLts] = useState(1.0);
  const [ic, setIc] = useState(null);
  const [lf, setLf] = useState(null);
  const [hist, setHist] = useState([]);
  const [summary, setSummary] = useState(null);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => { getMeta().then(setMeta).catch(() => { }); }, []);
  useEffect(() => {
    if (!meta) return;
    if (model === "PINN") pinnIC(pinnIndex).then((d) => setIc(d.ic)).catch(() => { });
    else if (source === "real") realTestIC(ridx).then((d) => setIc(d.ic)).catch(() => { });
    else buildIC(modes, amplitude).then((d) => setIc(d.ic)).catch(() => { });
  }, [meta, model, modes, amplitude, pinnIndex, source, ridx]);

  function run() {
    if (wsRef.current) wsRef.current.close();
    setHist([]); setLf(null); setSummary(null); setErr(null); setRunning(true);
    const payload = model === "PINN"
      ? { model, pinn_index: pinnIndex, switch_mode: switchMode, t_s: lts }
      : source === "real"
        ? { model, real_ic_index: ridx, switch_mode: switchMode, t_s: lts }
        : { model, ic, switch_mode: switchMode, t_s: lts };
    wsRef.current = runCoupling(payload,
      (f) => { setLf(f); setHist((h) => [...h, f]); },
      (s) => setSummary(s),
      () => setRunning(false),
      (e) => { setErr(e); setRunning(false); });
  }

  const x = meta?.x || [];
  const muted = "text-slate-500 dark:text-slate-400";
  const lyr = lf ? [Math.min(...lf.true, ...lf.hybrid, -1.1), Math.max(...lf.true, ...lf.hybrid, 1.1)] : [-1.1, 1.1];
  const vChip = {
    emerald: "bg-emerald-100 dark:bg-emerald-500/20 text-emerald-700 dark:text-emerald-300",
    amber: "bg-amber-100 dark:bg-amber-500/20 text-amber-700 dark:text-amber-300",
    rose: "bg-rose-100 dark:bg-rose-500/20 text-rose-700 dark:text-rose-300",
  }[vTone];

  return (
    <div className="space-y-6">
      {/* HEADER, its own identity: the relay */}
      <div className="rounded-2xl p-6 bg-gradient-to-r from-rose-50 via-white to-indigo-50 dark:from-rose-500/10 dark:via-slate-800 dark:to-indigo-500/10 border border-slate-200 dark:border-slate-700">
        <span className="text-xs font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">
          Hybrid components · Module 2
        </span>
        <h1 className="text-3xl font-extrabold mt-1">
          <span className="text-rose-600 dark:text-rose-400">Coupling</span>
          <span className="text-slate-400 dark:text-slate-500 mx-2">·</span>
          <span className="text-slate-800 dark:text-slate-100">the baton pass</span>
        </h1>
        <p className="text-slate-500 dark:text-slate-400 mt-1 text-sm">
          One trajectory, two runners. The <span className="font-semibold text-rose-600 dark:text-rose-400">fast ML model</span> carries
          the wave first; when the trust/control system requests a switch, my verified hand-off transfers its current state to the{" "}
          <span className="font-semibold text-indigo-600 dark:text-indigo-400">numerical solver</span>, which carries the rest of the trajectory.
        </p>
      </div>

      {/* TABS */}
      <div className="flex gap-2">
        {[["story", "Story"], ["built", "How it's built"], ["eval", "Evaluation"], ["live", "Try it live"]].map(([v, label]) => (
          <button key={v} onClick={() => setTab(v)}
            className={`px-4 py-2 rounded-xl text-sm font-medium border transition ${tab === v
              ? "bg-indigo-600 text-white border-indigo-600"
              : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700"}`}>
            {label}
          </button>
        ))}
      </div>

      {tab === "story" && (<div className="space-y-6">
        {/* ===== SECTION 2 · ROBUSTNESS, AS A STRIP — not a page. One toggle, one deduction.
            Reuses only committed numbers/figures already defined above (AGG, EVAL_ROBUST). ===== */}
        {/* <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/60 p-5">
          <div className="flex items-center justify-between flex-wrap gap-2 mb-3">
            <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">Why a fallback is needed</span>
            <div className="flex gap-1.5">
              {[["normal", "Held-out wave"], ["ood", "Severe OOD wave"]].map(([v, label]) => (
                <button key={v} onClick={() => setRobustView(v)}
                  className={`px-2.5 py-1 rounded-lg text-[11px] font-bold border transition ${robustView === v
                    ? "bg-indigo-600 text-white border-indigo-600" : inactiveBtn}`}>{label}</button>
              ))}
            </div>
          </div>
          <div className="relative h-7 rounded-lg overflow-hidden flex text-[11px] font-bold text-white select-none">
            <div className="bg-slate-400 dark:bg-slate-500 grid place-items-center" style={{ width: "50%" }}>trained region</div>
            <div className="bg-rose-500/90 grid place-items-center flex-1">future · extrapolation</div>
          </div>
          <div className="flex justify-between text-[10px] text-slate-400 dark:text-slate-500 mt-1 mb-3">
            <span>t = 0</span><span>t = 1</span><span>t = 2</span>
          </div>
          {robustView === "normal" ? (
            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
              On a normal held-out wave, ML is accurate inside what it learned, but its error starts growing once we
              ask it to predict further into the future. By t = 2 with no fallback, its own error alone reaches{" "}
              <b className="text-rose-600 dark:text-rose-400">{AGG.fnoTail1}%</b>.
            </p>
          ) : (
            <div className="flex flex-col items-center gap-1.5 max-w-xs mx-auto py-1">
              {["UNFAMILIAR INPUT", "ML state already badly wrong", "Numerics can continue it, but cannot reconstruct what was lost"].map((t, i) => (
                <div key={t} className="contents">
                  {i > 0 && <span className="text-slate-300 dark:text-slate-600 text-xs">↓</span>}
                  <div className={`rounded-lg px-3 py-1.5 text-center ${i === 0 ? "text-[11px] font-bold bg-slate-100 dark:bg-slate-700/50 text-slate-500 dark:text-slate-400" : i === 1 ? "text-xs font-semibold bg-rose-50 dark:bg-rose-500/10 text-rose-600 dark:text-rose-400" : "text-xs bg-slate-50 dark:bg-slate-700/30 text-slate-600 dark:text-slate-300"}`}>{t}</div>
                </div>
              ))}
            </div>
          )}
          <p className="text-xs font-medium text-slate-700 dark:text-slate-200 mt-3 pt-3 border-t border-slate-200 dark:border-slate-700">
            → This is why the system needs a fallback.
          </p>
        </div>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Module 1 detects, Module 3 decides, and my Module 2 executes the takeover.
        </p> */}
        <div className="flex items-center gap-2"><Badge kind="rep" /><span className="text-xs text-slate-400 dark:text-slate-500">the wave shape is one representative held-out wave; the numbers are the n = {AGG.nIC} committed aggregate at each hand-off time</span></div>
        {/* ===== THE RELAY TIMELINE, the page's centrepiece ===== */}
        <div className="rounded-2xl border-2 border-indigo-200 dark:border-indigo-500/30 bg-white dark:bg-slate-800 p-5">
          <div className="flex items-center justify-between mb-3">
            <div className="text-sm font-semibold text-slate-800 dark:text-slate-100 flex items-center gap-2">
              <GitCommitHorizontal size={16} className="text-indigo-500" />
              Who carries the wave · drag to set the hand-off, t_s = {sw.ts.toFixed(1)}
            </div>
            <span className={`text-[11px] font-bold uppercase tracking-wider px-2.5 py-1 rounded-full ${vChip}`}>
              {verdict}{verdict === "exceeds 10% error criterion" && ", still improves, but the state was already too degraded"}
            </span>
          </div>

          {/* ownership bar with playhead baton */}
          <div className="relative h-9 rounded-lg overflow-hidden flex text-[11px] font-bold text-white select-none">
            <div className="bg-rose-500/90 grid place-items-center transition-all duration-300"
              style={{ width: `${sw.i0f * 100}%` }}>ML, fast</div>
            <div className="bg-indigo-600 grid place-items-center flex-1 transition-all duration-300">
              numerical, verified restart
            </div>
            {/* training-horizon + boundary ticks */}
            <div className="absolute top-0 h-full w-0.5 bg-white/70" style={{ left: "50.2%" }} title="training horizon t=1" />
            <div className="absolute top-0 h-full w-0.5 bg-amber-300" style={{ left: `${(RL_META.boundary / 2) * 100}%` }} title="viability boundary" />
            {/* the baton */}
            <div className="absolute top-1/2 -translate-y-1/2 w-4 h-4 rounded-full border-2 border-white shadow transition-all duration-100"
              style={{
                left: `calc(${(tNow / 2) * 100}% - 8px)`,
                background: switched ? "#4f46e5" : "#e11d48",
              }} />
          </div>
          <div className="flex justify-between text-[10px] text-slate-400 dark:text-slate-500 mt-1">
            <span>t = 0</span>
            <span className="text-slate-500 dark:text-slate-300">│ t = 1 training ends</span>
            <span className="text-amber-500">│ t ≈ {RL_META.boundary} last handoff meeting the joint viability criterion</span>
            <span>t = 2</span>
          </div>

          <input type="range" min="0" max={RL_SWITCHES.length - 1} step="1" value={tsIdx}
            onChange={(e) => setTsIdx(+e.target.value)}
            className="w-full mt-3 accent-indigo-600" />

          {/* consequences of the chosen handoff, updates instantly */}
          <div className="grid grid-cols-4 gap-3 mt-3">
            <div className="rounded-xl bg-rose-50 dark:bg-rose-500/10 border border-rose-100 dark:border-rose-500/20 px-3 py-2 text-center">
              <div className="text-[10px] uppercase tracking-wide text-rose-500">error if we never switch</div>
              <div className="text-xl font-extrabold text-rose-600 dark:text-rose-400">{(cm.mlTail * 100).toFixed(1)}%</div>
            </div>
            <div className="rounded-xl bg-indigo-50 dark:bg-indigo-500/10 border border-indigo-100 dark:border-indigo-500/20 px-3 py-2 text-center">
              <div className="text-[10px] uppercase tracking-wide text-indigo-500">error if we switch here</div>
              <div className="text-xl font-extrabold text-indigo-600 dark:text-indigo-400">{(cm.hyTail * 100).toFixed(cm.hyTail < 0.1 ? 2 : 1)}%</div>
            </div>
            <div className="rounded-xl bg-slate-50 dark:bg-slate-700/40 border border-slate-100 dark:border-slate-600/40 px-3 py-2 text-center">
              <div className="text-[10px] uppercase tracking-wide text-slate-400">numerical work · the cost</div>
              <div className="text-xl font-extrabold text-slate-700 dark:text-slate-200">{(cm.work * 100).toFixed(0)}%</div>
            </div>
            <div className="rounded-xl bg-emerald-50 dark:bg-emerald-500/10 border border-emerald-100 dark:border-emerald-500/20 px-3 py-2 text-center">
              <div className="text-[10px] uppercase tracking-wide text-emerald-600">glitch at the switch</div>
              <div className="text-xl font-extrabold text-emerald-600 dark:text-emerald-400">{sw.jump === 0 ? "0" : eToSup(sw.jump)}</div>
            </div>
            <p className="col-span-4 text-[11px] text-slate-500 dark:text-slate-400 mt-2">
              {verdict === "meets 10% accuracy criterion"
                ? "The ML state is still good here, so numerical continuation gives a large improvement."
                : verdict === "diminishing benefit"
                  ? "Less numerical work at this point, but the handed-over ML state is already more degraded."
                  : "Numerics can stop further drift from here, but they can't undo error already present at the switch."}
            </p>
          </div>
        </div>

        {/* ===== SECTION 4 · WHAT PHYSICALLY GETS HANDED OVER ===== */}
        <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3">What physically gets handed over</div>
          <div className="flex items-center justify-center gap-1 flex-wrap text-center">
            {[
              { label: "ML's last reliable state (step before switch)", c: "#e11d48" },
              { label: "SAME STATE", c: "#64748b" },
              { label: "Numerical solver starts from it", c: "#4f46e5" },
              { label: "Continuation to t = 2", c: "#4f46e5" },
            ].map((s, i, arr) => (
              <div key={s.label} className="flex items-center">
                <div className="rounded-xl px-3 py-2 text-[11px] font-bold" style={{ background: s.c + "1A", color: s.c }}>{s.label}</div>
                {i < arr.length - 1 && <span className="mx-1.5 text-slate-300 dark:text-slate-600">→</span>}
              </div>
            ))}
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-3">No blending. No replacement with ground truth.</p>
          <p className="text-xs font-medium text-emerald-700 dark:text-emerald-400 mt-1">
            First numerical state = handed-over ML state → state jump = {sw.jump === 0 ? "0" : eToSup(sw.jump)}
          </p>
        </div>


      </div>)}
      {tab === "built" && (<div className="space-y-6">
        <div className="rounded-2xl p-6 md:p-7 bg-gradient-to-r from-indigo-50 via-indigo-50 to-violet-50 dark:from-indigo-500/10 dark:via-indigo-500/10 dark:to-violet-500/10 border border-indigo-100 dark:border-indigo-500/25">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">A clean, tested switch, no glitch</div>
          <h2 className="text-2xl font-bold mt-1 text-slate-800 dark:text-slate-100">How the hand-off is built</h2>
          <p className="text-sm text-slate-600 dark:text-slate-300 mt-2 leading-relaxed">
            The ML model runs first, while it&apos;s still reliable. At the switch, the trusted method picks up using the
            ML model&apos;s exact last prediction as its starting point, then continues on its own. This restart behaves
            exactly like the trusted method always does, and the switch itself creates no visible break in the result.
            Every step below exists to make sure of that, not just for convenience.
          </p>
        </div>

        {/* ===== WHERE THIS MODULE SITS, moved here from Story. Particularly important
            for a software-engineering evaluator: gives the interface before the internals. ===== */}
        <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3">Where this module sits</div>
          <div className="flex items-center justify-center gap-2 flex-wrap text-center">
            <div className="rounded-xl px-3 py-2.5 text-[11px] font-semibold bg-slate-50 dark:bg-slate-700/40 border border-slate-200 dark:border-slate-600 text-slate-600 dark:text-slate-300">Module 1<br /><span className="font-normal text-slate-400 dark:text-slate-500">&ldquo;Can ML still be trusted?&rdquo;</span></div>
            <span className="text-slate-300 dark:text-slate-600">→</span>
            <div className="rounded-xl px-3 py-2.5 text-[11px] font-semibold bg-slate-50 dark:bg-slate-700/40 border border-slate-200 dark:border-slate-600 text-slate-600 dark:text-slate-300">Module 3<br /><span className="font-normal text-slate-400 dark:text-slate-500">&ldquo;Spend numerical work now?&rdquo;</span></div>
            <span className="text-slate-300 dark:text-slate-600">→</span>
            <div className="rounded-xl px-3 py-2.5 text-[11px] font-bold bg-indigo-600 text-white">MODULE 2 · MY PART<br /><span className="font-normal text-indigo-100">Execute the takeover</span></div>
            <span className="text-slate-300 dark:text-slate-600">→</span>
            <div className="rounded-xl px-3 py-2.5 text-[11px] font-semibold bg-emerald-50 dark:bg-emerald-500/10 border border-emerald-200 dark:border-emerald-500/30 text-emerald-700 dark:text-emerald-300">Hybrid trajectory</div>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-4 text-xs">
            <div className="rounded-lg bg-indigo-50/50 dark:bg-indigo-500/10 border border-indigo-100 dark:border-indigo-500/20 px-3 py-2">
              <span className="font-bold text-indigo-600 dark:text-indigo-400">Inputs: </span>
              <span className="text-slate-600 dark:text-slate-300">current ML state · switch time / trigger · numerical solver</span>
            </div>
            <div className="rounded-lg bg-emerald-50/50 dark:bg-emerald-500/10 border border-emerald-100 dark:border-emerald-500/20 px-3 py-2">
              <span className="font-bold text-emerald-600 dark:text-emerald-400">Output: </span>
              <span className="text-slate-600 dark:text-slate-300">one continuous hybrid trajectory</span>
            </div>
          </div>
        </div>

        {/* ===== IMPLEMENTATION PIPELINE, same pattern as Cost Control and Trust score:
            a non-interactive step strip, then all steps shown fully, at once. ===== */}
        <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-5 py-4">
          <div className="flex flex-wrap items-center gap-y-2">
            {COUPLING_BUILD.map((s, i) => (
              <div key={s.n} className="flex items-center">
                <div className="flex items-center gap-2">
                  <span className="w-6 h-6 rounded-lg text-[11px] font-bold flex items-center justify-center shrink-0" style={{ background: s.color + "1A", color: s.color }}>{s.n}</span>
                  <span className="text-xs font-medium text-slate-600 dark:text-slate-300 whitespace-nowrap">{s.title}</span>
                </div>
                {i < COUPLING_BUILD.length - 1 && <span className="mx-2.5 text-slate-300 dark:text-slate-600 text-xs">→</span>}
              </div>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {COUPLING_BUILD.map((s) => (
            <div key={s.n} className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5 shadow-sm hover:shadow-md transition flex flex-col">
              <div className="flex items-center gap-3">
                <span className="w-9 h-9 rounded-xl text-sm font-bold flex items-center justify-center shrink-0" style={{ background: s.color + "1A", color: s.color }}>{s.n}</span>
                <h3 className="text-sm font-bold text-slate-800 dark:text-slate-100 leading-tight">{s.title}</h3>
              </div>
              <p className="text-sm text-slate-600 dark:text-slate-300 mt-3">{s.what}</p>
              {s.chips ? (
                <div className="mt-3 flex flex-wrap gap-2">
                  {s.chips.map((cp) => (
                    <span key={cp.label} className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-semibold" style={{ color: cp.color, background: cp.color + "14", border: `1px solid ${cp.color}33` }}>
                      <span className="w-1.5 h-1.5 rounded-full" style={{ background: cp.color }} />{cp.label}
                    </span>
                  ))}
                </div>
              ) : (
                <div className="mt-3 rounded-xl px-3 py-2.5 font-mono text-[12.5px] text-slate-700 dark:text-slate-100 text-center overflow-x-auto" style={{ background: s.color + "0D", border: `1px solid ${s.color}26` }}>{s.detail}</div>
              )}
              <div className="mt-auto pt-3">
                <span className="text-[10px] font-bold uppercase tracking-wide" style={{ color: s.color }}>Why</span>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5 leading-relaxed">{s.why}</p>
              </div>
            </div>
          ))}
        </div>

        {/* ===== WHY THESE CHOICES + HOW IT COMPARES, collapsed by default, only explained if asked ===== */}
        <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
          <button onClick={() => setShowDesignChoices((v) => !v)}
            className="w-full text-left text-sm font-bold text-slate-700 dark:text-slate-200 flex items-center justify-between">
            Why these design choices
            <span className="text-slate-400 dark:text-slate-500 text-xs">{showDesignChoices ? "▾" : "▸"}</span>
          </button>
          {showDesignChoices && (
            <div className="mt-4 space-y-5">
              <div>
                <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-2">Design choices</div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {WHY_CHOICES.map((w) => (
                    <div key={w.t} className="rounded-xl border border-slate-100 dark:border-slate-700 bg-slate-50 dark:bg-slate-700/30 px-3 py-2.5">
                      <div className="text-xs font-bold text-slate-700 dark:text-slate-200">{w.t}</div>
                      <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">{w.d}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* WORDING LOCK: "verified" not "proved/proven", "delivers" not "guarantees" — matches the viva Q&A defense. */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="rounded-2xl border border-emerald-200 dark:border-emerald-500/30 bg-emerald-50 dark:bg-emerald-500/10 p-5">
            <div className="text-sm font-bold text-emerald-800 dark:text-emerald-300">Tested, not assumed</div>
            <p className="text-sm text-slate-600 dark:text-slate-300 mt-1 leading-relaxed">
              The restart isn&apos;t just &ldquo;close enough&rdquo; to the trusted method, we compared them directly and got an
              exact match. And &ldquo;no glitch at the switch&rdquo; isn&apos;t something we assume either, we tested it. Every number
              on this page has a test behind it, proving it&apos;s true.
            </p>
          </div>
          <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-700/30 p-5">
            <div className="text-sm font-bold text-slate-700 dark:text-slate-200">It times the hand-off, it doesn&apos;t fix the ML</div>
            <p className="text-sm text-slate-600 dark:text-slate-300 mt-1 leading-relaxed">
              Our checks show my switch adds almost no error of its own, whatever error is left over came from the ML model.
              So my module makes sure the switch happens cleanly and at the right time, it can&apos;t fix a bad ML
              prediction, and it was never meant to.
            </p>
          </div>
        </div>

      </div>)}

      {tab === "eval" && (<div className="space-y-6">
        <div className="rounded-2xl p-6 bg-gradient-to-r from-indigo-50 via-white to-white dark:from-indigo-500/10 dark:via-slate-800 dark:to-slate-800 border border-slate-200 dark:border-slate-700">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">Evaluation</div>
          <h2 className="text-2xl font-bold mt-1 text-slate-800 dark:text-slate-100">Did the hand-off work, and how do I know?</h2>
          <p className="text-sm text-slate-600 dark:text-slate-300 mt-2 leading-relaxed">
            Every number below is scored offline against the exact Cole–Hopf answer, on held-out data. The coupling
            itself never sees the true answer at runtime, this is a check afterwards, not part of the switch.
          </p>
        </div>

        {/* ===== Q1 · DOES SWITCHING ACTUALLY HELP? ===== */}
        <div className="rounded-2xl border-2 border-indigo-200 dark:border-indigo-500/30 bg-white dark:bg-slate-800 p-5">
          <div className="text-sm font-bold text-indigo-700 dark:text-indigo-300 mb-1">Q1 · Does switching actually help?</div>
          <div className="text-[11px] text-slate-400 dark:text-slate-500 mb-3">FNO, hand-off at t_s = 1.0, full held-out test set (n = {AGG.nIC})</div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
            <div className="rounded-xl bg-rose-50 dark:bg-rose-500/10 border border-rose-100 dark:border-rose-500/20 px-3 py-2 text-center">
              <div className="text-[10px] uppercase tracking-wide text-rose-500">pure ML tail error</div>
              <div className="text-2xl font-extrabold text-rose-600 dark:text-rose-400">{AGG.fnoTail2}%</div>
            </div>
            <div className="rounded-xl bg-indigo-50 dark:bg-indigo-500/10 border border-indigo-100 dark:border-indigo-500/20 px-3 py-2 text-center">
              <div className="text-[10px] uppercase tracking-wide text-indigo-500">hybrid tail error</div>
              <div className="text-2xl font-extrabold text-indigo-600 dark:text-indigo-400">{AGG.hybTail2}%</div>
            </div>
            <div className="rounded-xl bg-emerald-50 dark:bg-emerald-500/10 border border-emerald-100 dark:border-emerald-500/20 px-3 py-2 text-center">
              <div className="text-[10px] uppercase tracking-wide text-emerald-600">error reduction</div>
              <div className="text-2xl font-extrabold text-emerald-600 dark:text-emerald-400">{AGG.reduction}%</div>
            </div>
            <div className="rounded-xl bg-slate-50 dark:bg-slate-700/40 border border-slate-100 dark:border-slate-600/40 px-3 py-2 text-center">
              <div className="text-[10px] uppercase tracking-wide text-slate-400">held-out cases improved</div>
              <div className="text-2xl font-extrabold text-slate-700 dark:text-slate-200">{AGG.icsImproved} / {AGG.nIC}</div>
            </div>
          </div>
          <FigCard src="/module2_figures/fig1_error_over_time.png" title="Error over time, FNO vs numerical vs hybrid" note="Mean ± std over 100 held-out ICs. After the switch the hybrid tracks the numerical solution instead of drifting with the ML." metric="Relative L2 error at each time t, mean ± std over 100 held-out waves." deduction="Switching halts the ML's continued drift and holds the error flat. Nothing more." script="make_figures.py" />
          <details className="mt-3 text-xs text-slate-500 dark:text-slate-400">
            <summary className="cursor-pointer font-medium text-slate-600 dark:text-slate-300">Same result with the real Module 1 trust trigger, not just a manual t_s</summary>
            <div className="space-y-1.5 mt-2">
              {BASELINES.map((b) => (
                <div key={b.m} className="grid grid-cols-[1fr_auto_auto] items-center gap-4 rounded-xl border border-slate-100 dark:border-slate-700 px-3 py-2">
                  <span className="text-sm text-slate-700 dark:text-slate-200 flex items-center gap-2"><span className="w-2 h-2 rounded-full" style={{ background: b.c }} />{b.m}</span>
                  <span className="text-sm font-bold w-16 text-right" style={{ color: b.c }}>{b.err}</span>
                  <span className="text-xs text-slate-400 dark:text-slate-500 w-28 text-right">{b.work} numerical</span>
                </div>
              ))}
            </div>
          </details>
        </div>

        {/* ===== Q2 · HOW LATE CAN I WAIT? ===== */}
        <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
          <div className="text-sm font-bold text-indigo-700 dark:text-indigo-300 mb-1">Q2 · How late can I wait?</div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mb-3">
            A switch is worth doing only if it helps enough <b>and</b> the result is still accurate enough: benefit ≥ 10%{" "}
            <b>and</b> hybrid tail &lt; 10% (both frozen before the experiment ran). &ldquo;vs 10% bar&rdquo; is the headroom
            under that accuracy limit, positive is fine, negative means too late.
          </p>
          <div className="grid grid-cols-1 lg:grid-cols-[1.5fr_1fr] gap-4 items-stretch">
            <div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead><tr className="text-[11px] uppercase tracking-wide text-slate-400 dark:text-slate-500 text-left">
                    <th className="py-1.5 pr-3">t_s</th>
                    <th className="py-1.5 px-3 text-right">pure-ML tail</th>
                    <th className="py-1.5 px-3 text-right">hybrid tail</th>
                    <th className="py-1.5 px-3 text-right">vs 10% bar</th>
                    <th className="py-1.5 px-3 text-right">benefit</th>
                    <th className="py-1.5 px-3 text-right">numerical work</th>
                    <th className="py-1.5 pl-3 text-center">viable?</th>
                  </tr></thead>
                  <tbody>
                    {sweep.results.map((r) => (
                      <tr key={r.t_s} className="border-t border-slate-100 dark:border-slate-700">
                        <td className="py-2 pr-3 font-semibold text-slate-700 dark:text-slate-200">{r.t_s.toFixed(1)}</td>
                        <td className="py-2 px-3 text-right text-rose-600 dark:text-rose-400">{(r.fno_tail * 100).toFixed(1)}%</td>
                        <td className="py-2 px-3 text-right font-bold text-indigo-600 dark:text-indigo-400">{(r.hybrid_tail * 100).toFixed(r.hybrid_tail < 0.1 ? 2 : 1)}%</td>
                        <td className={`py-2 px-3 text-right font-medium ${r.hybrid_tail < 0.10 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-500"}`}>{10 - r.hybrid_tail * 100 >= 0 ? "+" : "−"}{Math.abs(10 - r.hybrid_tail * 100).toFixed(1)}</td>
                        <td className="py-2 px-3 text-right text-emerald-600 dark:text-emerald-400">{(r.benefit * 100).toFixed(1)}%</td>
                        <td className="py-2 px-3 text-right text-slate-500 dark:text-slate-400">{(r.numerical_fraction * 100).toFixed(0)}%</td>
                        <td className={`py-2 pl-3 text-center font-bold ${r.viable ? "text-emerald-600 dark:text-emerald-400" : "text-rose-500"}`}>{r.viable ? "✓" : "✗"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-3">
                Latest measured viable point = <b>1.4</b>. Interpolated boundary ≈ <b>{AGG.boundary}</b>. Earlier = more
                accurate but more numerical work; later = cheaper but the state has already degraded more.
              </p>
            </div>
            <FigCard src="/module2_figures/fig2_switch_time_vs_benefit.png" h={260} title="Hand-off benefit vs when we switch" note="Benefit = 1 − hybrid/FNO across switch times, versus the fixed 10% benefit criterion used for this evaluation (met across the whole swept range)." metric="Benefit = 1 − hybrid/pure-ML tail, both relative L2 tail error over [t_s, 2]." deduction="The later you wait, the more damaged the handed-over state, so benefit falls monotonically. Switch early for the biggest gain." script="make_figures.py" />
          </div>
        </div>

        {/* ===== Q3 · DID MY TAKEOVER ITSELF BREAK ANYTHING? ===== */}
        <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
          <div className="text-sm font-bold text-indigo-700 dark:text-indigo-300 mb-1">Q3 · Did my takeover itself break anything?</div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mb-3">Three independent checks, all on the same hand-off.</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="rounded-xl border border-slate-100 dark:border-slate-700 bg-slate-50 dark:bg-slate-700/30 p-4">
              <div className="text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1">A · Numerical consistency</div>
              <div className="text-2xl font-extrabold text-emerald-600 dark:text-emerald-400">0.0</div>
              <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">relative difference, restart vs the original solver run from t = 0</div>
              <div className="text-[10px] font-mono text-slate-400 dark:text-slate-500 mt-2">verify_restart.py</div>
            </div>
            <div className="rounded-xl border border-slate-100 dark:border-slate-700 bg-slate-50 dark:bg-slate-700/30 p-4">
              <div className="text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1">B · State continuity</div>
              <div className="text-2xl font-extrabold text-emerald-600 dark:text-emerald-400">{_STAB0.state_jump.toFixed(1)}</div>
              <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">L2 norm of the state discontinuity at the switch</div>
              <div className="text-[10px] font-mono text-slate-400 dark:text-slate-500 mt-2">handoff_stability_diagnostic.py</div>
            </div>
          </div>

          {/* C · Physical consistency, stat and its graph side by side, definition always visible (no hover) */}
          <div className="mt-3 rounded-xl border border-slate-100 dark:border-slate-700 bg-slate-50 dark:bg-slate-700/30 p-4">
            <div className="text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1">C · Physical consistency</div>
            <div className="grid grid-cols-1 md:grid-cols-[240px_1fr] gap-4 items-center mt-2">
              <div>
                <div className="text-2xl font-extrabold text-emerald-600 dark:text-emerald-400">{_STAB0.residual_before.toFixed(3)} → {_STAB0.residual_after.toFixed(3)}</div>
                <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">
                  PDE residual, just before → just after the switch. <br></br><br></br>
                  How well the state satisfies the Burgers equation
                  itself (r = u_t + u·u_x − ν·u_xx), 
                  <br></br><br></br>smaller →  more physically consistent
                </div>
                <div className="text-[10px] font-mono text-slate-400 dark:text-slate-500 mt-2">handoff_stability_diagnostic.py</div>
              </div>
              <div className="rounded-xl border border-slate-100 dark:border-slate-700 bg-white dark:bg-slate-800 p-3">
                <div className="text-[11px] font-semibold text-slate-600 dark:text-slate-300 mb-1">Across every measured switch time</div>
                <div className="flex items-center justify-center h-[190px]">
                  <img src="/module2_figures/handoff_stability_diagnostic.png" alt="Handoff stability diagnostic: PDE residual before vs after the switch"
                    loading="lazy" className="max-w-full max-h-full object-contain rounded-lg border border-slate-100 dark:border-slate-700 bg-white" />
                </div>
              </div>
            </div>
          </div>

          <p className="text-xs font-medium text-slate-700 dark:text-slate-200 mt-3 pt-3 border-t border-slate-200 dark:border-slate-700">
            → No spike at the switch, at every measured switch time the residual drops after takeover. Numerically faithful, continuous, and physically consistent.
          </p>
        </div>

        {/* ===== Q4 · WHERE DOES THE REMAINING HYBRID ERROR COME FROM? ===== */}
        <div className="rounded-2xl border-2 border-indigo-200 dark:border-indigo-500/30 bg-white dark:bg-slate-800 p-5">
          <div className="text-sm font-bold text-indigo-700 dark:text-indigo-300 mb-1">Q4 · Where does the remaining hybrid error come from?</div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mb-3">
            REAL HYBRID (ML's last reliable state, the step before the switch) vs ORACLE CONTROL (the true state at that same moment), same solver · same grid · same time · only the starting state changes.
          </p>
          <div className="max-w-2xl mx-auto">
            <OracleBars mlTail={_AGG_R0.fno_tail} hyTail={_AGG_R0.hybrid_tail} oracle={_ORACLE} />
          </div>
          <p className="text-sm font-semibold text-slate-800 dark:text-slate-100 mt-3">
            The remaining hybrid error is mainly inherited from the ML state, not created by the hand-off.
          </p>
          <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">
            ML state error at hand-off ≈ {AGG.stateErr}% is close to the mean hybrid tail error {AGG.hybTail2}%, two different metrics that needn&apos;t match exactly, but both point the same way.
          </p>
        </div>

        {/* ===== Q5 · DOES THE SAME PATTERN APPEAR WHEN INPUT STATE QUALITY CHANGES? ===== */}
        <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
          <div className="text-sm font-bold text-indigo-700 dark:text-indigo-300 mb-1">Q5 · Does the same pattern hold when the input state quality changes?</div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mb-3">
            Same hand-off code, three ML models, only the incoming state changes. Hand-off at t_s = 1.0, mean over {transfer.n_waves} held-out waves.
          </p>
          <div className="flex items-center justify-center gap-2 flex-wrap text-center mb-3">
            <div className="rounded-lg px-3 py-1.5 text-[11px] font-bold bg-rose-50 dark:bg-rose-500/10 text-rose-600 dark:text-rose-400 border border-rose-100 dark:border-rose-500/20">STATE QUALITY AT HAND-OFF</div>
            <span className="text-slate-300 dark:text-slate-600">→</span>
            <div className="rounded-lg px-3 py-1.5 text-[11px] font-bold bg-indigo-50 dark:bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 border border-indigo-100 dark:border-indigo-500/20">RESULT AFTER NUMERICAL CONTINUATION</div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead><tr className="text-[11px] uppercase tracking-wide text-slate-400 dark:text-slate-500 text-left">
                <th className="py-1.5 pr-3">model</th>
                <th className="py-1.5 px-3 text-right">state error at hand-off</th>
                <th className="py-1.5 px-3 text-right">pure-ML tail</th>
                <th className="py-1.5 px-3 text-right">hybrid tail</th>
                <th className="py-1.5 pl-3 text-right">benefit</th>
              </tr></thead>
              <tbody>
                {MODEL_ROWS.map((r) => (
                  <tr key={r.m} className="border-t border-slate-100 dark:border-slate-700">
                    <td className="py-2 pr-3 font-semibold text-slate-700 dark:text-slate-200">{r.m}</td>
                    <td className="py-2 px-3 text-right text-slate-500 dark:text-slate-400">{r.es}%</td>
                    <td className="py-2 px-3 text-right text-rose-600 dark:text-rose-400">{r.pml}%</td>
                    <td className="py-2 px-3 text-right text-indigo-600 dark:text-indigo-400">{r.hyb}%</td>
                    <td className="py-2 pl-3 text-right font-bold text-emerald-600 dark:text-emerald-400">{r.ben}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-3 text-[11px] leading-snug text-indigo-700 dark:text-indigo-300 bg-indigo-50/60 dark:bg-indigo-500/10 border border-indigo-100 dark:border-indigo-500/20 rounded-lg px-2.5 py-1.5">
            <span className="font-bold">→ Deduction: </span>Better state handed over → better hybrid result, true for all three tested architectures.
          </div>
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-5 items-center rounded-xl border border-slate-100 dark:border-slate-700 bg-slate-50 dark:bg-slate-700/30 p-4">
            <img src={EVAL_ROBUST[0].src} alt={EVAL_ROBUST[0].title} loading="lazy"
              className="w-full rounded-lg border border-slate-200 dark:border-slate-700 bg-white" />
            <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
              On a severely out-of-distribution wave, the ML state is already badly corrupted before the hand-off.{" "}
              <span className="font-medium text-slate-700 dark:text-slate-200">Numerical continuation can limit further damage, but it cannot reconstruct information the ML has already lost.</span>
            </p>
          </div>
        </div>

        {/* ===== DEEP TECHNICAL VALIDATION, collapsed by default, not the headline ===== */}
        <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
          <button onClick={() => setShowDeepVal((v) => !v)}
            className="w-full text-left text-sm font-bold text-slate-700 dark:text-slate-200 flex items-center justify-between">
            Deep technical validation
            <span className="text-slate-400 dark:text-slate-500 text-xs">{showDeepVal ? "▾" : "▸"}</span>
          </button>
          {showDeepVal && (
            <div className="mt-4">
              <p className="text-xs text-slate-500 dark:text-slate-400 mb-2">
                A restart-safety stress test, run on a deliberately weakened restart (drops the production scheme&apos;s
                de-aliasing). Pushed to sharper waves (higher cell Reynolds number), the verified restart stays accurate;
                the weakened one climbs past a 1% target and eventually goes unstable.
              </p>
              <div className="max-w-2xl mx-auto">
                <SafetyChart data={SAFETY} cross={SAFETY_CROSS} />
              </div>
              <div className="flex gap-4 flex-wrap justify-center text-center text-[11px] text-slate-400 dark:text-slate-500 mt-1">
                <span className="text-emerald-600 dark:text-emerald-400">verified restart, stays ~10<sup>−11</sup></span>
                <span className="text-rose-500">weakened restart (no de-aliasing)</span>
                <span className="text-amber-500">-- 1% target · crosses at Re≈{SAFETY_CROSS.toFixed(1)}</span>
              </div>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-2 text-center" title="Cell Reynolds number is a difficulty indicator combining wave strength, grid spacing and viscosity.">
                Re_cell ≈ {SAFETY_CROSS.toFixed(1)} is a difficulty indicator (hover), not a physical constant.
              </p>
              <p className="text-[11px] text-amber-600 dark:text-amber-400 mt-1 font-medium text-center">
                This is the weakened restart&apos;s measured failure boundary on this experiment, not a universal limit of the verified solver.
              </p>
            </div>
          )}
        </div>

        {/* ===== FOOTER · CONCISE RESULT MAP ===== */}
        <div className="rounded-2xl border border-emerald-200 dark:border-emerald-500/30 bg-emerald-50/60 dark:bg-emerald-500/10 p-5">
          <div className="grid grid-cols-1 sm:grid-cols-5 gap-3 text-center">
            {[
              { q: "Does it help?", a: `${AGG.fnoTail1}% → ${AGG.hybTail2}%` },
              { q: "Can I restart faithfully?", a: "diff 0.0" },
              { q: "Is the seam continuous?", a: "jump 0" },
              { q: "Where does error come from?", a: "ML state" },
              { q: "How late can I switch?", a: `≈ ${AGG.boundary}` },
            ].map((c) => (
              <div key={c.q}>
                <div className="text-[10px] text-slate-500 dark:text-slate-400 leading-snug">{c.q}</div>
                <div className="text-sm font-bold text-emerald-700 dark:text-emerald-300 mt-0.5">✓ {c.a}</div>
              </div>
            ))}
          </div>
        </div>
      </div>)}
      {tab === "live" && (<div className="space-y-6">
        {/* RUN IT YOURSELF, real backend */}
        <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-1 flex items-center gap-2">
          </div>
          {err && <div className="mb-3 text-sm text-rose-600 dark:text-rose-300 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 rounded-lg px-3 py-2">{err}</div>}
          <div className="grid grid-cols-[300px_1fr] gap-5">
            <div className="space-y-3">
              <div className="flex gap-2">
                {MODELS.map((m) => (
                  <button key={m} onClick={() => setModel(m)}
                    className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium border ${model === m
                      ? "bg-indigo-600 text-white border-indigo-600" : inactiveBtn}`}>{m}</button>
                ))}
              </div>
              {model === "PINN" ? (
                <select value={pinnIndex} onChange={(e) => setPinnIndex(+e.target.value)}
                  className="w-full bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-2 text-sm">
                  {Array.from({ length: meta?.n_pinn_ics || 0 }, (_, k) => (
                    <option key={k} value={k}>trained wave #{k}</option>
                  ))}
                </select>
              ) : (
                <div className="space-y-3 text-sm">
                  <div className="flex gap-2">
                    {[["real", "Held-out test IC"], ["synthetic", "Random shape"]].map(([v, label]) => (
                      <button key={v} onClick={() => setSource(v)}
                        className={`flex-1 px-2 py-1.5 rounded-lg text-xs font-bold border transition ${source === v
                          ? "bg-indigo-600 text-white border-indigo-600" : inactiveBtn}`}>{label}</button>
                    ))}
                  </div>
                  {source === "real" ? (
                    <>
                      <select value={ridx} onChange={(e) => setRidx(+e.target.value)}
                        className="w-full bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-2 text-sm">
                        {Array.from({ length: meta?.n_real_test_ics || 10 }, (_, k) => (
                          <option key={k} value={k}>Test IC #{900 + k}</option>
                        ))}
                      </select>
                      <div className={`text-[11px] ${muted}`}>
                        one of the {meta?.n_real_test_ics || 10} held-out waves the aggregate hand-off
                        numbers (n = {AGG.nIC}) were measured on, not a random draw
                      </div>
                    </>
                  ) : (
                    <>
                      <label className={`block ${muted}`}>
                        modes: {modes}
                        <input type="range" min="1" max="4" value={modes} onChange={(e) => setModes(+e.target.value)} className="w-full" />
                      </label>
                      <label className={`block ${muted}`}>
                        amplitude: {amplitude.toFixed(2)}
                        <input type="range" min="0.2" max="1.5" step="0.05" value={amplitude} onChange={(e) => setAmplitude(+e.target.value)} className="w-full" />
                      </label>
                      <div className={`text-[11px] ${muted}`}>
                        a fresh random shape, useful for exploring, but not one of the evaluated waves
                      </div>
                    </>
                  )}
                </div>
              )}
              <div className="flex gap-2">
                {[["manual", "Manual t_s"], ["trust", "Trust-fired"]].map(([v, label]) => (
                  <button key={v} onClick={() => setSwitchMode(v)}
                    className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium border ${switchMode === v
                      ? "bg-indigo-600 text-white border-indigo-600" : inactiveBtn}`}>{label}</button>
                ))}
              </div>
              {switchMode === "manual" && (
                <label className={`block text-sm ${muted}`}>
                  switch time t_s = {lts.toFixed(2)}
                  <input type="range" min="0.5" max="1.9" step="0.05" value={lts} onChange={(e) => setLts(+e.target.value)} className="w-full" />
                </label>
              )}
              <button onClick={run} disabled={running || !ic}
                className="w-full px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium text-sm transition">
                {running ? "Running…" : "Run the relay"}
              </button>
              {summary && (
                <div className="space-y-2">
                  <div className="grid grid-cols-2 gap-2">
                    <Stat label="pure-ML error [1,2]" value={`${(summary.ml_tail_1_2 * 100).toFixed(1)}%`} tone="red" />
                    <Stat label="hybrid error [1,2]" value={`${(summary.hybrid_tail_1_2 * 100).toFixed(1)}%`} tone="green" />
                    <Stat label="benefit" value={summary.benefit != null ? `${(summary.benefit * 100).toFixed(0)}%` : "-"} tone="indigo" />
                    <Stat label="numerical work" value={`${(summary.numerical_fraction * 100).toFixed(0)}%`} />
                  </div>
                  <Banner ok={summary.handoff_jump === 0}
                    text={`handoff jump = ${summary.handoff_jump}, continuous by construction`} />
                  <div className="rounded-lg border border-indigo-100 dark:border-indigo-500/25 bg-indigo-50/60 dark:bg-indigo-500/10 px-3 py-2">
                    <div className="text-[10px] font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">
                      This wave vs. the full held-out set
                    </div>
                    <p className="text-[11px] text-slate-600 dark:text-slate-300 mt-1 leading-relaxed">
                      {source === "real"
                        ? <>This is held-out <b>Test IC #{900 + ridx}</b>, one of the {AGG.nIC} evaluated waves. </>
                        : <>This is a synthetic shape, not one of the evaluated waves. </>}
                      Across all <b>{AGG.nIC}</b> held-out waves at the <b>t_s = 1.0</b> hand-off, tail error falls{" "}
                      <b>{AGG.fnoTail1}% → {AGG.hybTail2}%</b> (mean), improving <b>all {AGG.icsImproved}/{AGG.nIC}</b>.
                      {switchMode === "manual" && Math.abs(lts - 1.0) > 1e-6 &&
                        <> Your hand-off is at t_s = {lts.toFixed(2)}, so this run won&apos;t match the t_s = 1.0 mean exactly.</>}
                    </p>
                  </div>
                </div>
              )}
            </div>
            <div className="space-y-3">
              {/* live status tiles, update every frame, same read as the Story tab's tiles */}
              <div className="grid grid-cols-4 gap-2">
                <Stat label="time" value={lf ? `t = ${lf.t.toFixed(2)}` : "—"} />
                <Stat label="who carries it" value={lf ? (lf.switched ? "numerical" : "ML") : "—"} tone={lf ? (lf.switched ? "indigo" : "red") : "slate"} />
                <Stat label="pure-ML error now" value={lf ? `${(lf.ml_err * 100).toFixed(1)}%` : "—"} tone="red" />
                <Stat label="hybrid error now" value={lf ? `${(lf.hybrid_err * 100).toFixed(1)}%` : "—"} tone="indigo" />
              </div>
              <Card title={lf ? `t = ${lf.t.toFixed(2)}${lf.switched ? ", numerical carries it" : ", ML carries it"}` : "run to start"}
                subtitle="grey dashed = true solution · red = pure ML · indigo = hybrid (after the switch)">
                <LineChart
                  series={[
                    { x, y: lf ? lf.true : [], color: "#94a3b8", dashed: true },
                    { x, y: lf ? lf.ml : [], color: "#e11d48" },
                    { x, y: lf ? lf.hybrid : [], color: "#4f46e5", width: 2.5 },
                  ]}
                  xr={[-1, 1]} yr={lyr} h={185} xlabel="x" ylabel="u(x, t)" />
              </Card>
              <Card title="Error over time (live run)"
                subtitle="red = pure ML error · indigo = hybrid error · dashed line = the switch">
                <LineChart
                  series={[
                    { x: hist.map((f) => f.t), y: hist.map((f) => f.ml_err), color: "#e11d48" },
                    { x: hist.map((f) => f.t), y: hist.map((f) => f.hybrid_err), color: "#4f46e5", width: 2.5 },
                  ]}
                  xr={[0, 2]} yr={[0, Math.max(0.3, ...hist.map((f) => f.ml_err))]}
                  vline={lf?.switch_t ?? null} vlineColor="#1e293b" vlineLabel="switch" h={150} xlabel="t" />
              </Card>
            </div>
          </div>
        </div>

        <p className="text-xs text-slate-400 dark:text-slate-500">
          On this page you control the switch yourself, including switching too early or too late on purpose, to see what happens.
          The Hybrid engine page works the opposite way: you just say how accurate you want the result, and the system decides when to switch for you.
        </p>
      </div>)}
    </div>
  );
}
