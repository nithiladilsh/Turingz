import { useEffect, useRef, useState } from "react";
import { Card, Stat, Banner, sci10, eToSup } from "../components/ui.jsx";
import { LineChart } from "../components/Charts.jsx";
import { getMeta, buildIC, pinnIC, realTestIC, runCoupling } from "../api.js";
import {
  RL_XS, RL_T, RL_FRAMES, RL_ERR_ML, RL_SWITCHES, RL_META,
} from "../couplingData.js";
import { Play, GitCommitHorizontal, Microscope, Code2, CheckCircle2 } from "lucide-react";
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

/* "How it's built", the hand-off mechanism, each step with the design reason (grounded in the code). */
const COUPLING_BUILD = [
  {
    n: 1, color: "#4f46e5", title: "Run the ML solver",
    what: "Roll out the fast ML surrogate (FNO / PINN / DeepONet) across the whole window.",
    chips: [{ label: "ML prediction stream", color: "#4f46e5" }],
    why: "The ML is cheap, so we let it carry the wave while it can be trusted, we only replace it once, at the switch."
  },
  {
    n: 2, color: "#0d9488", title: "Re-anchor at the switch",
    what: "Seed the team's pseudo-spectral solver with the EXACT ML state at the switch time t_s, and start there.",
    detail: "solve_from(u_ML(t_s)):  first numerical frame = u_ML(t_s)  →  jump = 0",
    why: "This makes the hand-off jump zero BY CONSTRUCTION, no blending, no interpolation. The first numerical frame IS the handed-over state, so the seam is continuous."
  },
  {
    n: 3, color: "#7c3aed", title: "Continue under the production scheme",
    what: "Advance with the team's verified pseudo-spectral solver, same grid, 2/3 de-aliasing, Nyquist zeroing, integrating-factor RK4.",
    chips: [{ label: "verified restart", color: "#7c3aed" }, { label: "= production solver", color: "#7c3aed" }],
    why: "It's the scheme everyone already trusts. The restart is proven bit-identical to it (rel diff 0.0 in verify_restart.py), so continuing changes nothing about the numerics."
  },
  {
    n: 4, color: "#e11d48", title: "Switch once, never hand back",
    what: "rollout() does a one-way hard switch at the first trust trigger, then stays on the numerical solver to the end.",
    detail: "if trust fires: switch once, never hand back",
    why: "Returning to ML would re-inject ML error. The numerical solver is injected, not hardcoded, so the same verified restart works with any trigger and any numerical backend."
  },
  {
    n: 5, color: "#059669", title: "Verify & decompose the error",
    what: "12 automated tests, plus an oracle restart from the TRUE state to separate the coupling's own error from the inherited ML error.",
    detail: "E_coupling ≈ 10⁻⁶  ≪  E_inherited  (dominates)",
    why: "The oracle proves the coupling itself adds almost nothing, all remaining hybrid error is inherited from the ML hand-off state, not produced by the switch."
  },
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

/* "Evaluation at a glance", the five strongest numbers, each from a committed file. */
const _STAB0 = stab.rows.find((r) => Math.abs(r.t_s - 1.0) < 1e-9) || stab.rows[0];
const GLANCE = [
  { v: `${AGG.reduction}%`, label: "Error reduction", cap: `${AGG.fnoTail1}% → ${AGG.hybTail2}% at t_s = 1.0 · ${AGG.nIC} waves`, metric: "relative L2 tail error", c: "#059669" },
  { v: `≈ ${AGG.boundary}`, label: "Viability boundary", cap: "latest switch meeting the pre-set rule", metric: "interpolated switch time t_s", c: "#7c3aed" },
  { v: `~${ORACLE_STR}`, label: "Oracle restart error", cap: "true-state restart adds ~nothing", metric: "relative L2 tail error, true-state restart", c: "#4f46e5" },
  { v: _STAB0.state_jump.toFixed(1), label: "State jump at switch", cap: "continuous hand-off, by construction", metric: "L2 norm of state discontinuity", c: "#0d9488" },
  { v: "0.0", label: "Restart verification", cap: "rel diff vs production solver", metric: "max relative diff vs full production trajectory", c: "#d97706" },
];

/* Cost end of the frontier, the latest viable measured switch (near the boundary ≈1.49). */
const _R14 = sweep.results.find((r) => Math.abs(r.t_s - 1.4) < 1e-9) || sweep.results[2];
const _R10n = (sweep.results.find((r) => Math.abs(r.t_s - 1.0) < 1e-9) || sweep.results[0]).numerical_fraction;
const AGG_COST = {
  ts: "1.4",
  hyb: (_R14.hybrid_tail * 100).toFixed(1),   // 6.8
  ben: (_R14.benefit * 100).toFixed(0),       // 66
  work: (_R14.numerical_fraction * 100).toFixed(0), // 31
  work10: (_R10n * 100).toFixed(0),           // 50
};

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

function FigCard({ src, title, note, metric, deduction, script }) {
  return (
    <figure className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4">
      <img src={src} alt={title} loading="lazy"
        className="w-full rounded-lg border border-slate-100 dark:border-slate-700 bg-white" />
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

const EVAL_CORE = [
  { src: "/module2_figures/fig1_error_over_time.png", title: "Error over time, FNO vs numerical vs hybrid", note: "Hand-off at t = 1, mean ± std over 100 held-out ICs. After the switch the hybrid tracks the numerical solution instead of drifting with the ML.", metric: "Relative L2 error at each time t, mean ± std over 100 held-out waves, the full curve over time, not just the tail scalar.", deduction: "Switching to the numerical solver halts the ML's continued extrapolation drift. It prevents further error growth, it does not undo error already present at the switch.", script: "make_figures.py" },
  { src: "/module2_figures/fig4_hybrid_vs_upper_bound.png", title: "Oracle decomposition, the coupling adds almost nothing", note: "Restarting from the TRUE state (upper bound) is negligibly better than from the FNO state (~10⁻⁶). All remaining hybrid error is inherited from the ML hand-off state.", metric: "Relative L2 tail error over [t_s, 2], comparing a restart from the ML state against a restart from the true state.", deduction: "Given the true state, the same solver's error is ~10⁻⁶, so the continuation itself adds almost nothing. The remaining hybrid error is inherited from the handed-over ML state, which sets the accuracy ceiling.", script: "make_figures.py" },
  { src: "/module2_figures/fig5_accuracy_vs_cost.png", title: "Accuracy vs cost", note: "Earlier hand-off = more numerical work, lower error. The cost proxy is the fraction of steps solved numerically (machine-independent).", metric: "Relative L2 tail error vs numerical work (fraction of steps solved numerically).", deduction: "Earlier switch = more numerical work but lower error; later = cheaper but less accurate. This is the trade-off curve Module 3 chooses a point on; Module 2 measures it.", script: "make_figures.py" },
];

const EVAL_ROBUST = [
  { src: "/module2_figures/fig8_ood_error_over_time.png", title: "Out-of-distribution wave", note: "A higher-frequency wave sin(6πx), beyond the trained band (modes 1–4). The hybrid still limits the damage after the hand-off, but cannot recover a state the ML has already lost.", metric: "Relative L2 error against the true Cole–Hopf solution, on an out-of-distribution input.", deduction: "The coupling is corrective, not reconstructive: it faithfully continues the state it receives, but cannot rebuild information the ML has already lost. This is exactly why switching early matters.", script: "ood_experiment.py" },
];

const EVAL_FIDELITY = [
  { src: "/module2_figures/handoff_stability_diagnostic.png", title: "Continuity across the switch", note: "State jump ≈ 0 and the Burgers PDE residual stays stable across the hand-off, vs a deliberately careless-restart negative control.", metric: "State jump = L2 norm of the discontinuity at switch. Residual = r = u_t + u·u_x − ν·u_xx, a physical-consistency check, not an error-vs-truth metric.", deduction: "The hand-off adds no discontinuity (jump = 0) and the continuation is more PDE-consistent than the ML (the residual drops, no spike). The seam is physically clean, not just numerically continuous.", script: "handoff_stability_diagnostic.py" },
  { src: "/module2_figures/restart_safety_boundary.png", title: "Restart-safety boundary (Re_cell ≈ 3.2)", note: "The careless restart fails past cell Reynolds number Re_cell ≈ 3.2. Includes a grid-refinement control (N = 1024/2048) and a reference-free high-k diagnostic.", metric: "Tail error vs cell Reynolds number Re_cell (a grid-resolution diagnostic), against a 1% error threshold.", deduction: "A restartable solver must keep the production scheme's safeguards: drop the 2/3 de-aliasing and it fails past Re_cell ≈ 3.2; the verified restart preserves them and stays ~10⁻¹¹. This proves the restart is faithful, not that it beats spectral methods.", script: "restart_safety_boundary.py" },
];

export default function CouplingPage() {
  const [tab, setTab] = useState("story");

  /* ---- the relay state ---- */
  const [tsIdx, setTsIdx] = useState(1);            // index into RL_SWITCHES (default t_s = 1.0)
  const [ph, setPh] = useState(0);                  // playhead frame
  const [playing, setPlaying] = useState(true);
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
          the wave while it can be trusted; my verified handoff passes it, mid-flight, zero jump, to the{" "}
          <span className="font-semibold text-indigo-600 dark:text-indigo-400">numerical solver</span> that carries it the rest of the way.{" "}
        </p>
      </div>

      {/* TABS */}
      <div className="flex gap-2">
        {[["story", "Story"], ["evidence", "Evidence"], ["built", "How it's built"], ["eval", "Evaluation"], ["live", "Try it live"]].map(([v, label]) => (
          <button key={v} onClick={() => setTab(v)}
            className={`px-4 py-2 rounded-xl text-sm font-medium border transition ${tab === v
              ? "bg-indigo-600 text-white border-indigo-600"
              : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700"}`}>
            {label}
          </button>
        ))}
      </div>

      {tab === "story" && (<div className="space-y-6">
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
              <div className="text-[10px] uppercase tracking-wide text-rose-500">pure-ML tail error · never hand off</div>
              <div className="text-xl font-extrabold text-rose-600 dark:text-rose-400">{(cm.mlTail * 100).toFixed(1)}%</div>
            </div>
            <div className="rounded-xl bg-indigo-50 dark:bg-indigo-500/10 border border-indigo-100 dark:border-indigo-500/20 px-3 py-2 text-center">
              <div className="text-[10px] uppercase tracking-wide text-indigo-500">hybrid tail error · hand off here</div>
              <div className="text-xl font-extrabold text-indigo-600 dark:text-indigo-400">{(cm.hyTail * 100).toFixed(cm.hyTail < 0.1 ? 2 : 1)}%</div>
            </div>
            <div className="rounded-xl bg-slate-50 dark:bg-slate-700/40 border border-slate-100 dark:border-slate-600/40 px-3 py-2 text-center">
              <div className="text-[10px] uppercase tracking-wide text-slate-400">numerical work · the cost</div>
              <div className="text-xl font-extrabold text-slate-700 dark:text-slate-200">{(cm.work * 100).toFixed(0)}%</div>
            </div>
            <div className="rounded-xl bg-emerald-50 dark:bg-emerald-500/10 border border-emerald-100 dark:border-emerald-500/20 px-3 py-2 text-center">
              <div className="text-[10px] uppercase tracking-wide text-emerald-600">jump at handoff</div>
              <div className="text-xl font-extrabold text-emerald-600 dark:text-emerald-400">{sw.jump === 0 ? "0" : eToSup(sw.jump)}</div>
            </div>
            <p className="col-span-4 text-[11px] text-slate-500 dark:text-slate-400 mt-2">
              Later hand-offs are <span className="font-medium">cheaper</span> (less numerical work) but{" "}
              <span className="font-medium">less accurate</span>, the state handed over is already degraded.
              The criterion above is about <span className="font-medium">accuracy</span>, not cost: how much that
              accuracy is worth paying for is Module 3&apos;s decision.
            </p>
          </div>
        </div>
                {/* FRONTIER ENDS, accuracy vs cost, both from the committed sweep */}
        <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-1">Two ends of the same trade-off</div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mb-3">
            The hand-off time trades accuracy against cost. Switch early = most accurate but most numerical work.
            Switch near the viability boundary (≈ {AGG.boundary}) = the least work that still meets the 10% error bar.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="rounded-xl border border-emerald-100 dark:border-emerald-500/20 bg-emerald-50 dark:bg-emerald-500/10 p-3">
              <div className="text-[10px] uppercase tracking-wide text-emerald-600 dark:text-emerald-400">Accuracy end · t_s = 1.0</div>
              <div className="text-sm mt-1 text-slate-700 dark:text-slate-200"><b>{AGG.hybTail2}%</b> error · <b>{AGG.reduction}%</b> benefit</div>
              <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">{AGG_COST.work10}% of the run solved numerically</div>
            </div>
            <div className="rounded-xl border border-indigo-100 dark:border-indigo-500/20 bg-indigo-50 dark:bg-indigo-500/10 p-3">
              <div className="text-[10px] uppercase tracking-wide text-indigo-600 dark:text-indigo-400">Cost end · t_s = {AGG_COST.ts} (latest viable)</div>
              <div className="text-sm mt-1 text-slate-700 dark:text-slate-200"><b>{AGG_COST.hyb}%</b> error · <b>{AGG_COST.ben}%</b> benefit</div>
              <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">{AGG_COST.work}% numerical, ~40% less work, still under 10%</div>
            </div>
          </div>
        </div>

        {/* aggregate headline, pins the real result to the Story tab so it stands alone
      <div className="rounded-2xl border border-emerald-200 dark:border-emerald-500/30 bg-emerald-50/60 dark:bg-emerald-500/10 px-4 py-3 flex flex-wrap items-center gap-x-6 gap-y-1.5">
        <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-700 dark:text-emerald-300">Aggregate · n = {AGG.nIC} held-out</span>
        <span className="text-sm text-slate-700 dark:text-slate-200">tail error <b className="text-rose-600 dark:text-rose-400">{AGG.fnoTail1}%</b> → <b className="text-indigo-600 dark:text-indigo-400">{AGG.hybTail2}%</b></span>
        <span className="text-sm text-slate-700 dark:text-slate-200"><b className="text-emerald-600 dark:text-emerald-400">{AGG.reduction}%</b> reduction</span>
        <span className="text-sm text-slate-700 dark:text-slate-200"><b>{AGG.icsImproved}/{AGG.nIC}</b> waves improved</span>
        <span className="text-[11px] text-slate-400 dark:text-slate-500 ml-auto">hand-off tₛ = 1.0 · the animation above is one representative wave</span>
      </div> */}

        {/* wave + error, reacting to the same t_s */}
        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4">
            <div className="flex items-center justify-between">
              <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">
                t = {tNow.toFixed(2)} · {switched
                  ? <span className="text-indigo-500">numerical carries it</span>
                  : <span className="text-rose-500">ML carries it</span>}
              </div>
              <button onClick={() => setPlaying((p) => !p)}
                className="text-xs px-2 py-1 rounded border border-slate-200 dark:border-slate-700 text-slate-500 dark:text-slate-300">
                {playing ? "pause" : "play"}
              </button>
            </div>
            <RelayWave frame={frame} hyField={hyField} switched={switched} />
            <div className="flex gap-4 text-[11px] text-slate-400 dark:text-slate-500">
              <span>true (dashed)</span>
              <span className="text-rose-500">pure ML{switched ? " (ghost, what would have happened)" : ""}</span>
              {switched && <span className="text-indigo-500 font-medium">hybrid</span>}
            </div>
          </div>
          <Card title="The cost of your decision"
            subtitle="red = never hand off · indigo = your relay, identical until t_s, then pinned">
            <LineChart
              series={[
                { x: RL_T, y: RL_ERR_ML, color: "#e11d48", width: 2 },
                { x: RL_T, y: hyErrCurve, color: "#4f46e5", width: 2.5 },
              ]}
              xr={[0, 2]} yr={[0, Math.max(...RL_ERR_ML) * 1.08]}
              vline={sw.ts} h={215} xlabel="t" ylabel="relative L2 error" />
          </Card>
        </div>



      </div>)}
      {tab === "evidence" && (<div className="space-y-6">
        {/* NOVELTY + KEY NUMBERS, the first thing the examiner sees on this tab */}
        <div className="rounded-2xl border-2 border-indigo-300 dark:border-indigo-500/40 bg-indigo-50/60 dark:bg-indigo-500/10 p-5">
          <div className="text-xs font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400 mb-1">My contribution</div>
          <p className="text-sm text-slate-800 dark:text-slate-100 font-medium">
            I made the numerical solver able to <span className="text-indigo-600 dark:text-indigo-400">restart from the ML solver&apos;s state mid-run</span>,
            proved that restart is <span className="text-indigo-600 dark:text-indigo-400">identical to the original solver</span>, and measured{" "}
            <span className="text-indigo-600 dark:text-indigo-400">when the hand-off is still worth doing</span>.
          </p>
          <div className="flex flex-wrap gap-2 mt-3 text-[11px] font-semibold">
            {["restart vs original = 0.0", "jump at switch = 0", `${AGG.fnoTail1}% → ${AGG.hybTail2}% (mean)`, `improved on all ${AGG.nIC} held-out ICs`, `last useful switch ≈ ${AGG.boundary}`].map((t) => (
              <span key={t} className="px-2.5 py-1 rounded-full bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200">{t}</span>
            ))}
          </div>
        </div>
        {/* AGGREGATE RESULT, the report headline, kept distinct from the single-wave animation */}
        <div className="rounded-2xl border-2 border-indigo-200 dark:border-indigo-500/30 bg-white dark:bg-slate-800 p-5">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sm font-semibold text-slate-800 dark:text-slate-100">Primary result, hand-off at t_s = 1.0</span>
            <Badge kind="agg" /><Badge kind="committed" />
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mb-3">
            Mean over the full {AGG.nIC}-IC held-out test set (1,000 trajectories total: 800 train / 100 validation / 100 test). The figure above animates one representative wave, so its numbers differ slightly.
          </p>
          <div className="grid grid-cols-4 gap-3">
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
              <div className="text-[10px] uppercase tracking-wide text-slate-400">held-out ICs improved</div>
              <div className="text-2xl font-extrabold text-slate-700 dark:text-slate-200">{AGG.icsImproved} / {AGG.nIC}</div>
            </div>
          </div>
        </div>

        {/* THE FINDING, pinned to the committed n=100 aggregate (not the slider) */}
        <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3 flex items-center gap-2">
            <Microscope size={14} /> The finding: hybrid error ≈ the ML state error at hand-off
          </div>
          <div className="flex items-center justify-center gap-6 flex-wrap">
            <div className="text-center">
              <div className="text-[11px] text-slate-500 dark:text-slate-400">ML state error at hand-off</div>
              <div className="text-4xl font-extrabold text-rose-600 dark:text-rose-400">{AGG.stateErr}%</div>
            </div>
            <div className="text-4xl font-black text-slate-300 dark:text-slate-600">≈</div>
            <div className="text-center">
              <div className="text-[11px] text-slate-500 dark:text-slate-400">mean hybrid tail error over [t_s, 2]</div>
              <div className="text-4xl font-extrabold text-indigo-600 dark:text-indigo-400">{AGG.hybTail2}%</div>
            </div>
            <div className="max-w-sm text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
              Aggregate over {AGG.nIC} held-out waves at the t_s = 1.0 hand-off. Two different metrics, yet they
              match: the numerical continuation adds only ~{ORACLE_STR} of its own error (oracle control), so the
              handed-over ML state <span className="font-semibold text-slate-700 dark:text-slate-200">sets the accuracy ceiling</span> and the
              restart adds negligible additional error on this benchmark. The tail is slightly lower because viscosity damps the inherited error.
            </div>
          </div>
        </div>

        {/* ORACLE DECOMPOSITION, visualises where the error comes from (was text-only) */}
        <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-1 flex items-center gap-2">
            <Microscope size={14} /> Where the error comes from, oracle decomposition
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mb-2">
            Each bar is the error against the true answer, over the tail. Same numerical continuation, different starting states, the ML state vs the true state (log scale).
          </p>
          <OracleBars mlTail={_AGG_R0.fno_tail} hyTail={_AGG_R0.hybrid_tail} oracle={_ORACLE} />
          <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">
            Committed n = {AGG.nIC} aggregate at the t_s = 1.0 hand-off. Restarting from the{" "}
            <span className="font-semibold">true</span> state collapses to ~{ORACLE_STR}, the continuation
            is near-perfect. So the hybrid&apos;s error is{" "}
            <span className="font-medium text-slate-700 dark:text-slate-200">inherited from the ML state you hand over, not created
              by the switch</span>, which is why hybrid error tracks the ML state error at hand-off (the relationship pinned on the Story tab).
          </p>
        </div>

        {/* NOVELTY IN CODE, mirrors the Cost page's card, with M2's receipts */}
        <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
          {/* <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-2 flex items-center gap-2">
          <Code2 size={14} /> Why this is trustworthy (in code)
        </div> */}
          <p className="text-sm text-slate-700 dark:text-slate-200">
            The team&apos;s production spectral solver only ran <span className="font-mono text-[13px]">solve(u0)</span> from t = 0.
            My part is making it restartable from an arbitrary ML state, and verifying it two ways:
          </p>
          <div className="mt-2 rounded-lg bg-slate-50 dark:bg-slate-700/40 px-3 py-2 font-mono text-[13px] text-slate-700 dark:text-slate-200">
            solve_from(u_ML(tₛ), i_start)
          </div>
          <p className="text-xs text-slate-600 dark:text-slate-300 mt-2">
            Run from t = 0, the restart <span className="font-semibold text-emerald-600 dark:text-emerald-400">reproduces the production scheme</span> (relative difference &lt; 10⁻¹⁰); a separate seam test confirms it preserves the handed state exactly, so the <span className="font-semibold">jump is 0</span>.
          </p>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">
            The spectral scheme is the team&apos;s; the restartable wrapper <span className="font-mono">solve_from</span> and its
            verification are my contribution. The exact adapter this page calls is the one Module 3&apos;s runtime executes.
          </p>
        </div>

        {/* RESTART-SAFETY BOUNDARY, visualises the stress-test claim (was text-only) */}
        <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-1 flex items-center gap-2">
            <Microscope size={14} /> Why the restart is verified, it holds where a shortcut breaks
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mb-2">
            Push to sharper waves (higher cell Reynolds number). The verified restart stays accurate (~10<sup>−11</sup>); a shortcut restart that
            drops the de-aliasing climbs past the target and finally goes unstable.
          </p>
          <SafetyChart data={SAFETY} cross={SAFETY_CROSS} />
          <div className="flex gap-4 flex-wrap text-[11px] text-slate-400 dark:text-slate-500 mt-1">
            <span className="text-emerald-600 dark:text-emerald-400">verified restart, error ≈ 0 (stays ~10<sup>−11</sup>)</span>
            <span className="text-rose-500">shortcut restart (no de-aliasing)</span>
            <span className="text-amber-500">-- 1% target · shortcut crosses at Re≈3.2</span>
          </div>
        </div>

        {/* CONCLUSION, mirrors the Trust and Cost pages' closing verdict */}
        <div className="rounded-2xl p-5 bg-gradient-to-r from-emerald-50 via-white to-white dark:from-emerald-500/10 dark:via-slate-800 dark:to-slate-800 border border-emerald-200 dark:border-emerald-500/30">
          <div className="text-sm font-semibold text-emerald-800 dark:text-emerald-300 flex items-center gap-2"><CheckCircle2 size={16} /> Conclusion</div>
          <p className="text-sm text-slate-700 dark:text-slate-200 mt-1">
            The verified hand-off cuts tail error <b>{AGG.fnoTail1}% → {AGG.hybTail2}%</b> and improves <b>every one of the {AGG.nIC} held-out waves</b>. The restart
            matches the production solver exactly (rel diff 0.0) and the state jump is <b>zero by construction</b>, so the hand-off adds no error
            of its own, what remains is inherited from the ML state. This holds only while the hand-off is still viable (up to <b>t_s ≈ {RL_META.boundary}</b>);
            switch too late and even a perfect restart cannot recover. <span className="font-semibold">The module verifies and times the hand-off, it does not fix a bad ML stream.</span>
          </p>
        </div>

      </div>)}
      {tab === "built" && (<div className="space-y-6">
        <div className="rounded-2xl p-6 md:p-7 bg-gradient-to-r from-indigo-50 via-indigo-50 to-violet-50 dark:from-indigo-500/10 dark:via-indigo-500/10 dark:to-violet-500/10 border border-indigo-100 dark:border-indigo-500/25">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">Verified re-anchoring · state jump = 0 by construction</div>
          <h2 className="text-2xl font-bold mt-1 text-slate-800 dark:text-slate-100">How the hand-off is built</h2>
          <p className="text-sm text-slate-600 dark:text-slate-300 mt-2 leading-relaxed">
            The ML solver runs while it&apos;s trusted; at the switch, the production numerical solver is re-seeded with the exact
            ML state and continues to the end. The restart is proven identical to the trusted solver, and the hand-off adds no
            discontinuity, every step below is a control or a check, not a convenience.
          </p>
        </div>

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

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="rounded-2xl border border-emerald-200 dark:border-emerald-500/30 bg-emerald-50 dark:bg-emerald-500/10 p-5">
            <div className="text-sm font-bold text-emerald-800 dark:text-emerald-300">Verified, not assumed</div>
            <p className="text-sm text-slate-600 dark:text-slate-300 mt-1 leading-relaxed">
              The restart isn&apos;t &ldquo;close&rdquo; to the production solver, it&apos;s proven bit-identical (rel diff 0.0), and
              the zero-jump property is a test, not a claim. Every headline number has a passing test behind it.
            </p>
          </div>
          <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-700/30 p-5">
            <div className="text-sm font-bold text-slate-700 dark:text-slate-200">It times the hand-off, it doesn&apos;t fix the ML</div>
            <p className="text-sm text-slate-600 dark:text-slate-300 mt-1 leading-relaxed">
              The oracle decomposition shows the coupling adds ~10⁻⁶; the rest is inherited from the ML state. So the module
              guarantees a faithful, well-timed hand-off, it can&apos;t repair a bad ML prediction, and it doesn&apos;t claim to.
            </p>
          </div>
        </div>
      </div>)}

      {tab === "eval" && (<div className="space-y-6">
        <div className="rounded-2xl p-6 bg-gradient-to-r from-indigo-50 via-white to-white dark:from-indigo-500/10 dark:via-slate-800 dark:to-slate-800 border border-slate-200 dark:border-slate-700">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">How the module is evaluated</div>
          <h2 className="text-2xl font-bold mt-1 text-slate-800 dark:text-slate-100">The actual figures behind the results</h2>
          <p className="text-sm text-slate-600 dark:text-slate-300 mt-2 leading-relaxed">
            Every figure is generated by my own scripts from committed held-out data, scored against the exact
            Cole–Hopf answer <b>in analysis only</b>, the coupling itself never uses the true answer at runtime.
            The source script is named under each figure.
          </p>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {GLANCE.map((g) => (
            <div key={g.label} className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4">
              <div className="text-2xl font-extrabold" style={{ color: g.c }}>{g.v}</div>
              <div className="text-xs font-semibold text-slate-700 dark:text-slate-200 mt-1">{g.label}</div>
              <div className="text-[10px] text-slate-400 dark:text-slate-500 mt-0.5 leading-snug">{g.cap}</div>
              <div className="text-[10px] text-teal-600 dark:text-teal-400 mt-1 font-medium">Metric: {g.metric}</div>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-[1.5fr_1fr] gap-4 items-stretch">
        <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-1">Hand-off sweep, the numbers behind the curves</div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mb-3">
            Mean over the {AGG.nIC} held-out waves at each switch time. Viable = benefit ≥ 10% <b>and</b> hybrid tail &lt; 10% (frozen a priori). The <b>vs 10% bar</b> column is the hybrid tail&apos;s headroom under that 10% bar, in percentage points; it goes negative exactly when the hand-off stops being viable. Source: <span className="font-mono text-[11px]">handoff_sweep_results.json</span>.
          </p>
          <p className="text-[11px] text-teal-600 dark:text-teal-400 font-medium mb-3">Metric: tail columns are relative L2 error over [t_s, 2]; benefit = 1 − hybrid/pure-ML tail; numerical work = fraction of steps solved numerically.</p>
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
            Earlier hand-off gives more accuracy but more numerical work. Every measured switch still beats pure ML, but after t_s = 1.4 the hybrid tail error crosses the 10% acceptance bar. <span className="font-medium text-slate-700 dark:text-slate-200">Measured latest viable point = 1.4; interpolated boundary ≈ {AGG.boundary}.</span>
          </p>
        </div>
        <FigCard src="/module2_figures/fig2_switch_time_vs_benefit.png" title="Hand-off benefit vs when we switch" note="Benefit = 1 − hybrid/FNO across switch times, versus the pre-registered 10% benefit threshold (met across the whole swept range). The binding viability condition is the hybrid tail < 10%, the table's 'vs 10% bar' column." metric="Benefit = 1 − hybrid/pure-ML tail, both relative L2 tail error over [t_s, 2], plotted against switch time t_s." deduction="The later you wait, the more damaged the handed-over state, so benefit falls monotonically (93% at t_s = 1.0 down to 20% at 1.8). Switch early for the biggest gain." script="make_figures.py" />
        </div>

        <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-1">Final integration, the hand-off vs baselines</div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mb-3">
            Error over the extrapolation window [1, 2] on the {cmp.n} held-out waves, with the fraction of steps solved numerically.
            Source: <span className="font-mono text-[11px]">trust_hardswitch_compare.json</span>.
          </p>
          <p className="text-[11px] text-teal-600 dark:text-teal-400 font-medium mb-3">Metric: relative L2 error over the extrapolation window, same definition as the sweep table.</p>
          <div className="space-y-1.5">
            {BASELINES.map((b) => (
              <div key={b.m} className="grid grid-cols-[1fr_auto_auto] items-center gap-4 rounded-xl border border-slate-100 dark:border-slate-700 px-3 py-2">
                <span className="text-sm text-slate-700 dark:text-slate-200 flex items-center gap-2"><span className="w-2 h-2 rounded-full" style={{ background: b.c }} />{b.m}</span>
                <span className="text-sm font-bold w-16 text-right" style={{ color: b.c }}>{b.err}</span>
                <span className="text-xs text-slate-400 dark:text-slate-500 w-28 text-right">{b.work} numerical</span>
              </div>
            ))}
          </div>
          <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-3 leading-relaxed">
            This baseline feeds the raw trust flag straight in; it fires early (mean ≈ {Number(cmp.mean_trigger).toFixed(2)}), so it
            reaches {(cmp.means.trust * 100).toFixed(1)}% error but does ~{(cmp.trust_workload * 100).toFixed(0)}% of the work numerically.
            In the full system Module 3 turns a requested accuracy into the switch point, so the cost is a deliberate choice. Module 2&apos;s
            job is only to keep the hand-off faithful wherever it happens.
          </p>
        </div>

        <div>
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-1">Core hand-off results</div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mb-3">Earlier hand-offs give the largest gains; once the ML state is too inaccurate, numerical continuation halts further error growth but cannot undo error already present.</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {EVAL_CORE.map((f) => <FigCard key={f.src} {...f} />)}
          </div>
        </div>

        <div>
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-1">Robustness &amp; transfer</div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mb-3">Better handed-over state → better hybrid, for every architecture tested; in the OOD stress cases the continuation reduces subsequent error but cannot recover a severely corrupted hand-off state.</p>
          <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
            <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">Model-agnostic demonstrated; numerical backend decoupled by interface design</div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 mb-3">
              The identical <span className="font-mono text-[11px]">solve_from()</span> hand-off runs for FNO, PINN and DeepONet, only the ML
              array changes, the coupling code does not. Hand-off at t_s = 1.0, mean over {transfer.n_waves} held-out waves.
              Source: <span className="font-mono text-[11px]">transfer_models.py → transfer_models_results.json</span>.
            </p>
            <p className="text-[11px] text-teal-600 dark:text-teal-400 font-medium mb-3">Metric: state error at hand-off is the ML's own relative L2 error at t_s; tail and benefit columns as in the sweep table.</p>
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
              <span className="font-bold">→ Deduction: </span>Same coupling, three architectures: the better the state handed over, the better the hybrid. This relationship was observed across all three tested ML models (FNO, PINN, DeepONet), so it is a property of the coupling, not of one model.
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
            {EVAL_ROBUST.map((f) => <FigCard key={f.src} {...f} />)}
          </div>
        </div>

        <div>
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-1">Continuity, fidelity &amp; safety</div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mb-3">The switch is continuous and physically stable, the Burgers residual falls after the hand-off, and the verified restart holds where an approximate one fails.</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {EVAL_FIDELITY.map((f) => <FigCard key={f.src} {...f} />)}
          </div>
        </div>
      </div>)}
      {tab === "live" && (<div className="space-y-6">
        <div className="flex items-center gap-2"><Badge kind="live" /><span className="text-xs text-slate-400 dark:text-slate-500">runs the real M2Coupling adapter with the verified pseudo-spectral restart</span></div>
        {/* RUN IT YOURSELF, real backend */}
        <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-1 flex items-center gap-2">
            <Play size={14} /> Run it yourself, your wave, the real M2Coupling
          </div>
          <p className={`text-xs ${muted} mb-4`}>
            Everything above is precomputed from committed results on one held-out wave. Here the backend runs the real
            adapter live: any model, any wave, manual switch or Module 1&apos;s trust signal.
          </p>
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
              <Card title={lf ? `t = ${lf.t.toFixed(2)}${lf.switched ? ", numerical carries it" : ", ML carries it"}` : "run to start"}>
                <LineChart
                  series={[
                    { x, y: lf ? lf.true : [], color: "#94a3b8", dashed: true },
                    { x, y: lf ? lf.ml : [], color: "#e11d48" },
                    { x, y: lf ? lf.hybrid : [], color: "#4f46e5", width: 2.5 },
                  ]}
                  xr={[-1, 1]} yr={lyr} h={185} xlabel="x" ylabel="u(x, t)" />
              </Card>
              <Card title="error over time (live run)">
                <LineChart
                  series={[
                    { x: hist.map((f) => f.t), y: hist.map((f) => f.ml_err), color: "#e11d48" },
                    { x: hist.map((f) => f.t), y: hist.map((f) => f.hybrid_err), color: "#4f46e5", width: 2.5 },
                  ]}
                  xr={[0, 2]} yr={[0, Math.max(0.3, ...hist.map((f) => f.ml_err))]}
                  vline={lf?.switch_t ?? null} h={150} xlabel="t" />
              </Card>
            </div>
          </div>
        </div>

        <p className="text-xs text-slate-400 dark:text-slate-500">
          This page dissects the handoff, you control the switch and may deliberately hand off outside the viability criterion.
          The Hybrid engine page is the opposite: you set an accuracy target and the trust + control layer decides for you.
        </p>
      </div>)}
    </div>
  );
}
