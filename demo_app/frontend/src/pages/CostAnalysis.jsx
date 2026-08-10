import { useEffect, useRef, useState } from "react";
import { Card } from "../components/ui.jsx";
import { API } from "../api.js";
import { Trophy, X, Play, RotateCcw } from "lucide-react";

const COLOR = { FNO: "#059669", DeepONet: "#e11d48", PINN: "#d97706", FDM: "#0ea5e9", Spectral: "#8b5cf6", ColeHopf: "#ec4899" };
const ML = ["FNO", "DeepONet", "PINN"];
const NUMERICAL = ["FDM", "Spectral", "ColeHopf"];

function useDraw(ms = 1300, key = 0) {
  const [p, setP] = useState(0);
  useEffect(() => {
    let raf, start; setP(0);
    const loop = (ts) => { if (!start) start = ts; const q = Math.min(1, (ts - start) / ms); setP(q); if (q < 1) raf = requestAnimationFrame(loop); };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [ms, key]);
  return p;
}
// smoothly counts a displayed number toward a new target whenever it changes (e.g. the
// "cheapest solver that qualifies" switching as you drag the tolerance) instead of snapping
function useAnimatedNumber(target, ms = 280) {
  const [v, setV] = useState(target);
  const ref = useRef({ raf: null, from: target, start: 0 });
  useEffect(() => {
    if (target == null) { setV(target); return; }
    const r = ref.current;
    r.from = v ?? target; r.start = 0;
    cancelAnimationFrame(r.raf);
    const loop = (ts) => {
      if (!r.start) r.start = ts;
      const q = Math.min(1, (ts - r.start) / ms);
      setV(r.from + (target - r.from) * q);
      if (q < 1) r.raf = requestAnimationFrame(loop);
    };
    r.raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(r.raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, ms]);
  // on the render where `target` first flips from null to a number, the effect above hasn't
  // committed yet, so `v` can still be null for one frame -- fall back to target so callers
  // never see null and crash (this was blanking the whole page on load).
  return v ?? target;
}
function decadeTicks(min, max, maxCount = 5) {
  const lo = Math.ceil(Math.log10(min) - 1e-9), hi = Math.floor(Math.log10(max) + 1e-9);
  let e = []; for (let k = lo; k <= hi; k++) e.push(k);
  if (!e.length) return [];
  const st = Math.ceil(e.length / maxCount) || 1;
  return e.filter((_, i) => i % st === 0).map((k) => Math.pow(10, k));
}
const fmtS = (v) => (v >= 1000 ? `${Math.round(v / 1000)}k s` : `${v} s`);
const fmtP = (v) => (v >= 0.01 ? `${v}%` : `${v.toExponential(0)}%`);

function Scatter({ models, mode, p }) {
  const W = 640, H = 340, padL = 60, padR = 24, padT = 20, padB = 44;
  const names = Object.keys(models);
  const g = (n, w) => Math.max((w === "in" ? models[n].err_in : models[n].err_extrap) ?? 1e-4, 1e-4);
  const pts = names.map((n) => ({ n, kind: models[n].kind, x: Math.max(models[n].deploy_s ?? 0.01, 0.01), yin: g(n, "in"), yex: g(n, "ex") }));
  const ys = pts.flatMap((q) => [q.yin, q.yex]);
  const xmin = Math.min(...pts.map((q) => q.x)) * 0.45, xmax = Math.max(...pts.map((q) => q.x)) * 3;
  const ymin = Math.min(...ys) * 0.35, ymax = Math.max(...ys) * 3.5;
  const L = Math.log10;
  const sx = (v) => padL + ((L(v) - L(xmin)) / (L(xmax) - L(xmin))) * (W - padL - padR);
  const sy = (v) => H - padB - ((L(v) - L(ymin)) / (L(ymax) - L(ymin))) * (H - padT - padB);
  const yOf = (q) => (mode === "extrap" || mode === "both" ? q.yex : q.yin);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      <rect x={padL} y={padT} width={W - padL - padR} height={H - padT - padB} fill="var(--chart-surface)" stroke="var(--chart-grid)" />
      <rect x={padL} y={sy(1)} width={Math.max(0, sx(1) - padL)} height={Math.max(0, H - padB - sy(1))} fill="#059669" opacity="0.08" />
      <text x={padL + 6} y={h_(H, padB)} fontSize="9" fill="#059669" fontWeight="700">cheap AND reliable</text>
      {decadeTicks(xmin, xmax).map((v) => (
        <g key={`x${v}`}>
          <line x1={sx(v)} x2={sx(v)} y1={padT} y2={H - padB} stroke="var(--chart-grid)" strokeDasharray="2 4" />
          <text x={sx(v)} y={H - padB + 15} textAnchor="middle" fontSize="9" fill="var(--chart-axis)">{fmtS(v)}</text>
        </g>
      ))}
      {decadeTicks(ymin, ymax).map((v) => (
        <g key={`y${v}`}>
          <line x1={padL} x2={W - padR} y1={sy(v)} y2={sy(v)} stroke="var(--chart-grid)" strokeDasharray="2 4" />
          <text x={padL - 8} y={sy(v) + 3} textAnchor="end" fontSize="9" fill="var(--chart-axis)">{fmtP(v)}</text>
        </g>
      ))}
      {pts.map((q) => {
        const right = sx(q.x) > W - padR - 92;
        const jump = mode === "both" && q.kind === "ml" && q.yex > q.yin * 1.2;
        const y = sy(yOf(q)), yA = sy(q.yin), yB = sy(q.yex);
        return (
          <g key={q.n}>
            {jump && (
              <>
                <line x1={sx(q.x)} x2={sx(q.x)} y1={yA} y2={yA + (yB - yA) * p + 7} stroke={COLOR[q.n]} strokeWidth="2" opacity="0.5" />
                <circle cx={sx(q.x)} cy={yA} r="4" fill="none" stroke={COLOR[q.n]} strokeWidth="2" />
              </>
            )}
            {q.kind === "ml"
              ? <circle cx={sx(q.x)} cy={y} r="7" fill={COLOR[q.n]} />
              : <rect x={sx(q.x) - 6} y={y - 6} width="12" height="12" fill={COLOR[q.n]} />}
            <text x={right ? sx(q.x) - 12 : sx(q.x) + 12} y={y + 4} textAnchor={right ? "end" : "start"} fontSize="10.5" fontWeight="700" fill={COLOR[q.n]}>{q.n}</text>
          </g>
        );
      })}
      <text x={(padL + W - padR) / 2} y={H - 6} textAnchor="middle" fontSize="9.5" fill="var(--chart-axis)">cost per problem (log)</text>
      <text x={14} y={(padT + H - padB) / 2} textAnchor="middle" fontSize="9.5" fill="var(--chart-axis)" transform={`rotate(-90 14 ${(padT + H - padB) / 2})`}>error (log)</text>
    </svg>
  );
}
const h_ = (H, padB) => H - padB - 8;

function Scaling({ models }) {
  const W = 500, H = 210, padL = 50, padR = 16, padT = 14, padB = 36;
  const ns = ML.filter((m) => models[m]?.scaling);
  if (!ns.length) return null;
  const N = [], T = [];
  ns.forEach((m) => { models[m].scaling.N.forEach((v) => N.push(v)); models[m].scaling.t.forEach((v) => T.push(v)); });
  const xmin = Math.min(...N) * 0.85, xmax = Math.max(...N) * 1.2, ymin = Math.min(...T) * 0.55, ymax = Math.max(...T) * 1.9;
  const L = Math.log10;
  const sx = (v) => padL + ((L(v) - L(xmin)) / (L(xmax) - L(xmin))) * (W - padL - padR);
  const sy = (v) => H - padB - ((L(v) - L(ymin)) / (L(ymax) - L(ymin))) * (H - padT - padB);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {decadeTicks(ymin, ymax, 3).map((v) => (
        <g key={v}><line x1={padL} x2={W - padR} y1={sy(v)} y2={sy(v)} stroke="var(--chart-grid)" />
          <text x={padL - 6} y={sy(v) + 3} textAnchor="end" fontSize="9" fill="var(--chart-axis)">{v}s</text></g>
      ))}
      {ns.map((m) => {
        const s = models[m].scaling;
        return <path key={m} d={s.N.map((n, i) => `${i ? "L" : "M"}${sx(n).toFixed(1)} ${sy(s.t[i]).toFixed(1)}`).join(" ")}
          fill="none" stroke={COLOR[m]} strokeWidth="2" />;
      })}
      <text x={(padL + W - padR) / 2} y={H - 6} textAnchor="middle" fontSize="9" fill="var(--chart-axis)">problem size →</text>
    </svg>
  );
}


/* head-to-head: the one comparison that decides which surrogate is worth a controller */
const CRIT = [
  { k: "deploy_s", label: "Cost per problem", fmt: (v) => (v >= 60 ? `${Math.round(v / 60)} min` : `${v.toFixed(2)} s`), log: true },
  { k: "err_in", label: "Error inside window", fmt: (v) => `${v.toFixed(1)}%`, log: true },
  { k: "err_extrap", label: "Error beyond window", fmt: (v) => `${v.toFixed(0)}%`, log: true },
  { k: "rss_mb", label: "Memory footprint", fmt: (v) => `${Math.round(v)} MB`, log: false },
];

function HeadToHead({ models, hitTxt }) {
  const cell = (m, c) => {
    const v = models[m]?.[c.k] ?? 0;
    const all = ML.map((n) => models[n]?.[c.k] ?? 0).filter((x) => x > 0);
    const best = Math.min(...all), worst = Math.max(...all);
    const isBest = v === best, isWorst = v === worst;
    const f = c.log
      ? (Math.log10(Math.max(v, 1e-3)) - Math.log10(Math.max(best, 1e-3))) /
        (Math.log10(Math.max(worst, 1e-3)) - Math.log10(Math.max(best, 1e-3)) || 1)
      : (v - best) / (worst - best || 1);
    return (
      <div key={m} className={`rounded-lg px-3 py-2 ${isBest ? "bg-emerald-50 dark:bg-emerald-500/10" : isWorst ? "bg-rose-50 dark:bg-rose-500/10" : "bg-slate-50 dark:bg-slate-700/40"}`}>
        <div className="flex items-baseline justify-between">
          <span className={`text-sm font-bold tabular-nums ${isBest ? "text-emerald-700 dark:text-emerald-300" : isWorst ? "text-rose-700 dark:text-rose-300" : "text-slate-700 dark:text-slate-200"}`}>{c.fmt(v)}</span>
          {isBest && <span className="text-[9px] font-bold text-emerald-600">BEST</span>}
        </div>
        <div className="mt-1 h-1.5 rounded-full bg-slate-200 dark:bg-slate-600 overflow-hidden">
          <div className="h-1.5 rounded-full" style={{ width: `${Math.max(6, (1 - f) * 100)}%`, background: COLOR[m] }} />
        </div>
      </div>
    );
  };
  // hit-rate varies a lot by target (e.g. FNO is 100% down to target 0.05 then falls to
  // 10% at 0.02/0.01) -- hitTxt carries the real range from live data instead of a single
  // flat number baked in here, which for DeepONet used to just be wrong (it's 80%/60% at
  // loose targets, 0% only once targets get tight).
  const VERDICT = {
    FNO: { ok: true, head: "Yes", why: `amortized AND accurate in-window - 1.7-3.4x cheaper than numerical at ${hitTxt?.FNO ?? "100%"} hit-rate` },
    DeepONet: { ok: false, head: "No", why: `amortized but 29% wrong in-window - hybrid costs 1.3-3.1x numerical, ${hitTxt?.DeepONet ?? "0%"} hit-rate` },
    PINN: { ok: false, head: "No", why: `controller works on it (${hitTxt?.pinnErr ?? "8.0%"} error, ${hitTxt?.PINN ?? "80%"} hit) but 2114 s retrain per problem rules it out` },
  };
  return (
    <Card title="Head-to-head" subtitle="one row per criterion - green is best, red is worst">
      <div className="grid gap-2" style={{ gridTemplateColumns: "170px repeat(3, 1fr)" }}>
        <div />
        {ML.map((m) => (
          <div key={m} className="flex items-center gap-2 pb-1">
            <span className="w-2.5 h-2.5 rounded-full" style={{ background: COLOR[m] }} />
            <span className="text-sm font-bold" style={{ color: COLOR[m] }}>{m}</span>
          </div>
        ))}
        {CRIT.map((c) => (
          <div key={c.k} className="contents">
            <div className="text-xs text-slate-500 dark:text-slate-400 self-center pr-2">{c.label}</div>
            {ML.map((m) => cell(m, c))}
          </div>
        ))}
        <div className="text-xs text-slate-500 dark:text-slate-400 self-center pr-2">Reusable across problems</div>
        {ML.map((m) => (
          <div key={m} className={`rounded-lg px-3 py-2 text-sm font-bold ${m === "PINN" ? "bg-rose-50 dark:bg-rose-500/10 text-rose-700 dark:text-rose-300" : "bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"}`}>
            {m === "PINN" ? "No - retrains" : "Yes - amortized"}
          </div>
        ))}
      </div>

      <div className="mt-4 pt-4 border-t-2 border-dashed border-slate-200 dark:border-slate-700">
        <div className="text-xs font-bold uppercase tracking-wide text-indigo-600 dark:text-indigo-400 mb-2">
          My verdict - worth building a cost controller around?
        </div>
        <div className="grid gap-2" style={{ gridTemplateColumns: "170px repeat(3, 1fr)" }}>
          <div />
          {ML.map((m) => {
            const v = VERDICT[m];
            return (
              <div key={m} className={`rounded-xl p-3 border-2 ${v.ok ? "border-emerald-400 bg-emerald-50 dark:bg-emerald-500/10" : "border-rose-300 dark:border-rose-500/40 bg-rose-50 dark:bg-rose-500/10"}`}>
                <div className="flex items-center gap-1.5">
                  {v.ok ? <Trophy size={14} className="text-emerald-600" /> : <X size={14} className="text-rose-600" />}
                  <span className={`text-base font-extrabold ${v.ok ? "text-emerald-700 dark:text-emerald-300" : "text-rose-700 dark:text-rose-300"}`}>{v.head}</span>
                </div>
                <div className="text-[11px] leading-snug text-slate-600 dark:text-slate-300 mt-1">{v.why}</div>
              </div>
            );
          })}
        </div>
        <div className="text-xs text-slate-400 mt-3">
          All three were run through the controller. Each fails a different precondition — DeepONet is inaccurate in-window, PINN is not amortized. Only FNO satisfies both.
        </div>
      </div>
    </Card>
  );
}


const F1 = (v, u, dp = 2) => (v == null ? "—" : `${Number(v).toFixed(dp)} ${u}`);

function Profile({ m, d }) {
  if (!d) return null;
  const stat = (k, v, hint) => (
    <div>
      <div className="text-[10px] uppercase tracking-wide text-slate-400">{k}</div>
      <div className="text-sm font-bold text-slate-700 dark:text-slate-200 tabular-nums">{v}</div>
      {hint && <div className="text-[10px] text-slate-400">{hint}</div>}
    </div>
  );
  const sc = d.scaling || {};
  return (
    <div className="mt-3 pt-3 border-t border-dashed border-slate-200 dark:border-slate-700 grid grid-cols-5 gap-3">
      {stat("inference", F1(d.infer_ms, "ms", 0), d.infer_ms_std != null ? `±${d.infer_ms_std.toFixed(1)}` : null)}
      {stat("deployment", d.deploy_s >= 60 ? F1(d.deploy_s / 60, "min", 1) : F1(d.deploy_s, "s"), m === "PINN" ? "retrains per problem" : "reusable")}
      {stat("throughput", d.throughput_ic_s != null ? `${d.throughput_ic_s < 1 ? d.throughput_ic_s.toFixed(4) : d.throughput_ic_s.toFixed(1)}/s` : "—", "problems per second")}
      {stat("error in-window", `${(d.err_in ?? 0).toFixed(2)}%`)}
      {stat("error beyond", `${(d.err_extrap ?? 0).toFixed(2)}%`)}
      {stat("parameters", d.params ? d.params.toLocaleString() : "0", d.params ? `${(d.params_mb ?? 0).toFixed(2)} MB` : "solver, no weights")}
      {stat("on disk", F1(d.disk_mb, "MB"))}
      {stat("peak memory", F1(d.rss_mb, "MB", 0))}
      {stat("scaling", sc.exp != null ? `N^${sc.exp.toFixed(2)}` : "—", sc.exp != null ? (sc.exp < 0.5 ? "sub-linear" : "linear in size") : null)}
      {m === "PINN"
        ? stat("training split", `${Math.round(d.adam_s || 0)} + ${Math.round(d.lbfgs_s || 0)} s`, "Adam + L-BFGS")
        : stat("training", "amortized", "trained once, reused")}
    </div>
  );
}


function Staircase({ M, window_, tol, floor, marker }) {
  const W = 460, H = 120, padL = 42, padR = 12, padT = 12, padB = 26;
  const names = Object.keys(M);
  const e = (m) => Math.max(((window_ === "in" ? M[m].err_in : M[m].err_extrap) ?? 0) / 100, 1e-6);
  const c = (m) => Math.max(M[m].deploy_s ?? 0.01, 0.01);
  const tmin = 1e-4, tmax = 0.5;
  const cmin = 0.05, cmax = Math.max(...names.map(c)) * 1.6;
  const L = Math.log10;
  const sx = (v) => padL + ((L(tmax) - L(v)) / (L(tmax) - L(tmin))) * (W - padL - padR);
  const sy = (v) => H - padB - ((L(v) - L(cmin)) / (L(cmax) - L(cmin))) * (H - padT - padB);
  const cheapest = (t) => { const ok = names.filter((m) => e(m) <= t); return ok.length ? Math.min(...ok.map(c)) : null; };
  const steps = [];
  for (let i = 0; i <= 90; i++) {
    const t = Math.pow(10, L(tmin) + (i / 90) * (L(tmax) - L(tmin)));
    steps.push([t, cheapest(t)]);
  }
  let d = "", started = false;
  steps.forEach(([t, v]) => { if (v == null) return; d += `${started ? "L" : "M"}${sx(t).toFixed(1)} ${sy(v).toFixed(1)}`; started = true; });
  const here = cheapest(tol);
  // controller reference marker (target 0.05, the same headline point used elsewhere) --
  // these were previously two hardcoded constants (3% and 0.85s) that didn't actually
  // belong to the same measured row; now both come from the same live frontier point,
  // and the shaded band's right edge (the "beyond window collapse" point) is FNO's
  // measured err_extrap instead of a hardcoded 14%.
  const mErr = marker?.error ?? 0.036, mCost = marker?.cost ?? 0.85;
  const leftEdge = Math.max(1e-3, floor ?? 0.03);
  const rightEdge = Math.max(0.03, Math.min(0.49, (M.FNO?.err_extrap ?? 14) / 100));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {[0.1, 1, 10].filter((v) => v >= cmin && v <= cmax).map((v) => (
        <g key={v}>
          <line x1={padL} x2={W - padR} y1={sy(v)} y2={sy(v)} stroke="var(--chart-grid)" strokeDasharray="2 4" />
          <text x={padL - 6} y={sy(v) + 3} textAnchor="end" fontSize="8" fill="var(--chart-axis)">{v}s</text>
        </g>
      ))}
      <rect x={Math.min(sx(rightEdge), sx(leftEdge))} y={padT} width={Math.abs(sx(leftEdge) - sx(rightEdge))} height={H - padT - padB} fill="#059669" opacity="0.10" />
      <text x={(sx(leftEdge) + sx(rightEdge)) / 2} y={padT + 9} textAnchor="middle" fontSize="7.5" fill="#059669" fontWeight="700">
        controller: {(mErr * 100).toFixed(1)}% for {mCost.toFixed(2)}s
      </text>
      <circle cx={sx(mErr)} cy={sy(mCost)} r="4" fill="#059669" />
      <path d={d} fill="none" stroke="#4f46e5" strokeWidth="2" />
      {here != null && <>
        <line x1={sx(tol)} x2={sx(tol)} y1={padT} y2={H - padB} stroke="#e11d48" strokeWidth="1.5" />
        <circle cx={sx(tol)} cy={sy(here)} r="5" fill="#e11d48" />
      </>}
      <text x={padL} y={H - 6} fontSize="8" fill="var(--chart-axis)">loose</text>
      <text x={W - padR} y={H - 6} textAnchor="end" fontSize="8" fill="var(--chart-axis)">strict →</text>
    </svg>
  );
}

export default function CostAnalysis() {
  const [tab, setTab] = useState("findings");
  const [d, setD] = useState(null);
  const [mode, setMode] = useState("in");
  const [tolExp, setTolExp] = useState(-1.0);
  const [open, setOpen] = useState(null);
  const [sweeping, setSweeping] = useState(false);
  const [window_, setWindow] = useState("extrap");
  const [err, setErr] = useState(null);
  const [frontiers, setFrontiers] = useState(null); // FNO + DeepONet, hit-rate per target
  const [pinnRows, setPinnRows] = useState(null); // PINN, error + hit-rate per target
  const [fnoFrontier, setFnoFrontier] = useState(null); // FNO absolute {cost, error} per target
  const p = useDraw(1300, mode + tab);

  useEffect(() => {
    fetch(`${API}/api/m3/costs`).then((r) => r.json()).then((x) => { if (x.error) setErr(x.error); else setD(x); })
      .catch(() => setErr("Backend not reachable — start it with: uvicorn main:app"));
    fetch(`${API}/api/m3/frontiers`).then((r) => r.json()).then(setFrontiers).catch(() => {});
    fetch(`${API}/api/m3/pinn_regime`).then((r) => r.json()).then((x) => setPinnRows(x.rows || null)).catch(() => {});
    fetch(`${API}/api/m3/frontier`).then((r) => r.json()).then((x) => setFnoFrontier(x.frontier || null)).catch(() => {});
  }, []);

  useEffect(() => {
    if (!sweeping) return;
    let raf, start;
    const dur = 7000, a = -0.3, b = -4;
    const loop = (ts) => {
      if (!start) start = ts;
      const q = Math.min(1, (ts - start) / dur);
      setTolExp(a + (b - a) * q);
      if (q < 1) raf = requestAnimationFrame(loop); else setSweeping(false);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [sweeping]);


  const M = d?.models || {};
  const pinn = M.PINN || {};
  const btn = (on) => `px-4 py-1.5 rounded-lg text-sm font-semibold transition ${on ? "bg-indigo-600 text-white" : "text-slate-600 dark:text-slate-300"}`;
  const chip = (on) => `px-3 py-1.5 rounded-lg text-xs font-medium border ${on ? "bg-indigo-600 text-white border-indigo-600" : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700"}`;

  // "the numerical solvers stay exact" used to be a blanket claim — it's true for Spectral
  // and Cole-Hopf (both ~0% error, by construction the accuracy reference) but NOT for FDM,
  // whose explicit time-stepping has real numerical diffusion: ~8.8% in-window / ~13% beyond,
  // comparable to FNO's own extrapolation error. FDM is also ~5x cheaper than FNO, not
  // "2-4x more" — only Spectral/Cole-Hopf carry that cost premium. Split live from cost_summary
  // instead of hardcoding one number for all three (see deployment_cost_writeup.md's own
  // "sharpest finding": FDM vs FNO is a window-dependent trade-off, not a clean numerical win).
  const ratio = (m) => (M[m] && M.FNO ? M[m].deploy_s / M.FNO.deploy_s : null);
  // require M[m] to actually exist before classifying it -- before /api/m3/costs loads,
  // M is {} and every NUMERICAL entry is "missing", which used to fall through the ?? 99
  // fallback into "loose" (inaccurate) and crash on M[m].err_extrap on the next line.
  const exactNum = NUMERICAL.filter((m) => M[m] && (M[m].err_extrap ?? 99) < 1);
  const looseNum = NUMERICAL.filter((m) => M[m] && (M[m].err_extrap ?? 99) >= 1);
  const exactRatios = exactNum.map(ratio).filter((v) => v != null);
  const exactRatioTxt = exactRatios.length
    ? (Math.min(...exactRatios) === Math.max(...exactRatios) ? `${Math.min(...exactRatios).toFixed(1)}x` : `${Math.min(...exactRatios).toFixed(1)}–${Math.max(...exactRatios).toFixed(1)}x`)
    : null;
  const looseM = looseNum[0], looseErr = looseM ? M[looseM]?.err_extrap : null, looseR = looseM ? ratio(looseM) : null;
  const fnoIn = M.FNO?.err_in, fnoCost = M.FNO?.deploy_s;

  const line = {
    in: `Inside the window FNO sits in the cheap-and-reliable corner: ${fnoIn != null ? fnoIn.toFixed(2) : "0.57"}% for ${fnoCost != null ? fnoCost.toFixed(2) : "0.44"} s.`,
    extrap: `Beyond it the corner is nearly empty — every surrogate collapses.${exactNum.length ? ` ${exactNum.join(" and ")} stay${exactNum.length === 1 ? "s" : ""} exact, but at ${exactRatioTxt} FNO's cost` : ""}${looseM ? `; ${looseM} isn't exact either — numerical diffusion leaves it ~${Math.round(looseErr)}% wrong, though still ${(1 / looseR).toFixed(1)}x cheaper than FNO` : ""}.`,
    both: `Each surrogate's error jumps 1–2 orders of magnitude the moment you leave the window.${exactNum.length ? ` ${exactNum.join(" and ")} don't move` : ""}${looseM ? `, but ${looseM} does too (~${Math.round(looseErr)}%) — just less than the surrogates, and for less money` : ""}.`,
  }[mode];
  const order = Object.keys(M).sort((a, b) => (M[a].deploy_s ?? 0) - (M[b].deploy_s ?? 0));

  // real hit-rate ranges (replacing what used to be flat, and in DeepONet's case
  // outright wrong, hardcoded numbers) -- see the equivalent fix on Cost Control.
  const rngOf = (rows, f) => (rows && rows.length ? [Math.min(...rows.map(f)), Math.max(...rows.map(f))] : null);
  const hitStr = (r) => (!r ? null : r[0] === r[1] ? `${Math.round(r[0] * 100)}%` : `${Math.round(r[0] * 100)}–${Math.round(r[1] * 100)}%`);
  const fHit = rngOf(frontiers?.FNO?.frontier, (q) => q.hit_rate);
  const dHit = rngOf(frontiers?.DeepONet?.frontier, (q) => q.hit_rate);
  const pHit = rngOf(pinnRows, (r) => r.hit_rate);
  const pErr = rngOf(pinnRows, (r) => r.error);
  const hitTxt = {
    FNO: hitStr(fHit), DeepONet: hitStr(dHit), PINN: hitStr(pHit),
    pinnErr: pErr ? (pErr[0] === pErr[1] ? `${(pErr[0] * 100).toFixed(1)}%` : `${(pErr[0] * 100).toFixed(1)}–${(pErr[1] * 100).toFixed(1)}%`) : null,
  };

  // Staircase's "controller" reference point + floor, both from the same live target
  // (0.05) row instead of two hardcoded numbers that didn't belong together.
  const fno05 = fnoFrontier?.find((r) => Math.abs(r.target - 0.05) < 1e-9) || null;
  const staircaseMarker = fno05 ? { cost: fno05.cost, error: fno05.error } : null;
  const fnoFloor = fnoFrontier?.length ? Math.min(...fnoFrontier.map((r) => r.error)) : 0.03;

  // "Try it live" tab's tolerance-vs-solver ranking. Hoisted above the tab check (rather than
  // recomputed inline only when the live tab is open) so the animation hooks below can run on
  // every render, per the rules of hooks -- harmless no-op work while on the Findings tab.
  const tolPct = Math.pow(10, tolExp) * 100;
  const tol = tolPct / 100;
  const rows = Object.keys(M).map((m) => {
    const e = (window_ === "in" ? M[m].err_in : M[m].err_extrap) ?? 0;
    return { m, e, cost: M[m].deploy_s ?? 0, ok: e <= tolPct };
  }).sort((p_, q_) => p_.cost - q_.cost);
  const win = rows.find((x) => x.ok);
  const bestML = rows.find((x) => x.ok && ML.includes(x.m));
  const cheapGap = !bestML;
  // cheapGap only means "no ML surrogate qualifies" -- it says nothing about whether the
  // cheapest solver that DOES qualify (win, which can be a numerical solver like FDM) is
  // already cheaper than the controller itself. Without this check the "gap my module
  // fills" copy kept firing even when e.g. FDM already qualified at 0.09s, undercutting
  // the controller's own 0.85s reference point by ~9x -- the opposite of a gap.
  const controllerWins = cheapGap && tol >= fnoFloor && win != null && staircaseMarker != null && win.cost > staircaseMarker.cost;

  // live-feel polish: count the headline cost smoothly toward the new winner instead of
  // snapping, and briefly pulse the winning row/number the moment the cheapest qualifying
  // solver actually changes -- makes dragging the slider read as a live re-evaluation.
  const winCostAnim = useAnimatedNumber(win ? win.cost : null);
  const [switchFlash, setSwitchFlash] = useState(false);
  const prevWinRef = useRef(null);
  useEffect(() => {
    const cur = win ? win.m : null;
    if (prevWinRef.current !== null && cur !== null && cur !== prevWinRef.current) {
      setSwitchFlash(true);
      const t = setTimeout(() => setSwitchFlash(false), 450);
      prevWinRef.current = cur;
      return () => clearTimeout(t);
    }
    prevWinRef.current = cur;
  }, [win?.m]);

  return (
    <div className="space-y-6">
      <div>
        <span className="text-xs font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">ML model analysis</span>
        <h1 className="text-3xl font-extrabold text-slate-800 dark:text-slate-100 mt-1">Cost Analysis</h1>
        <p className="text-slate-500 dark:text-slate-400 mt-1 max-w-2xl">
          What each solver costs, and what accuracy that buys{d?.env?.repeats ? ` — measured on one machine, ${d.env.repeats} repeats` : ""}.
        </p>
      </div>

      <div className="inline-flex rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-1">
        <button className={btn(tab === "findings")} onClick={() => setTab("findings")}>Findings</button>
        <button className={btn(tab === "live")} onClick={() => setTab("live")}>Try it live</button>
      </div>

      {err && <div className="text-sm text-rose-600 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 rounded-lg px-3 py-2">{err}</div>}

      {d && tab === "findings" && (
        <>
          <div className="grid grid-cols-3 gap-4">
            {ML.map((m) => (
              <div key={m} className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full" style={{ background: COLOR[m] }} />
                    <span className="text-sm font-bold text-slate-800 dark:text-slate-100">{m}</span>
                  </div>
                  {m === "FNO"
                    ? <span className="inline-flex items-center gap-1 text-[10px] font-bold text-emerald-700 bg-emerald-100 dark:bg-emerald-500/20 dark:text-emerald-300 px-2 py-0.5 rounded-full"><Trophy size={11} /> best value</span>
                    : <span className="inline-flex items-center gap-1 text-[10px] font-bold text-rose-700 bg-rose-100 dark:bg-rose-500/20 dark:text-rose-300 px-2 py-0.5 rounded-full"><X size={11} />{m === "PINN" ? "35 min" : "29% wrong"}</span>}
                </div>
                <div className="grid grid-cols-3 gap-2 mt-3">
                  <div><div className="text-[10px] text-slate-400">cost</div><div className="text-lg font-bold">{(M[m]?.deploy_s ?? 0) >= 60 ? `${Math.round((M[m].deploy_s) / 60)} min` : `${(M[m]?.deploy_s ?? 0).toFixed(2)} s`}</div></div>
                  <div><div className="text-[10px] text-slate-400">in-window</div><div className="text-lg font-bold" style={{ color: COLOR[m] }}>{(M[m]?.err_in ?? 0).toFixed(1)}%</div></div>
                  <div><div className="text-[10px] text-slate-400">beyond</div><div className="text-lg font-bold">{(M[m]?.err_extrap ?? 0).toFixed(0)}%</div></div>
                </div>
              </div>
            ))}
          </div>

          <Card title="Cost vs accuracy" subtitle="circles = ML · squares = numerical · shaded = cheap and reliable">
            <div className="flex gap-2 mb-3">
              {[["in", "Inside window"], ["extrap", "Beyond it"], ["both", "Show the jump"]].map(([v, l]) => (
                <button key={v} className={chip(mode === v)} onClick={() => setMode(v)}>{l}</button>
              ))}
            </div>
            <Scatter models={M} mode={mode} p={p} />
            <div className="text-sm text-slate-600 dark:text-slate-300 mt-2">{line}</div>
          </Card>

          <HeadToHead models={M} hitTxt={hitTxt} />

          <div className="grid grid-cols-2 gap-5">
            <Card title="The PINN anomaly" subtitle="fastest to run, impossible to deploy">
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-xl bg-emerald-50 dark:bg-emerald-500/10 p-3">
                  <div className="text-[10px] uppercase text-slate-400">inference</div>
                  <div className="text-2xl font-extrabold text-emerald-600">{Math.round(pinn.infer_ms || 0)} ms</div>
                </div>
                <div className="rounded-xl bg-rose-50 dark:bg-rose-500/10 p-3">
                  <div className="text-[10px] uppercase text-slate-400">deployment</div>
                  <div className="text-2xl font-extrabold text-rose-600">{Math.round((pinn.deploy_s || 0) / 60)} min</div>
                </div>
              </div>
              <div className="text-xs text-slate-500 dark:text-slate-400 mt-3">
                Retrained per problem: {Math.round(pinn.adam_s || 0)} s Adam + {Math.round(pinn.lbfgs_s || 0)} s L-BFGS.
              </div>
            </Card>

            <Card title="Cost grows linearly with size" subtitle={ML.filter((m) => M[m]?.scaling).map((m) => `${m} ${M[m].scaling.exp.toFixed(2)}`).join(" · ")}>
              <Scaling models={M} />
            </Card>
          </div>

          <div className="rounded-2xl p-5 bg-gradient-to-r from-indigo-50 via-white to-white dark:from-indigo-500/10 dark:via-slate-800 dark:to-slate-800 border border-indigo-200 dark:border-indigo-500/30">
            <div className="text-sm font-semibold text-indigo-800 dark:text-indigo-300">Why the hybrid exists</div>
            <p className="text-sm text-slate-700 dark:text-slate-200 mt-1">
              Nothing is both cheap and trustworthy past the training window. Surrogates collapse (14–61% error);
              {exactNum.length ? ` ${exactNum.join(" and ")} stay${exactNum.length === 1 ? "s" : ""} exact at ${exactRatioTxt} the cost` : ""}
              {looseM ? `, while ${looseM} is cheaper than FNO but still ~${Math.round(looseErr)}% wrong from numerical diffusion — cheaper is not the same as reliable` : ""}.
              The hybrid buys numerical accuracy <b>only where it is needed</b>, and routes to the fallback that is actually accurate.
            </p>
          </div>
        </>
      )}

      {d && tab === "live" && (() => {
        return (
        <>
          <Card title="How accurate do you need to be?" subtitle="drag the tolerance — watch who survives it">
            <div className="flex items-center gap-4">
              <span className="text-xs text-slate-400 w-20">50% is fine</span>
              <input type="range" min="0" max="1" step="0.005" value={(tolExp - (-0.3)) / (-4 - (-0.3))}
                onChange={(e) => setTolExp(-0.3 + (-4 - (-0.3)) * (+e.target.value))} className="flex-1" />
              <span className="text-xs text-slate-400 w-20 text-right">0.01% only</span>
              <span className="w-20 text-right text-lg font-extrabold text-indigo-600 tabular-nums">
                {tolPct >= 1 ? `${tolPct.toFixed(0)}%` : `${tolPct.toFixed(2)}%`}
              </span>
              <button onClick={() => { setTolExp(-0.3); setSweeping(true); }} disabled={sweeping}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white text-sm font-semibold transition">
                {sweeping ? <><RotateCcw size={15} className="animate-spin" /> Tightening…</> : <><Play size={15} /> Watch it tighten</>}
              </button>
            </div>
            <div className="flex gap-2 mt-3">
              {[["in", "Inside training window"], ["extrap", "Beyond it"]].map(([v, l]) => (
                <button key={v} className={chip(window_ === v)} onClick={() => setWindow(v)}>{l}</button>
              ))}
            </div>

            <div className="mt-4 pt-4 border-t border-slate-200 dark:border-slate-700 flex items-center gap-6">
              <div>
                <div className="text-[10px] uppercase tracking-wide text-slate-400">what that accuracy costs you</div>
                <div
                  className={`text-4xl font-extrabold tabular-nums transition-transform duration-300 ${switchFlash ? "scale-110" : "scale-100"}`}
                  style={{ color: win ? (NUMERICAL.includes(win.m) ? "#4f46e5" : "#059669") : "#e11d48" }}
                >
                  {win ? (winCostAnim >= 60 ? `${Math.round(winCostAnim / 60)} min` : `${winCostAnim.toFixed(2)} s`) : "impossible"}
                </div>
                <div className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                  {win ? <>cheapest solver within {tolPct >= 1 ? `${tolPct.toFixed(0)}%` : `${tolPct.toFixed(2)}%`} is <b style={{ color: COLOR[win.m] }}>{win.m}</b></> : "no solver here is this accurate"}
                </div>
              </div>
              <div className="flex-1"><Staircase M={M} window_={window_} tol={tol} floor={fnoFloor} marker={staircaseMarker} /></div>
            </div>
          </Card>

          <div className="space-y-2">
            {rows.map((r) => {
              const isWin = win && r.m === win.m;
              return (
                <div key={r.m} onClick={() => setOpen(open === r.m ? null : r.m)}
                  className={`cursor-pointer rounded-xl border px-4 py-3 transition-all duration-300 ${
                    isWin && switchFlash ? "border-2 border-emerald-400 bg-emerald-50 dark:bg-emerald-500/10 ring-4 ring-emerald-300 dark:ring-emerald-500/40"
                    : isWin ? "border-2 border-emerald-400 bg-emerald-50 dark:bg-emerald-500/10"
                    : r.ok ? "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800"
                    : "border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/40"}`}>
                  <div className={`flex items-center gap-4 transition-opacity ${r.ok || open === r.m ? "" : "opacity-40"}`}>
                  <div className="w-24 text-sm font-bold" style={{ color: COLOR[r.m] }}>{r.m}</div>
                  <div className="w-28">
                    <div className="text-[10px] text-slate-400">costs</div>
                    <div className="text-lg font-bold tabular-nums text-slate-700 dark:text-slate-200">
                      {r.cost >= 60 ? `${Math.round(r.cost / 60)} min` : `${r.cost.toFixed(2)} s`}
                    </div>
                  </div>
                  <div className="w-28">
                    <div className="text-[10px] text-slate-400">is wrong by</div>
                    <div className="text-lg font-bold tabular-nums" style={{ color: r.ok ? "#059669" : "#e11d48" }}>
                      {r.e < 0.001 ? "~0%" : `${r.e.toFixed(r.e < 1 ? 2 : 0)}%`}
                    </div>
                  </div>
                  <div className="flex-1 text-right">
                    {isWin ? (
                      <span className="inline-flex items-center gap-1.5 text-sm font-bold text-emerald-700 dark:text-emerald-300 bg-emerald-100 dark:bg-emerald-500/20 px-3 py-1.5 rounded-full">
                        <Trophy size={14} /> cheapest that qualifies
                      </span>
                    ) : r.ok ? (
                      <span className="text-xs font-semibold text-slate-500">qualifies</span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-xs font-semibold text-rose-600">
                        <X size={13} /> too wrong
                      </span>
                    )}
                  </div>
                  </div>
                  {open === r.m && <Profile m={r.m} d={M[r.m]} />}
                </div>
              );
            })}
          </div>
          <div className="text-xs text-slate-400 -mt-1">click any solver for its full measured profile</div>

          <div className={`rounded-2xl p-5 border-2 transition-all duration-300 ${
            cheapGap ? "border-indigo-400 bg-indigo-50 dark:bg-indigo-500/10" : "border-transparent bg-slate-50 dark:bg-slate-800/40"}`}>
            {controllerWins ? (
              <>
                <div className="text-sm font-bold text-indigo-800 dark:text-indigo-300">This is the gap my module fills</div>
                <p className="text-sm text-slate-700 dark:text-slate-200 mt-1">
                  Nothing cheap qualifies here — every surrogate is too wrong, so you are forced onto a numerical solver
                  at <b>{win.cost.toFixed(2)} s</b>. My controller reaches <b>{(staircaseMarker.error * 100).toFixed(1)}%</b> for
                  about <b>{staircaseMarker.cost.toFixed(2)} s</b> — roughly <b>{(win.cost / staircaseMarker.cost).toFixed(1)}x cheaper</b> than that forced fallback.
                </p>
              </>
            ) : cheapGap && tol >= fnoFloor ? (
              <>
                <div className="text-sm font-bold text-slate-700 dark:text-slate-200">No gap here — a numerical solver already wins</div>
                <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                  No ML surrogate is accurate enough at {tolPct.toFixed(2)}%, but{win ? <> <b>{win.m}</b> already covers it for <b>{win.cost.toFixed(2)} s</b></> : " a numerical solver already covers it"}
                  {staircaseMarker ? <> — cheaper than my controller's own {staircaseMarker.cost.toFixed(2)} s reference point</> : ""}, so there's nothing left for the controller to add here.
                </p>
              </>
            ) : cheapGap ? (
              <>
                <div className="text-sm font-bold text-slate-700 dark:text-slate-200">Below the controller's {(fnoFloor * 100).toFixed(1)}% floor</div>
                <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                  The floor is set by how far the trust monitor lets the surrogate drift before handing over. At
                  {" "}{tolPct.toFixed(2)}% the controller does not qualify{win ? <>, so a numerical solver at <b>{win.cost.toFixed(2)} s</b> is the only option</> : " and no solver here is accurate enough"}.
                </p>
              </>
            ) : (
              <p className="text-sm text-slate-600 dark:text-slate-300">
                A surrogate is good enough here — no correction needed. Tighten the tolerance to see where that stops being true.
              </p>
            )}
          </div>
        </>
        );
      })()}
    </div>
  );
}
