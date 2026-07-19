import { useEffect, useState } from "react";
import { Card } from "../components/ui.jsx";
import { API } from "../api.js";
import { Trophy, X, Timer } from "lucide-react";

const COLOR = { FNO: "#059669", DeepONet: "#e11d48", PINN: "#d97706", FDM: "#64748b", Spectral: "#4f46e5", ColeHopf: "#7c3aed" };
const ML = ["FNO", "DeepONet", "PINN"];

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

function HeadToHead({ models }) {
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
  const VERDICT = {
    FNO: { ok: true, head: "Yes", why: "amortized AND accurate in-window - 1.7-3.4x cheaper than numerical at 100% hit-rate" },
    DeepONet: { ok: false, head: "No", why: "amortized but 29% wrong in-window - hybrid costs 1.3-3.1x numerical, 0% hit-rate" },
    PINN: { ok: false, head: "No", why: "controller works on it (8.0% error, 80% hit) but 2114 s retrain per problem rules it out" },
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
          Measured, not assumed: all three were run through the controller. Each fails a different precondition - DeepONet is inaccurate in-window, PINN is not amortized. Only FNO satisfies both.
        </div>
      </div>
    </Card>
  );
}

export default function CostAnalysis() {
  const [tab, setTab] = useState("findings");
  const [d, setD] = useState(null);
  const [mode, setMode] = useState("in");
  const [budget, setBudget] = useState(60);
  const [err, setErr] = useState(null);
  const p = useDraw(1300, mode + tab);

  useEffect(() => {
    fetch(`${API}/api/m3/costs`).then((r) => r.json()).then((x) => { if (x.error) setErr(x.error); else setD(x); })
      .catch(() => setErr("Backend not reachable — start it with: uvicorn main:app"));
  }, []);

  const M = d?.models || {};
  const pinn = M.PINN || {};
  const btn = (on) => `px-4 py-1.5 rounded-lg text-sm font-semibold transition ${on ? "bg-indigo-600 text-white" : "text-slate-600 dark:text-slate-300"}`;
  const chip = (on) => `px-3 py-1.5 rounded-lg text-xs font-medium border ${on ? "bg-indigo-600 text-white border-indigo-600" : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700"}`;
  const line = { in: "Inside the window FNO sits in the cheap-and-reliable corner: 0.57% for 0.44 s.",
                 extrap: "Beyond it the corner is empty — every surrogate collapses, the numerical solvers stay exact but cost 2–4× more.",
                 both: "Each surrogate's error jumps 1–2 orders of magnitude the moment you leave the window. The numerical solvers do not move." }[mode];
  const order = Object.keys(M).sort((a, b) => (M[a].deploy_s ?? 0) - (M[b].deploy_s ?? 0));
  const maxN = Math.max(1, ...order.map((m) => Math.floor(budget / Math.max(M[m].deploy_s ?? 1, 1e-6))));

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

          <HeadToHead models={M} />

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
              Nothing is both cheap and trustworthy past the training window. Surrogates are fast but collapse; numerical solvers stay exact at 2–4× the cost.
              The hybrid buys numerical accuracy <b>only where it is needed</b>.
            </p>
          </div>
        </>
      )}

      {d && tab === "live" && (
        <>
          <Card title="How much can each solver actually get done?" subtitle="drag a compute budget and see how many problems finish">
            <div className="flex items-center gap-3">
              <Timer size={16} className="text-slate-400" />
              <input type="range" min="1" max="3600" step="1" value={budget} onChange={(e) => setBudget(+e.target.value)} className="flex-1" />
              <span className="w-24 text-right text-sm font-bold text-slate-700 dark:text-slate-200">
                {budget >= 60 ? `${Math.floor(budget / 60)} min ${budget % 60}s` : `${budget} s`}
              </span>
            </div>
          </Card>

          <div className="space-y-2">
            {order.map((m) => {
              const n = Math.floor(budget / Math.max(M[m].deploy_s ?? 1, 1e-6));
              const e = M[m].err_extrap ?? 0;
              const usable = e < 10;
              return (
                <div key={m} className="flex items-center gap-3 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-4 py-3">
                  <div className="w-24 text-sm font-bold" style={{ color: COLOR[m] }}>{m}</div>
                  <div className="flex-1 h-6 rounded-full bg-slate-100 dark:bg-slate-700 overflow-hidden">
                    <div className="h-6 rounded-full transition-all duration-500"
                      style={{ width: `${Math.max(n > 0 ? 3 : 0, (n / maxN) * 100)}%`, background: COLOR[m], opacity: usable ? 1 : 0.45 }} />
                  </div>
                  <div className="w-28 text-right text-sm font-bold text-slate-700 dark:text-slate-200">
                    {n === 0 ? "none" : `${n.toLocaleString()} solved`}
                  </div>
                  <div className={`w-24 text-right text-xs font-semibold ${usable ? "text-emerald-600" : "text-rose-600"}`}>
                    {e < 0.01 ? "exact" : `${e.toFixed(0)}% err`}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="rounded-2xl p-4 bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/30 text-sm text-slate-700 dark:text-slate-200">
            Faded bars are solvers whose answers are <b>over 10% wrong beyond the window</b> — throughput you cannot use.
            At small budgets PINN finishes <b>nothing at all</b>.
          </div>
        </>
      )}
    </div>
  );
}
