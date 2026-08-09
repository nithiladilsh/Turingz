import { useEffect, useRef, useState } from "react";
import { Card, Stat } from "../components/ui.jsx";
import { LineChart } from "../components/Charts.jsx";
import { API, WS, STATIC, getMeta, buildIC, pinnIC, realTestIC, pinnRegime, switchingAblation, achievability, costModel, icRepresentativeness } from "../api.js";
import { Trophy, Check, X, Play } from "lucide-react";

const COLOR = { FNO: "#059669", DeepONet: "#e11d48", PINN: "#d97706" };

function thresholds(t) {
  return [Math.min(0.58, Math.max(0.12, 0.62 - 1.4 * t)), null];
}
function useDraw(ms = 1400, key = 0) {
  const [p, setP] = useState(0);
  useEffect(() => {
    let raf, start; setP(0);
    const loop = (ts) => { if (!start) start = ts; const q = Math.min(1, (ts - start) / ms); setP(q); if (q < 1) raf = requestAnimationFrame(loop); };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [ms, key]);
  return p;
}
const pct = (v) => `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`;

/* ---------------- findings ---------------- */
function RegimeCard({ name, best, a, b, c, ok, foot, active, onClick }) {
  return (
    <button onClick={onClick}
      className={`text-left rounded-2xl border p-4 transition-all hover:-translate-y-1 bg-white dark:bg-slate-800 ${active ? "border-2 shadow-md" : "border-slate-200 dark:border-slate-700"}`}
      style={active ? { borderColor: COLOR[name] } : {}}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full" style={{ background: COLOR[name] }} />
          <span className="text-sm font-bold text-slate-800 dark:text-slate-100">{name}</span>
        </div>
        {best
          ? <span className="inline-flex items-center gap-1 text-[10px] font-bold text-emerald-700 dark:text-emerald-300 bg-emerald-100 dark:bg-emerald-500/20 px-2 py-0.5 rounded-full"><Trophy size={11} /> pays off</span>
          : <span className="inline-flex items-center gap-1 text-[10px] font-bold text-rose-700 dark:text-rose-300 bg-rose-100 dark:bg-rose-500/20 px-2 py-0.5 rounded-full"><X size={11} /> {ok}</span>}
      </div>
      <div className="grid grid-cols-3 gap-2 mt-3">
        {[a, b, c].map((s, i) => (
          <div key={i}>
            <div className="text-[10px] text-slate-400 dark:text-slate-500">{s.k}</div>
            <div className="text-lg font-bold" style={i === 1 ? { color: COLOR[name] } : {}}>{s.v}</div>
          </div>
        ))}
      </div>
      {foot && <div className="text-[10px] leading-snug text-slate-400 dark:text-slate-500 mt-2">{foot}</div>}
    </button>
  );
}

function Runway({ name, lo, hi }) {
  const cheap = hi <= 1;
  const scale = (v) => Math.min(100, (v / 3.2) * 100);
  return (
    <div className="flex items-center gap-3">
      <div className="w-20 text-xs font-semibold" style={{ color: COLOR[name] }}>{name}</div>
      <div className="flex-1 h-5 rounded-full bg-slate-100 dark:bg-slate-700 relative overflow-hidden">
        <div className="absolute h-5 rounded-full transition-all duration-700"
          style={{ left: `${scale(lo)}%`, width: `${Math.max(3, scale(hi) - scale(lo))}%`, background: COLOR[name] }} />
        <div className="absolute inset-y-0 w-0.5 bg-indigo-500" style={{ left: `${scale(1)}%` }} />
      </div>
      <div className={`w-28 text-right text-xs font-semibold ${cheap ? "text-emerald-600" : "text-rose-600"}`}>
        {lo.toFixed(2)}–{hi.toFixed(2)}× {cheap ? "cheaper" : "dearer"}
      </div>
    </div>
  );
}

function FrontierCompare({ models, p }) {
  const W = 640, H = 300, padL = 52, padR = 20, padT = 18, padB = 40;
  const pts = Object.keys(models).flatMap((n) => models[n].frontier);
  const xs = pts.map((q) => Math.max(q.rel_cost, 0.05)), ys = pts.map((q) => Math.max(q.error, 1e-3));
  const xmin = Math.min(...xs, 0.9) * 0.7, xmax = Math.max(...xs, 1.1) * 1.3;
  const ymin = Math.min(...ys) * 0.6, ymax = Math.max(...ys) * 1.6;
  const L = Math.log10;
  const sx = (v) => padL + ((L(Math.max(v, 0.05)) - L(xmin)) / (L(xmax) - L(xmin))) * (W - padL - padR);
  const sy = (v) => H - padB - ((L(Math.max(v, 1e-3)) - L(ymin)) / (L(ymax) - L(ymin))) * (H - padT - padB);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {[0.01, 0.03, 0.1, 0.3].filter((g) => g >= ymin && g <= ymax).map((g) => (
        <g key={g}>
          <line x1={padL} x2={W - padR} y1={sy(g)} y2={sy(g)} stroke="var(--chart-grid)" />
          <text x={padL - 6} y={sy(g) + 3} textAnchor="end" fontSize="9" fill="var(--chart-axis)">{Math.round(g * 100)}%</text>
        </g>
      ))}
      {[0.3, 1, 3].filter((g) => g >= xmin && g <= xmax).map((g) => (
        <text key={g} x={sx(g)} y={H - padB + 14} textAnchor="middle" fontSize="9" fill="var(--chart-axis)">{g}×</text>
      ))}
      <rect x={padL} y={padT} width={Math.max(0, sx(1) - padL)} height={H - padT - padB} fill="#059669" fillOpacity="0.05" />
      <line x1={sx(1)} x2={sx(1)} y1={padT} y2={H - padB} stroke="#6366f1" strokeDasharray="4 3" strokeWidth="1.5" />
      <text x={sx(1) - 6} y={padT + 12} textAnchor="end" fontSize="9" fill="#059669" fontWeight="700">cheaper than numerical</text>
      <text x={sx(1) + 6} y={padT + 12} fontSize="9" fill="#e11d48" fontWeight="700">dearer</text>
      {Object.keys(models).map((n) => {
        const fr = [...models[n].frontier].sort((a, b) => a.rel_cost - b.rel_cost);
        const k = Math.max(2, Math.ceil(p * fr.length));
        const d = fr.slice(0, k).map((q, i) => `${i ? "L" : "M"}${sx(q.rel_cost).toFixed(1)} ${sy(q.error).toFixed(1)}`).join(" ");
        return (
          <g key={n}>
            <path d={d} fill="none" stroke={COLOR[n]} strokeWidth="2.5" strokeLinecap="round" />
            {fr.slice(0, k).map((q, i) => <circle key={i} cx={sx(q.rel_cost)} cy={sy(q.error)} r="4" fill={COLOR[n]} />)}
            {k > 1 && <text x={sx(fr[k - 1].rel_cost) + 9} y={sy(fr[k - 1].error) + 4} fontSize="10.5" fontWeight="700" fill={COLOR[n]}>{n}</text>}
          </g>
        );
      })}
      <text x={(padL + W - padR) / 2} y={H - 6} textAnchor="middle" fontSize="9.5" fill="var(--chart-axis)">cost ÷ numerical solver</text>
    </svg>
  );
}

/* ---------------- live ---------------- */
function Tile({ label, value, sub, tone = "slate" }) {
  const c = { slate: "text-slate-800 dark:text-slate-100", green: "text-emerald-600 dark:text-emerald-400", red: "text-rose-600 dark:text-rose-400", indigo: "text-indigo-600 dark:text-indigo-400" }[tone];
  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4">
      <div className="text-[10px] uppercase tracking-wider text-slate-400 dark:text-slate-500">{label}</div>
      <div className={`text-2xl font-extrabold mt-1 ${c}`}>{value}</div>
      {sub && <div className="text-[11px] text-slate-400 dark:text-slate-500 mt-0.5">{sub}</div>}
    </div>
  );
}

function KnobFrontier({ fr, sel, p }) {
  const W = 620, H = 260, padL = 50, padR = 18, padT = 16, padB = 38;
  const xs = fr.map((q) => q.cost), ys = fr.map((q) => q.error);
  const xmin = Math.min(...xs) * 0.8, xmax = Math.max(...xs) * 1.2;
  const ymin = Math.min(...ys) * 0.8, ymax = Math.max(...ys) * 1.2;
  const sx = (v) => padL + ((v - xmin) / (xmax - xmin)) * (W - padL - padR);
  const sy = (v) => H - padB - ((v - ymin) / (ymax - ymin)) * (H - padT - padB);
  const s = [...fr].sort((a, b) => a.cost - b.cost);
  const k = Math.max(2, Math.ceil(p * s.length));
  const d = s.slice(0, k).map((q, i) => `${i ? "L" : "M"}${sx(q.cost).toFixed(1)} ${sy(q.error).toFixed(1)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {[0, 0.5, 1].map((f) => {
        const v = ymin + f * (ymax - ymin);
        return <g key={f}><line x1={padL} x2={W - padR} y1={sy(v)} y2={sy(v)} stroke="var(--chart-grid)" />
          <text x={padL - 6} y={sy(v) + 3} textAnchor="end" fontSize="9" fill="var(--chart-axis)">{(v * 100).toFixed(1)}%</text></g>;
      })}
      <path d={d} fill="none" stroke="#059669" strokeWidth="2.5" strokeLinecap="round" />
      {s.slice(0, k).map((q, i) => <circle key={i} cx={sx(q.cost)} cy={sy(q.error)} r="4" fill="#059669" />)}
      {sel && <circle cx={sx(sel.cost)} cy={sy(sel.error)} r="10" fill="none" stroke="#111827" strokeWidth="2.5" className="transition-all duration-300" />}
      <text x={(padL + W - padR) / 2} y={H - 6} textAnchor="middle" fontSize="9.5" fill="var(--chart-axis)">cost (seconds) →</text>
    </svg>
  );
}



/* live run over the websocket (self-contained: does not touch shared api.js) */
function runCostControl(payload, onFrame, onSummary, onDone, onError) {
  const ws = new WebSocket(`${WS}/ws/costcontrol`);
  ws.onopen = () => ws.send(JSON.stringify(payload));
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (typeof m.error === "string") return onError && onError(m.error);
    if (m.done) { onDone && onDone(); ws.close(); return; }
    if (m.summary) return onSummary && onSummary(m.summary);
    onFrame(m);
  };
  ws.onerror = () => onError && onError("Could not reach backend — start it with: uvicorn main:app");
  return ws;
}

function WaveChart({ x, frame }) {
  const W = 620, H = 240, padL = 34, padR = 14, padT = 14, padB = 26;
  if (!x.length || !frame) return <svg viewBox={`0 0 ${W} ${H}`} className="w-full" />;
  const sx = (v) => padL + ((v + 1) / 2) * (W - padL - padR);
  const sy = (v) => H - padB - ((Math.max(-1.9, Math.min(1.9, v)) + 2) / 4) * (H - padT - padB);
  const path = (arr) => arr.map((v, i) => `${i ? "L" : "M"}${sx(x[i]).toFixed(1)} ${sy(v).toFixed(1)}`).join(" ");
  const col = frame.correcting ? "#e11d48" : "#059669";
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {[-1, 0, 1].map((g) => <line key={g} x1={padL} x2={W - padR} y1={sy(g)} y2={sy(g)} stroke="var(--chart-grid)" />)}
      <path d={path(frame.true)} fill="none" stroke="#94a3b8" strokeWidth="2" strokeDasharray="5 4" />
      <path d={path(frame.u)} fill="none" stroke={col} strokeWidth="2.5" strokeLinejoin="round" />
      <text x={W - padR} y={padT + 4} textAnchor="end" fontSize="10" fontWeight="700" fill={col}>
        {frame.correcting ? "numerical correcting" : "running ML"}
      </text>
    </svg>
  );
}

function TrustTrace({ hist, lo }) {
  const W = 620, H = 150, padL = 34, padR = 14, padT = 12, padB = 24;
  const sx = (t) => padL + (t / 2) * (W - padL - padR);
  const sy = (v) => H - padB - v * (H - padT - padB);
  const d = hist.map((f, i) => `${i ? "L" : "M"}${sx(f.t).toFixed(1)} ${sy(f.trust).toFixed(1)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      <rect x={padL} y={sy(lo)} width={W - padL - padR} height={Math.max(0, H - padB - sy(lo))} fill="#e11d48" fillOpacity="0.07" />
      <line x1={padL} x2={W - padR} y1={sy(lo)} y2={sy(lo)} stroke="#e11d48" strokeDasharray="4 3" />
      {hist.filter((f) => f.correcting).map((f, i) => (
        <rect key={i} x={sx(f.t)} y={padT} width={2.2} height={H - padT - padB} fill="#e11d48" fillOpacity="0.16" />
      ))}
      <path d={d} fill="none" stroke="#4f46e5" strokeWidth="2" />
      <text x={padL + 4} y={sy(lo) + 12} fontSize="8.5" fill="#e11d48" fontWeight="700">below θlo — controller hands over and stays over</text>
      <text x={W - padR} y={H - 6} textAnchor="end" fontSize="9" fill="var(--chart-axis)">time t →</text>
    </svg>
  );
}

function EndLabel({ x, y, text, color, above }) {
  return (
    <text x={x - 5} y={above ? y - 5 : y + 11} textAnchor="end" fontSize="9.5" fontWeight="700" fill={color}>
      {text}
    </text>
  );
}

function ErrorCompareTrace({ hist, target }) {
  const W = 620, H = 150, padL = 34, padR = 14, padT = 12, padB = 24;
  const vals = hist.flatMap((f) => [f.error, f.pure_ml_error_running]).filter((v) => v != null);
  const maxErr = Math.max(target || 0, 0.05, ...vals, 1e-6) * 1.15;
  const sx = (t) => padL + (t / 2) * (W - padL - padR);
  const sy = (v) => H - padB - (Math.min(v, maxErr) / maxErr) * (H - padT - padB);
  const pts = (key) => hist.filter((f) => f[key] != null);
  const line = (key) => pts(key).map((f, i) => `${i ? "L" : "M"}${sx(f.t).toFixed(1)} ${sy(f[key]).toFixed(1)}`).join(" ");
  const lastOf = (key) => { const a = pts(key); return a.length ? a[a.length - 1] : null; };
  const lastErr = lastOf("error"), lastMl = lastOf("pure_ml_error_running");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {target != null && (
        <>
          <line x1={padL} x2={W - padR} y1={sy(target)} y2={sy(target)} stroke="#d97706" strokeDasharray="4 3" />
          <text x={padL + 4} y={sy(target) - 4} fontSize="8.5" fill="#d97706" fontWeight="700">target {target}</text>
        </>
      )}
      <path d={line("pure_ml_error_running")} fill="none" stroke="#e11d48" strokeWidth="1.6" strokeDasharray="3 2" />
      <path d={line("error")} fill="none" stroke="#059669" strokeWidth="2" />
      {lastMl && <EndLabel x={sx(lastMl.t)} y={sy(lastMl.pure_ml_error_running)} text={pct(lastMl.pure_ml_error_running)} color="#e11d48" above />}
      {lastErr && <EndLabel x={sx(lastErr.t)} y={sy(lastErr.error)} text={pct(lastErr.error)} color="#059669" />}
      <text x={W - padR} y={H - 6} textAnchor="end" fontSize="9" fill="var(--chart-axis)">time t →</text>
    </svg>
  );
}

function CostCompareTrace({ hist }) {
  const W = 620, H = 150, padL = 34, padR = 14, padT = 12, padB = 24;
  const vals = hist.flatMap((f) => [f.cost_s, f.cost_ml_only, f.cost_num_only]).filter((v) => v != null);
  const maxCost = Math.max(0.05, ...vals, 1e-6) * 1.1;
  const sx = (t) => padL + (t / 2) * (W - padL - padR);
  const sy = (v) => H - padB - (Math.min(v, maxCost) / maxCost) * (H - padT - padB);
  const pts = (key) => hist.filter((f) => f[key] != null);
  const line = (key) => pts(key).map((f, i) => `${i ? "L" : "M"}${sx(f.t).toFixed(1)} ${sy(f[key]).toFixed(1)}`).join(" ");
  const lastOf = (key) => { const a = pts(key); return a.length ? a[a.length - 1] : null; };
  const lastCost = lastOf("cost_s"), lastMl = lastOf("cost_ml_only"), lastNum = lastOf("cost_num_only");
  const fmt = (v) => `${v.toFixed(2)}s`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      <path d={line("cost_num_only")} fill="none" stroke="#94a3b8" strokeWidth="1.6" strokeDasharray="2 3" />
      <path d={line("cost_ml_only")} fill="none" stroke="#e11d48" strokeWidth="1.6" strokeDasharray="3 2" />
      <path d={line("cost_s")} fill="none" stroke="#059669" strokeWidth="2" />
      {lastNum && <EndLabel x={sx(lastNum.t)} y={sy(lastNum.cost_num_only)} text={fmt(lastNum.cost_num_only)} color="#64748b" above />}
      {lastMl && <EndLabel x={sx(lastMl.t)} y={sy(lastMl.cost_ml_only)} text={fmt(lastMl.cost_ml_only)} color="#e11d48" above />}
      {lastCost && <EndLabel x={sx(lastCost.t)} y={sy(lastCost.cost_s)} text={fmt(lastCost.cost_s)} color="#059669" />}
      <text x={W - padR} y={H - 6} textAnchor="end" fontSize="9" fill="var(--chart-axis)">time t →</text>
    </svg>
  );
}

/* ---------------- how it's built ---------------- */
const BUILD_STEPS = [
  { n: 1, color: "#4f46e5", title: "Read the request",
    what: "Take a requested accuracy target from whoever is asking — e.g. \"give me 3% error.\"",
    chips: [{ label: "accuracy target", color: "#4f46e5" }],
    why: "The whole point of the controller is that this number is chosen by the caller, not hardcoded into the system." },
  { n: 2, color: "#0d9488", title: "Target → trust threshold",
    what: "Map the target into how sensitive the hand-over should be to dropping trust.",
    detail: "θlo = min(0.58, max(0.12, 0.62 − 1.4 × target))",
    why: "Not arbitrary — calibrated to the trust signal's real measured range, then independently checked with a brute-force sweep over the whole threshold range. At the tight targets that matter most, the sweep's own best values land right where this formula already puts them." },
  { n: 3, color: "#7c3aed", title: "Watch trust, decide live",
    what: "Every step, read Module 1's trust score for the actual output so far.",
    chips: [{ label: "trust score", color: "#7c3aed" }, { label: "current step", color: "#7c3aed" }],
    why: "The decision is made from the trajectory as actually produced — including any earlier correction — not from a raw, unaware prediction stream." },
  { n: 4, color: "#e11d48", title: "One-way latch",
    what: "If trust drops below the threshold, start correcting with the numerical solver, and keep correcting every step until the run ends — no separate lookup for how much, no handing back once it's engaged.",
    detail: "if trust < θlo: correct, and never release",
    why: "Tested against a two-way version that can revert: on the real trust signal, one-way matches or beats it at every target — simpler, and nothing measured is lost by dropping the extra rule." },
  { n: 5, color: "#059669", title: "Add up the real cost",
    what: "Total cost is the ML steps taken plus the numerical steps taken, each at its own measured cost.",
    detail: "cost = ml_steps × ml_step_s + correction_steps × correction_step_s",
    why: "A simple additive model lets cost be predicted from step counts alone, and it's checked against real measured wall-clock time to confirm it actually holds." },
];

function HowBuilt() {
  return (
    <div className="space-y-6">
      <div className="rounded-2xl p-6 md:p-7 bg-gradient-to-r from-indigo-50 via-indigo-50 to-violet-50 dark:from-indigo-500/10 dark:via-indigo-500/10 dark:to-violet-500/10 border border-indigo-100 dark:border-indigo-500/25">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">Accuracy-budget knob · one requested target, one spending decision</div>
        <h2 className="text-2xl font-bold mt-1 text-slate-800 dark:text-slate-100">How the controller is built</h2>
        <p className="text-sm text-slate-600 dark:text-slate-300 mt-2 leading-relaxed">
          A requested accuracy target is turned into one thing: how sensitive the hand-over should be to falling trust. That mapping is
          fixed once, from measurements taken offline, then simply applied live — once it triggers, it's a one-way switch, not a
          per-step negotiation.
        </p>
      </div>

      <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-5 py-4">
        <div className="flex flex-wrap items-center gap-y-2">
          {BUILD_STEPS.map((s, i) => (
            <div key={s.n} className="flex items-center">
              <div className="flex items-center gap-2">
                <span className="w-6 h-6 rounded-lg text-[11px] font-bold flex items-center justify-center shrink-0" style={{ background: s.color + "1A", color: s.color }}>{s.n}</span>
                <span className="text-xs font-medium text-slate-600 dark:text-slate-300 whitespace-nowrap">{s.title}</span>
              </div>
              {i < BUILD_STEPS.length - 1 && <span className="mx-2.5 text-slate-300 dark:text-slate-600 text-xs">→</span>}
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {BUILD_STEPS.map((s) => (
          <div key={s.n}
            className="group rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5 shadow-sm hover:shadow-md hover:border-slate-300 dark:hover:border-slate-600 transition flex flex-col">
            <div className="flex items-center gap-3">
              <span className="w-9 h-9 rounded-xl text-sm font-bold flex items-center justify-center shrink-0" style={{ background: s.color + "1A", color: s.color }}>{s.n}</span>
              <h3 className="text-sm font-bold text-slate-800 dark:text-slate-100 leading-tight">{s.title}</h3>
            </div>
            <p className="text-sm text-slate-600 dark:text-slate-300 mt-3">{s.what}</p>
            {s.chips ? (
              <div className="mt-3 flex flex-wrap gap-2">
                {s.chips.map((cp) => (
                  <span key={cp.label} className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-semibold"
                    style={{ color: cp.color, background: cp.color + "14", border: `1px solid ${cp.color}33` }}>
                    <span className="w-1.5 h-1.5 rounded-full" style={{ background: cp.color }} />
                    {cp.label}
                  </span>
                ))}
              </div>
            ) : (
              <div className="mt-3 rounded-xl px-3 py-2.5 font-mono text-[12.5px] text-slate-700 dark:text-slate-100 text-center overflow-x-auto"
                style={{ background: s.color + "0D", border: `1px solid ${s.color}26` }}>{s.detail}</div>
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
          <div className="text-sm font-bold text-emerald-800 dark:text-emerald-300">A knob, not extra accuracy at one point</div>
          <p className="text-sm text-slate-600 dark:text-slate-300 mt-1 leading-relaxed">
            At any single target, the latch and a well-chosen fixed threshold land close together in raw error — that's expected, at one
            operating point the controller <i>is</i> a threshold. What the latch adds is that the requested target is actually reachable
            without hand-tuning: it tracks the target across the whole range and wins clearly on hit-rate, not on one point in isolation.
          </p>
        </div>
        <div className="rounded-2xl border border-teal-200 dark:border-teal-500/30 bg-teal-50 dark:bg-teal-500/10 p-5">
          <div className="text-sm font-bold text-teal-800 dark:text-teal-300">One-way, by evidence — not by convenience</div>
          <p className="text-sm text-slate-600 dark:text-slate-300 mt-1 leading-relaxed">
            A two-way version that can hand back control was built and tested first, and did help on an early, noisier signal. On the real
            trust signal it stopped adding anything measurable, so the simpler one-way rule was kept — a simplification the data justified,
            not one that was assumed from the start.
          </p>
        </div>
      </div>
    </div>
  );
}

/* ---------------- results ---------------- */
function PlotCard({ title, subtitle, src }) {
  return (
    <Card title={title} subtitle={subtitle}>
      <img src={src} alt={title} className="w-full rounded-xl border border-slate-100 dark:border-slate-700" loading="lazy" />
    </Card>
  );
}

function ResultsTab({ ach, cm, icRep, cmp }) {
  const achRows = ach?.rows || [];
  const F = cmp?.FNO;
  const headline = F?.frontier?.find((q) => Math.abs(q.target - 0.1) < 1e-9);

  return (
    <div className="space-y-6">
      {headline && F.pure_numerical && (
        <Card title="What the hybrid actually delivers" subtitle="real wall-clock time, target 0.1 · numerical-only cost as baseline · averaged across the held-out test set, not a single run">
          <div className="grid grid-cols-3 gap-3">
            <Stat label="hybrid" value={`${headline.cost.toFixed(2)} s`} tone="green" />
            <Stat label="numerical-only" value={`${F.pure_numerical.cost.toFixed(2)} s`} tone="slate" />
            <Stat label="savings" value={pct(1 - headline.cost / F.pure_numerical.cost)} tone="indigo" />
          </div>
        </Card>
      )}

      <Card title="The full cost-accuracy frontier" subtitle="every target swept, FNO hybrid vs. both pure baselines — real timed runs">
        <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-6 items-center">
          <PlotCard title="Cheaper than numerical, more accurate than the surrogate, across the whole sweep" src={`${STATIC}/step9d_coarse_integration/fig_6_5_1_frontier.png`} />

          {F?.frontier?.length > 0 && F.pure_ml && F.pure_numerical && (
            <div>
              <div className="text-xs font-semibold text-slate-600 dark:text-slate-300 mb-2">each point, against both baselines</div>
              <div className="overflow-x-auto -mx-1">
                <table className="w-full text-xs border-collapse">
                  <thead>
                    <tr className="text-left text-[9.5px] uppercase tracking-wider text-slate-400 dark:text-slate-500">
                      <th className="py-1.5 px-1 font-semibold">target</th>
                      <th className="py-1.5 px-1 font-semibold">cost / err</th>
                      <th className="py-1.5 px-1 font-semibold">vs. ML</th>
                      <th className="py-1.5 px-1 font-semibold">vs. numerical</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...F.frontier].sort((a, b) => b.target - a.target).map((r) => (
                      <tr key={r.target} className="border-t border-slate-100 dark:border-slate-700">
                        <td className="py-2 px-1 font-semibold text-slate-700 dark:text-slate-200">{r.target}</td>
                        <td className="py-2 px-1 text-slate-500 dark:text-slate-400 whitespace-nowrap">{r.cost.toFixed(2)}s · {pct(r.error)}</td>
                        <td className="py-2 px-1 font-semibold text-rose-600 dark:text-rose-400 whitespace-nowrap">{(F.pure_ml.error / r.error).toFixed(1)}×</td>
                        <td className="py-2 px-1 font-semibold text-indigo-600 dark:text-indigo-400 whitespace-nowrap">{(F.pure_numerical.cost / r.cost).toFixed(1)}×</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-[10.5px] text-slate-400 dark:text-slate-500 mt-2">
                "vs. ML" = pure-ML error ÷ hybrid error (more accurate). "vs. numerical" = numerical-only cost ÷ hybrid cost (cheaper).
              </p>
            </div>
          )}
        </div>
        <p className="text-[11px] text-slate-400 dark:text-slate-500 mt-4">
          the hybrid (green) never leaves the shaded region — cheaper than running the numerical solver alone, and more accurate than
          running the ML surrogate alone — across every target from loose to tight; error bars are real run-to-run spread, not estimated.
        </p>
      </Card>

      {cm && cm.pearson_r != null && (
        <Card title="Cost model checks out against real wall-clock time" subtitle={`n = ${cm.n_points ?? "60"} timed runs, predicted vs measured`}>
          <div className="grid grid-cols-3 gap-3">
            <Stat label="Pearson r" value={cm.pearson_r.toFixed(3)} tone="green" />
            <Stat label="MAPE" value={pct(cm.mape)} tone="indigo" />
            <Stat label="slope (measured / predicted)" value={cm.slope_measured_vs_predicted.toFixed(3)} tone="slate" />
          </div>
        </Card>
      )}

      {achRows.length > 0 && (
        <Card title="Accuracy target vs. what's actually achieved" subtitle="held-out test conditions — shown honestly, floor included">
          <div className="space-y-2">
            {achRows.map((r) => (
              <div key={r.target} className="flex items-center gap-3 text-sm">
                <div className="w-20 text-xs font-semibold text-slate-600 dark:text-slate-300">target {r.target}</div>
                <div className="flex-1 h-4 rounded-full bg-slate-100 dark:bg-slate-700 relative overflow-hidden">
                  <div className="absolute h-4 rounded-full transition-all duration-700"
                    style={{ width: `${Math.max(3, r.hit_rate * 100)}%`, background: r.hit_rate >= 0.9 ? "#059669" : r.hit_rate >= 0.5 ? "#d97706" : "#e11d48" }} />
                </div>
                <div className="w-24 text-right text-xs font-bold text-slate-600 dark:text-slate-300">{pct(r.hit_rate)} hit</div>
                <div className="w-20 text-right text-xs text-slate-400 dark:text-slate-500">{pct(r.mean_error)} err</div>
              </div>
            ))}
          </div>
          <p className="text-[11px] text-slate-400 dark:text-slate-500 mt-3">
            hit-rate stays at 100% down to target 0.05, then drops once the target passes a real ~3% accuracy floor — the controller
            can't correct its way past what the surrogate and coarse monitor are able to resolve.
          </p>
        </Card>
      )}

      <Card title="Why the switch thresholds sit where they do" subtitle="ceiling (0.58), floor (0.20), saturation (0.029) — each checked against real held-out data">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <PlotCard title="Upper clamp (0.58)" src={`${STATIC}/threshold_calibration/ceiling_derivation.png`} />
          <PlotCard title="Loosest-target threshold (0.20)" src={`${STATIC}/threshold_calibration/floor_derivation.png`} />
          <PlotCard title="Accuracy-saturation reference (0.029)" src={`${STATIC}/threshold_calibration/saturation_point.png`} />
          <PlotCard title="Trust signal's observed operating range" src={`${STATIC}/threshold_calibration/trust_range.png`} />
        </div>
      </Card>

      <Card title="Why the latch policy" subtitle="four switching designs, same held-out test conditions, every target">
        <PlotCard title="Achieved error and hit-rate by switching policy" src={`${STATIC}/step11_switching_ablation/fig_6_5_2_ablation.png`} />
        <p className="text-[11px] text-slate-400 dark:text-slate-500 mt-3">
          fixed interval (grey) ignores the trust signal entirely and pays for it in error; the latch (green) is the only policy that
          keeps tracking the requested target down to the tightest ones, not just matching the alternatives at one operating point.
        </p>
      </Card>

      {icRep?.representativeness && icRep?.bias_check && (
        <Card title="Is the held-out test set representative?" subtitle="checked, not assumed">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <PlotCard title="Held-out test conditions vs. the full dataset" src={`${STATIC}/ic_representativeness/ic_representativeness.png`} />
            <PlotCard title="Does the skew correlate with error?" src={`${STATIC}/ic_representativeness/ic_bias_check.png`} />
          </div>
        </Card>
      )}
    </div>
  );
}

/* ---------------- page ---------------- */
export default function CostControl() {
  const [tab, setTab] = useState("findings");
  const [cmp, setCmp] = useState(null);
  const [rob, setRob] = useState(null);
  const [regime, setRegime] = useState(null);
  const [pinnR, setPinnR] = useState(null);
  const [swAbl, setSwAbl] = useState(null);
  const [ach, setAch] = useState(null);
  const [cm, setCm] = useState(null);
  const [icRep, setIcRep] = useState(null);
  const [pick, setPick] = useState("FNO");
  // fr (built below from /api/m3/frontiers) is [0.30, 0.20, 0.10, 0.05, 0.02, 0.01] in
  // that order -- index 1 is target 0.20. Kept in sync dynamically below in case the
  // underlying frontier file's target list is ever reordered or regenerated.
  const [idx, setIdx] = useState(1);
  const [meta, setMeta] = useState(null);
  const [modes, setModes] = useState(2);
  const [amp, setAmp] = useState(1.0);
  const [model, setModel] = useState("FNO");
  const [pidx, setPidx] = useState(0);
  const [source, setSource] = useState("real"); // "synthetic" | "real" -- default to the validated held-out set
  const [ridx, setRidx] = useState(4); // Test IC #904 -- its pure-ML error (~14%) matches
  // FNO's published mean extrapolation error (14.13%, Cost Analysis), so this is the
  // representative default, not the cherry-picked-easiest one (IC #900 was ~5.7%, an
  // unusually good case, not typical).
  const [ic, setIc] = useState(null);
  const [frame, setFrame] = useState(null);
  const [hist, setHist] = useState([]);
  const [sum, setSum] = useState(null);
  const [running, setRunning] = useState(false);
  const wsRef = useRef(null);
  const [err, setErr] = useState(null);
  const p = useDraw(1400, tab);
  const idxTouched = useRef(false);

  useEffect(() => {
    fetch(`${API}/api/m3/frontiers`).then((r) => r.json()).then(setCmp)
      .catch(() => setErr("Backend not reachable — start it with: uvicorn main:app"));
    fetch(`${API}/api/m3/robustness`).then((r) => r.json()).then(setRob).catch(() => {});
    fetch(`${API}/api/m3/regime`).then((r) => r.json()).then(setRegime).catch(() => {});
    pinnRegime().then(setPinnR).catch(() => {});
    switchingAblation().then(setSwAbl).catch(() => {});
    achievability().then(setAch).catch(() => {});
    costModel().then(setCm).catch(() => {});
    icRepresentativeness().then(setIcRep).catch(() => {});
    getMeta().then(setMeta).catch(() => {});
  }, []);

  // default the accuracy target to 0.20 once the real frontier loads, by target
  // value rather than a hardcoded index -- so this stays correct even if the
  // underlying frontier file's target list is ever reordered or regenerated.
  // Only runs before the user has touched the slider themselves.
  useEffect(() => {
    if (idxTouched.current) return;
    const arr = cmp?.FNO?.frontier;
    if (!arr || !arr.length) return;
    const i = arr.findIndex((q) => Math.abs(q.target - 0.2) < 1e-9);
    if (i >= 0) setIdx(i);
  }, [cmp]);

  useEffect(() => {
    if (!meta) return;
    if (model === "PINN") pinnIC(pidx).then((d) => setIc(d.ic)).catch(() => {});
    else if (source === "real") realTestIC(ridx).then((d) => setIc(d.ic)).catch(() => {});
    else buildIC(modes, amp).then((d) => setIc(d.ic)).catch(() => {});
  }, [meta, model, modes, amp, pidx, source, ridx]);

  function run() {
    if (wsRef.current) wsRef.current.close();
    setHist([]); setFrame(null); setSum(null); setRunning(true);
    wsRef.current = runCostControl(
      model === "PINN"
        ? { model, pinn_index: pidx, target: sel?.target ?? 0.05 }
        : source === "real"
        ? { model, real_ic_index: ridx, target: sel?.target ?? 0.05 }
        : { model, ic, target: sel?.target ?? 0.05 },
      (f) => { setFrame(f); setHist((h) => [...h, f]); },
      (s2) => setSum(s2),
      () => setRunning(false),
      (e) => { setErr(e); setRunning(false); });
  }

  const F = cmp?.FNO, D = cmp?.DeepONet;
  const rng = (m, f) => (m ? [Math.min(...m.frontier.map(f)), Math.max(...m.frontier.map(f))] : [0, 0]);
  const [fLo, fHi] = rng(F, (q) => q.rel_cost), [fE1, fE2] = rng(F, (q) => q.error);
  const [dLo, dHi] = rng(D, (q) => q.rel_cost), [dE1, dE2] = rng(D, (q) => q.error);
  // hit-rate varies a lot by target (e.g. FNO is 100% down to target 0.05, then falls
  // to 10% at 0.02/0.01; DeepONet is 80%/60% at loose targets, 0% only once targets
  // get tight) -- show the real range instead of one flat, misleading number.
  const [fH1, fH2] = rng(F, (q) => q.hit_rate);
  const [dH1, dH2] = rng(D, (q) => q.hit_rate);
  const pRows = pinnR?.rows || [];
  const rngArr = (rows, f) => (rows.length ? [Math.min(...rows.map(f)), Math.max(...rows.map(f))] : [0, 0]);
  const [pE1, pE2] = rngArr(pRows, (r) => r.error);
  const [pH1, pH2] = rngArr(pRows, (r) => r.hit_rate);
  const hitStr = (lo, hi) => (lo === hi ? pct(lo) : `${Math.round(lo * 100)}–${Math.round(hi * 100)}%`);
  const errStr = (lo, hi) => (lo === hi ? pct(lo) : `${pct(lo)}–${pct(hi)}`);
  const pinnX = regime?.surrogates?.PINN && regime?.numerical?.ColeHopf
    ? Math.round(regime.surrogates.PINN.deploy_s / regime.numerical.ColeHopf.deploy_s) : 950;
  const fr = F?.frontier || [];
  const sel = fr[Math.min(idx, fr.length - 1)];
  const [lo] = thresholds(sel?.target ?? 0.1);
  const corr = F && sel ? Math.max(0, Math.min(1, (sel.rel_cost - F.pure_ml.rel_cost) / (1 - F.pure_ml.rel_cost))) : 0;
  const btn = (on) => `px-4 py-1.5 rounded-lg text-sm font-semibold transition ${on ? "bg-indigo-600 text-white" : "text-slate-600 dark:text-slate-300"}`;

  return (
    <div className="space-y-6">
      <div>
        <span className="text-xs font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">Hybrid components</span>
        <h1 className="text-3xl font-extrabold text-slate-800 dark:text-slate-100 mt-1">Cost Control</h1>
        <p className="text-slate-500 dark:text-slate-400 mt-1 max-w-2xl">
          One accuracy knob decides how much numerical help to buy. Below: what it delivers, and where it stops working.
        </p>
      </div>

      <div className="inline-flex rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-1">
        <button className={btn(tab === "findings")} onClick={() => setTab("findings")}>Findings</button>
        <button className={btn(tab === "live")} onClick={() => setTab("live")}>Try it live</button>
        <button className={btn(tab === "built")} onClick={() => setTab("built")}>How it's built</button>
        <button className={btn(tab === "results")} onClick={() => setTab("results")}>Evaluation</button>
      </div>

      {err && <div className="text-sm text-rose-600 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 rounded-lg px-3 py-2">{err}</div>}

      {tab === "findings" && cmp && F && D && (
        <>
          <div className="grid grid-cols-3 gap-4">
            <RegimeCard name="FNO" best active={pick === "FNO"} onClick={() => setPick("FNO")}
              a={{ k: "cost", v: `${fLo.toFixed(2)}×` }} b={{ k: "error", v: pct(fE1) }} c={{ k: "hit-rate", v: hitStr(fH1, fH2) }} />
            <RegimeCard name="DeepONet" ok="dominated" active={pick === "DeepONet"} onClick={() => setPick("DeepONet")}
              a={{ k: "cost", v: `${dLo.toFixed(2)}×` }} b={{ k: "error", v: pct(dE1) }} c={{ k: "hit-rate", v: hitStr(dH1, dH2) }} />
            <RegimeCard name="PINN" ok="not amortized" active={pick === "PINN"} onClick={() => setPick("PINN")}
              a={{ k: "cost", v: `${pinnX}×` }} b={{ k: "error", v: pct(pE1) }} c={{ k: "hit-rate", v: hitStr(pH1, pH2) }}
              foot="error and hit-rate given a free pre-trained model — the controller works, the 2114 s retrain per problem is what rules it out" />
          </div>

          <div className="grid grid-cols-[1fr_360px] gap-5">
            <Card title="Below the line, the hybrid is worth it" subtitle="cost ÷ numerical solver · both frontiers measured">
              <FrontierCompare models={{ FNO: F, DeepONet: D }} p={p} />
            </Card>

            <div className="space-y-4">
              <Card title="Cheaper than the numerical solver?">
                <div className="space-y-3">
                  <Runway name="FNO" lo={fLo} hi={fHi} />
                  <Runway name="DeepONet" lo={dLo} hi={dHi} />
                </div>
                <div className="text-[11px] text-slate-400 dark:text-slate-500 mt-3">the indigo line is the numerical solver (1×)</div>
              </Card>

              {rob?.adaptive_err != null && (
                <Card title="Timing beats brute force" subtitle="same number of corrections">
                  {[["adaptive", rob.adaptive_err, true], ["fixed every-N", rob.fixed_err, false]].map(([l, v, good]) => (
                    <div key={l} className="flex items-center gap-2 mt-2">
                      <div className="w-24 text-xs text-slate-500 dark:text-slate-400">{l}</div>
                      <div className="flex-1 h-4 rounded-full bg-slate-100 dark:bg-slate-700">
                        <div className="h-4 rounded-full transition-all duration-700"
                          style={{ width: `${Math.max(2, (v / Math.max(rob.fixed_err, 1e-6)) * 100)}%`, background: good ? "#059669" : "#e11d48" }} />
                      </div>
                      <div className={`w-14 text-right text-xs font-bold ${good ? "text-emerald-600" : "text-rose-600"}`}>{pct(v)}</div>
                    </div>
                  ))}
                </Card>
              )}
            </div>
          </div>

          {swAbl?.target_response && (
            <Card title="Why the latch policy" subtitle="four switching designs on the same 10 held-out ICs, every target — alternative smart triggers, not a brute-force baseline">
              <div className="space-y-2.5">
                {[["latch", "latch (ours)"], ["deadband", "deadband"], ["naive", "naive"], ["hardcoded", "hardcoded"]].map(([k, label]) => {
                  const tr = swAbl.target_response[k];
                  const hr = swAbl.hit_range[k];
                  if (!tr || !hr) return null;
                  const responds = tr.responds_to_target;
                  return (
                    <div key={k} className="flex flex-wrap items-center gap-2 text-sm">
                      <div className={`w-28 font-semibold ${k === "latch" ? "text-indigo-600 dark:text-indigo-400" : "text-slate-600 dark:text-slate-300"}`}>{label}</div>
                      <div className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${responds
                        ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-300"
                        : "bg-rose-100 text-rose-700 dark:bg-rose-500/20 dark:text-rose-300"}`}>
                        {responds ? "responds to target" : "ignores target"}
                      </div>
                      <div className="text-xs text-slate-500 dark:text-slate-400 sm:ml-auto">
                        error {errStr(tr.min, tr.max)} · hit-rate {hitStr(hr[0], hr[1])}
                      </div>
                    </div>
                  );
                })}
              </div>
              <p className="text-[11px] text-slate-400 dark:text-slate-500 mt-3">
                hardcoded uses a fixed threshold no matter what target you ask for, so its error and hit-rate never move —
                it isn't actually a knob. Latch isn't always the lowest error (hardcoded gets lucky at loose targets),
                but it's the only policy that reliably tracks the requested target across the whole range.
              </p>
            </Card>
          )}

          <div className="rounded-2xl p-5 bg-gradient-to-r from-emerald-50 via-white to-white dark:from-emerald-500/10 dark:via-slate-800 dark:to-slate-800 border border-emerald-200 dark:border-emerald-500/30">
            <div className="text-sm font-semibold text-emerald-800 dark:text-emerald-300 flex items-center gap-2"><Check size={16} /> Conclusion</div>
            <p className="text-sm text-slate-700 dark:text-slate-200 mt-1">
              With FNO the hybrid runs at <b>{fLo.toFixed(2)}–{fHi.toFixed(2)}× the numerical cost</b> for <b>{pct(fE1)}–{pct(fE2)} error</b>.
              With a weak surrogate it is <b>strictly worse than doing nothing</b> — DeepONet costs {dLo.toFixed(2)}–{dHi.toFixed(2)}× and stays at {pct(dE1)}.
              The controller <b>protects a good surrogate; it cannot rescue a bad one.</b>
            </p>
          </div>
        </>
      )}

      {tab === "results" && <ResultsTab ach={ach} cm={cm} icRep={icRep} cmp={cmp} />}

      {tab === "live" && (
        <>
          <div className="grid grid-cols-4 gap-4">
            <Tile label="time" value={frame ? `t = ${frame.t.toFixed(2)}` : "—"} sub={running ? "running…" : "press run"} />
            <Tile label="trust" value={frame ? frame.trust.toFixed(2) : "—"}
              tone={frame ? (frame.correcting ? "red" : "green") : "slate"}
              sub={frame ? (frame.correcting ? "correcting" : "trusting ML") : "—"} />
            <Tile label="cost so far" value={frame ? `${frame.cost_s.toFixed(2)} s` : "—"}
              tone={model === "PINN" ? "red" : "green"}
              sub={frame
                ? `${frame.ml_steps} ML · ${frame.corr_steps} numerical${model === "PINN" ? " · excludes 2114 s retrain" : ""}`
                : "—"} />
            <Tile label="error now" value={frame ? pct(frame.error) : "—"}
              tone={sum ? (sum.hit ? "green" : "red") : "slate"} sub={`target ${sel ? sel.target : "—"}`} />
          </div>

          <div className="grid grid-cols-[320px_1fr] gap-5">
            <div className="space-y-4">
              <Card title="1 · Choose a surrogate">
                <div className="flex gap-2">
                  {["FNO", "DeepONet", "PINN"].map((m) => (
                    <button key={m} onClick={() => setModel(m)}
                      className={`flex-1 px-2 py-1.5 rounded-lg text-xs font-bold border transition ${model === m ? "text-white border-transparent" : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700"}`}
                      style={model === m ? { background: COLOR[m] } : {}}>{m}</button>
                  ))}
                </div>
                {model !== "FNO" && (
                  <div className="text-[11px] mt-2 px-2 py-1.5 rounded-lg bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-300">
                    {model === "DeepONet"
                      ? "outside the regime — ~29% wrong in-window, so expect it to bail out at once"
                      : "outside the regime — cost below excludes PINN's 2114 s retrain per problem"}
                  </div>
                )}
              </Card>

              <Card title="2 · Set the input">
                {model === "PINN" ? (
                  <select value={pidx} onChange={(e) => setPidx(+e.target.value)}
                    className="w-full bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-2 text-sm">
                    {Array.from({ length: meta?.n_pinn_ics || 0 }, (_, i) => <option key={i} value={i}>Trained wave #{i}</option>)}
                  </select>
                ) : (
                  <>
                    <div className="flex gap-2 mb-3">
                      {[["real", "Held-out test IC"], ["synthetic", "Random shape"]].map(([v, label]) => (
                        <button key={v} onClick={() => setSource(v)}
                          className={`flex-1 px-2 py-1.5 rounded-lg text-xs font-bold border transition ${source === v ? "bg-indigo-600 text-white border-transparent" : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700"}`}>
                          {label}
                        </button>
                      ))}
                    </div>
                    {source === "real" ? (
                      <>
                        <select value={ridx} onChange={(e) => setRidx(+e.target.value)}
                          className="w-full bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-2 text-sm">
                          {Array.from({ length: meta?.n_real_test_ics || 10 }, (_, i) => <option key={i} value={i}>Test IC #{900 + i}</option>)}
                        </select>
                        <div className="text-[11px] text-slate-400 dark:text-slate-500 mt-2">
                          one of the 10 problems the reported hit-rate numbers were measured on — not a random draw
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="flex justify-between text-[11px] text-slate-500 dark:text-slate-400"><span>modes</span><span>{modes}</span></div>
                        <input type="range" min="1" max="4" value={modes} onChange={(e) => setModes(+e.target.value)} className="w-full" />
                        <div className="flex justify-between text-[11px] text-slate-500 dark:text-slate-400 mt-2"><span>amplitude</span><span>{amp.toFixed(1)}</span></div>
                        <input type="range" min="0.5" max="1.5" step="0.1" value={amp} onChange={(e) => setAmp(+e.target.value)} className="w-full" />
                        {modes > 4 && <div className="text-[11px] text-amber-600 mt-1">above 4 = out-of-distribution</div>}
                        <div className="text-[11px] text-slate-400 dark:text-slate-500 mt-2">
                          a fresh random shape — may land off the distribution the hit-rate was measured on
                        </div>
                      </>
                    )}
                  </>
                )}
                {ic && (
                  <div className="mt-3">
                    <div className="text-[11px] mb-1 text-slate-400 dark:text-slate-500">starting wave preview</div>
                    <LineChart series={[{ x: meta?.x || [], y: ic, color: "#6366f1", width: 2 }]}
                      xr={[-1, 1]} yr={[-1.6, 1.6]} h={130} xlabel="x" />
                  </div>
                )}
              </Card>

              <Card title="3 · Accuracy target (Module 3)"
                subtitle="the accuracy you ask for — the controller turns it into when to correct">
                <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400"><span>loose 0.30</span><span>tight 0.01</span></div>
                <input type="range" min="0" max={Math.max(0, fr.length - 1)} value={idx}
                  onChange={(e) => { idxTouched.current = true; setIdx(+e.target.value); }} className="w-full" />
                <div className="text-xs mt-1 text-slate-400 dark:text-slate-500">
                  target {sel ? sel.target : "—"} → θlo {lo.toFixed(2)} (one-way handover)
                </div>
              </Card>

              <button onClick={run} disabled={running || !ic}
                className="w-full px-4 py-2.5 rounded-xl bg-emerald-600 text-white text-sm font-bold hover:bg-emerald-700 disabled:opacity-50 inline-flex items-center justify-center gap-2">
                <Play size={16} /> {running ? "Running…" : "4 · Run the controller"}
              </button>
            </div>

            <div className="space-y-4">
              <Card title="Solution" subtitle="green = running ML · red = numerical correction · grey dashed = truth">
                <WaveChart x={meta?.x || []} frame={frame} />
              </Card>
              <Card title="Trust vs the deadband" subtitle="red bands = steps where it paid for numerical · tracks whichever signal is actually running, ML then the corrected output">
                <TrustTrace hist={hist} lo={lo} />
              </Card>
              <Card title="Error so far: corrected vs. pure ML" subtitle="green = the actual (corrected) trajectory · red dashed = what pure ML alone would show, same problem">
                <ErrorCompareTrace hist={hist} target={sel?.target} />
              </Card>
              <Card title="Cost so far: hybrid vs. running it alone" subtitle="green = actual spend · red dashed = pure ML the whole way · grey dotted = pure numerical the whole way">
                <CostCompareTrace hist={hist} />
              </Card>
            </div>
          </div>

          {sum && (
            <div className={`rounded-2xl p-5 border ${sum.hit
              ? "bg-emerald-50 dark:bg-emerald-500/10 border-emerald-200 dark:border-emerald-500/30"
              : "bg-amber-50 dark:bg-amber-500/10 border-amber-200 dark:border-amber-500/30"}`}>
              <div className="flex items-center gap-2 text-sm font-bold text-slate-800 dark:text-slate-100">
                {sum.hit ? <Check size={16} /> : <X size={16} />}
                {sum.hit ? `Target ${sum.target} met` : `Target ${sum.target} missed — the ${pct(sum.error)} floor is the monitor, not the knob`}
              </div>
              <div className="flex h-7 rounded-lg overflow-hidden text-[11px] font-bold mt-3">
                {(() => {
                  const mlPct = Math.round((sum.ml_steps / (sum.ml_steps + sum.corr_steps)) * 100);
                  const numPct = 100 - mlPct;
                  return (
                    <>
                      {mlPct > 0 && (
                        <div className="text-white flex items-center justify-center whitespace-nowrap px-1.5"
                          style={{ flex: `${mlPct} 1 0%`, minWidth: "38px", background: "#059669" }}>
                          ML {mlPct}%
                        </div>
                      )}
                      {numPct > 0 && (
                        <div className="text-white flex items-center justify-center whitespace-nowrap px-1.5"
                          style={{ flex: `${numPct} 1 0%`, minWidth: "44px", background: "#e11d48" }}>
                          num {numPct}%
                        </div>
                      )}
                    </>
                  );
                })()}
              </div>
              <div className="text-sm text-slate-700 dark:text-slate-200 mt-3">
                <b>{sum.cost_s.toFixed(2)} s</b> ({sum.rel_cost.toFixed(2)}× the numerical solver) at <b>{pct(sum.error)}</b> error
                {sum.switch_t != null && <> · first correction at t = {sum.switch_t.toFixed(2)}</>}
              </div>
              <div className="text-xs text-slate-500 dark:text-slate-400 mt-1.5">
                {sum.rel_cost > 0 && sum.rel_cost < 1
                  ? <>that's <b className="text-emerald-600 dark:text-emerald-400">{(1 / sum.rel_cost).toFixed(2)}×</b> faster than running the numerical solver alone on this problem</>
                  : <>that's <b className="text-rose-600 dark:text-rose-400">{sum.rel_cost.toFixed(2)}×</b> the numerical solver's time — slower here, not cheaper</>}
              </div>
              {sum.pure_ml_error != null && (
                <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                  vs <b className="text-rose-600 dark:text-rose-400">{pct(sum.pure_ml_error)}</b> if the model had run the whole thing alone, uncorrected — on this exact problem
                </div>
              )}
            </div>
          )}
        </>
      )}

      {tab === "built" && <HowBuilt />}
    </div>
  );
}
