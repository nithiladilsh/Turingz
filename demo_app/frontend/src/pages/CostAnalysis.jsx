import { useEffect, useState } from "react";
import { Card, Stat, Banner } from "../components/ui.jsx";
import { API } from "../api.js";

const COLORS = { FNO: "#059669", DeepONet: "#e11d48", PINN: "#d97706", FDM: "#64748b", Spectral: "#4f46e5", ColeHopf: "#7c3aed" };
const ML = ["FNO", "DeepONet", "PINN"];

/* clean decade ticks - no floating-point artefacts, never more than 6 per axis */
function decadeTicks(min, max, maxCount = 6) {
  const lo = Math.ceil(Math.log10(min) - 1e-9);
  const hi = Math.floor(Math.log10(max) + 1e-9);
  let exps = [];
  for (let e = lo; e <= hi; e++) exps.push(e);
  if (!exps.length) return [];
  const step = Math.ceil(exps.length / maxCount) || 1;
  return exps.filter((_, i) => i % step === 0).map((e) => Math.pow(10, e));
}
const fmtSec = (v) => (v >= 1000 ? `${Math.round(v / 1000)}k s` : v >= 1 ? `${v} s` : `${v} s`);
const fmtPct = (v) => (v >= 0.01 ? `${v}%` : `${v.toExponential(0)}%`);

function Scatter({ models, mode }) {
  const w = 640, h = 360, padL = 62, padR = 26, padT = 22, padB = 46;
  const names = Object.keys(models);
  const val = (n, which) => Math.max((which === "in" ? models[n].err_in : models[n].err_extrap) ?? 1e-4, 1e-4);
  const pts = names.map((n) => ({
    n, kind: models[n].kind,
    x: Math.max(models[n].deploy_s ?? 0.01, 0.01),
    yin: val(n, "in"), yex: val(n, "extrap"),
  }));
  const allY = pts.flatMap((p) => [p.yin, p.yex]);
  const xmin = Math.min(...pts.map((p) => p.x)) * 0.45, xmax = Math.max(...pts.map((p) => p.x)) * 3;
  const ymin = Math.min(...allY) * 0.35, ymax = Math.max(...allY) * 3.5;
  const L = Math.log10;
  const sx = (v) => padL + ((L(v) - L(xmin)) / (L(xmax) - L(xmin))) * (w - padL - padR);
  const sy = (v) => h - padB - ((L(v) - L(ymin)) / (L(ymax) - L(ymin))) * (h - padT - padB);
  const yOf = (p) => (mode === "extrap" ? p.yex : p.yin);

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full">
      <rect x={padL} y={padT} width={w - padL - padR} height={h - padT - padB} fill="var(--chart-surface)" stroke="var(--chart-grid)" />
      {/* the "cheap AND reliable" corner - empty once you leave the training window */}
      <rect x={padL} y={sy(1)} width={Math.max(0, sx(1) - padL)} height={Math.max(0, h - padB - sy(1))} fill="#059669" opacity="0.07" />
      <text x={padL + 6} y={h - padB - 8} fontSize="9" fill="#059669" fontWeight="700">cheap AND reliable</text>

      {decadeTicks(xmin, xmax).map((v) => (
        <g key={`x${v}`}>
          <line x1={sx(v)} x2={sx(v)} y1={padT} y2={h - padB} stroke="var(--chart-grid)" strokeDasharray="2 4" />
          <text x={sx(v)} y={h - padB + 15} textAnchor="middle" fontSize="9" fill="var(--chart-axis)">{fmtSec(v)}</text>
        </g>
      ))}
      {decadeTicks(ymin, ymax).map((v) => (
        <g key={`y${v}`}>
          <line x1={padL} x2={w - padR} y1={sy(v)} y2={sy(v)} stroke="var(--chart-grid)" strokeDasharray="2 4" />
          <text x={padL - 8} y={sy(v) + 3} textAnchor="end" fontSize="9" fill="var(--chart-axis)">{fmtPct(v)}</text>
        </g>
      ))}

      {pts.map((p) => {
        const right = sx(p.x) > w - padR - 90;
        const anchor = right ? "end" : "start";
        const lx = right ? sx(p.x) - 12 : sx(p.x) + 12;
        const jump = mode === "both" && p.kind === "ml" && p.yex > p.yin * 1.2;
        return (
          <g key={p.n}>
            {jump && (
              <>
                <line x1={sx(p.x)} x2={sx(p.x)} y1={sy(p.yin)} y2={sy(p.yex) + 7} stroke={COLORS[p.n]} strokeWidth="2" opacity="0.55" />
                <path d={`M ${sx(p.x)} ${sy(p.yex)} l -4 -7 l 8 0 z`} fill={COLORS[p.n]} opacity="0.85" />
                <circle cx={sx(p.x)} cy={sy(p.yin)} r="4" fill="none" stroke={COLORS[p.n]} strokeWidth="2" />
              </>
            )}
            {p.kind === "ml"
              ? <circle cx={sx(p.x)} cy={sy(mode === "both" ? p.yex : yOf(p))} r="7" fill={COLORS[p.n]} />
              : <rect x={sx(p.x) - 6} y={sy(mode === "both" ? p.yex : yOf(p)) - 6} width="12" height="12" fill={COLORS[p.n]} />}
            <text x={lx} y={sy(mode === "both" ? p.yex : yOf(p)) + 4} textAnchor={anchor} fontSize="10.5" fontWeight="700" fill={COLORS[p.n]}>{p.n}</text>
          </g>
        );
      })}
      <text x={(padL + w - padR) / 2} y={h - 6} textAnchor="middle" fontSize="10" fill="var(--chart-axis)">deployment cost per problem (log) — lower is better</text>
      <text x={14} y={(padT + h - padB) / 2} textAnchor="middle" fontSize="10" fill="var(--chart-axis)"
        transform={`rotate(-90 14 ${(padT + h - padB) / 2})`}>error (log) — lower is better</text>
    </svg>
  );
}

function ScalingChart({ models }) {
  const w = 520, h = 250, padL = 56, padR = 20, padT = 18, padB = 44;
  const ns = ML.filter((m) => models[m] && models[m].scaling);
  if (!ns.length) return null;
  const allN = [], allT = [];
  ns.forEach((m) => { models[m].scaling.N.forEach((v) => allN.push(v)); models[m].scaling.t.forEach((v) => allT.push(v)); });
  const xmin = Math.min(...allN) * 0.8, xmax = Math.max(...allN) * 1.25;
  const ymin = Math.min(...allT) * 0.5, ymax = Math.max(...allT) * 2;
  const L = Math.log10;
  const sx = (v) => padL + ((L(v) - L(xmin)) / (L(xmax) - L(xmin))) * (w - padL - padR);
  const sy = (v) => h - padB - ((L(v) - L(ymin)) / (L(ymax) - L(ymin))) * (h - padT - padB);
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full">
      <rect x={padL} y={padT} width={w - padL - padR} height={h - padT - padB} fill="var(--chart-surface)" stroke="var(--chart-grid)" />
      {decadeTicks(xmin, xmax, 4).map((v) => (
        <g key={`x${v}`}>
          <line x1={sx(v)} x2={sx(v)} y1={padT} y2={h - padB} stroke="var(--chart-grid)" strokeDasharray="2 4" />
          <text x={sx(v)} y={h - padB + 15} textAnchor="middle" fontSize="9" fill="var(--chart-axis)">{v >= 1000 ? `${v / 1000}k` : v}</text>
        </g>
      ))}
      {decadeTicks(ymin, ymax, 4).map((v) => (
        <g key={`y${v}`}>
          <line x1={padL} x2={w - padR} y1={sy(v)} y2={sy(v)} stroke="var(--chart-grid)" strokeDasharray="2 4" />
          <text x={padL - 8} y={sy(v) + 3} textAnchor="end" fontSize="9" fill="var(--chart-axis)">{v >= 1 ? `${v}s` : `${v}s`}</text>
        </g>
      ))}
      {ns.map((m) => {
        const s = models[m].scaling;
        const d = s.N.map((n, i) => `${i ? "L" : "M"}${sx(n).toFixed(1)} ${sy(s.t[i]).toFixed(1)}`).join(" ");
        return (
          <g key={m}>
            <path d={d} fill="none" stroke={COLORS[m]} strokeWidth="2" />
            {s.N.map((n, i) => <circle key={i} cx={sx(n)} cy={sy(s.t[i])} r="3.5" fill={COLORS[m]} />)}
          </g>
        );
      })}
      <text x={(padL + w - padR) / 2} y={h - 6} textAnchor="middle" fontSize="10" fill="var(--chart-axis)">problem size N = n_t x n_x (log)</text>
      <text x={13} y={(padT + h - padB) / 2} textAnchor="middle" fontSize="10" fill="var(--chart-axis)"
        transform={`rotate(-90 13 ${(padT + h - padB) / 2})`}>time (s, log)</text>
    </svg>
  );
}

export default function CostAnalysis() {
  const [d, setD] = useState(null);
  const [mode, setMode] = useState("in");
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/m3/costs`).then((r) => r.json()).then((x) => { if (x.error) setErr(x.error); else setD(x); })
      .catch(() => setErr("Backend not reachable. Start it with: uvicorn main:app --port 8000"));
  }, []);

  const muted = "text-slate-500 dark:text-slate-400";
  const faint = "text-slate-400 dark:text-slate-500";
  const M = d?.models || {};
  const pinn = M.PINN || {};
  const inactive = "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700";
  const banner = {
    in: "Inside the window every option looks fine: FNO is 0.57% wrong for 0.44 s, and it sits in the cheap-and-reliable corner.",
    extrap: "Beyond the window the corner is empty. Every surrogate collapses - FNO 14.1%, PINN 35.8%, DeepONet 61.1% - while the numerical solvers stay exact but cost 2-4x more.",
    both: "The arrows are the whole problem: each surrogate's error jumps by 1-2 orders of magnitude the moment you leave the training window. The numerical solvers do not move.",
  }[mode];

  return (
    <div className="space-y-6">
      <div>
        <span className="text-xs font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">ML model analysis</span>
        <h1 className="text-3xl font-extrabold text-slate-800 dark:text-slate-100 mt-1">Cost analysis</h1>
        <p className="text-slate-500 dark:text-slate-400 mt-1 max-w-2xl">
          What each solver costs to deploy, and what accuracy that buys — every number measured on the same machine
          {d?.env?.repeats ? `, ${d.env.repeats} repeats, median reported` : ""}.
        </p>
      </div>
      {err && <div className="text-sm text-rose-600 dark:text-rose-300 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 rounded-lg px-3 py-2">{err}</div>}

      {d && (
        <>
          <Card title="Cost vs accuracy — the whole landscape"
            subtitle="circles = ML surrogates · squares = numerical solvers · the shaded corner is cheap AND reliable">
            <div className="flex gap-2 mb-3 flex-wrap">
              {[["in", "Inside training window"], ["extrap", "Beyond it"], ["both", "Show the jump"]].map(([v, l]) => (
                <button key={v} onClick={() => setMode(v)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium border ${mode === v ? "bg-indigo-600 text-white border-indigo-600" : inactive}`}>{l}</button>
              ))}
            </div>
            <Scatter models={M} mode={mode} />
            <div className="mt-2"><Banner ok={mode === "in"} text={banner} /></div>
          </Card>

          <div className="grid grid-cols-2 gap-5">
            <Card title="The PINN anomaly" subtitle="fastest to run, impossible to deploy">
              <div className="grid grid-cols-2 gap-2">
                <Stat label="inference" value={`${Math.round(pinn.infer_ms || 0)} ms`} tone="green" />
                <Stat label="deployment" value={`${Math.round((pinn.deploy_s || 0) / 60)} min`} tone="red" />
              </div>
              <p className={`text-xs mt-3 ${muted}`}>
                PINN has the <b>fastest inference of any model here</b> ({Math.round(pinn.infer_ms || 0)} ms vs FNO's {Math.round(M.FNO?.infer_ms || 0)} ms)
                and the smallest footprint ({(pinn.params || 0).toLocaleString()} parameters). But it is retrained for <b>every new problem</b>:
                {" "}{Math.round(pinn.train_s || 0)} s ({Math.round(pinn.adam_s || 0)} s Adam + {Math.round(pinn.lbfgs_s || 0)} s L-BFGS) —
                about {Math.round((pinn.deploy_s || 0) / (M.ColeHopf?.deploy_s || 1))}x the exact solver. Fast to run, unusable as a fast path.
              </p>
            </Card>

            <Card title="How cost grows with problem size" subtitle="all three surrogates scale linearly">
              <ScalingChart models={M} />
              <div className="flex items-center justify-center gap-4 text-[11px] mt-1 flex-wrap">
                {ML.filter((m) => M[m]?.scaling).map((m) => (
                  <span key={m} className="flex items-center gap-1" style={{ color: COLORS[m] }}>
                    <span className="w-3 h-1 rounded-full" style={{ background: COLORS[m] }} /> {m} · exp {M[m].scaling.exp.toFixed(2)}
                  </span>
                ))}
              </div>
            </Card>
          </div>

          <Card title="Footprint and throughput" subtitle="what each solver costs to store, load and run">
            <div className="overflow-hidden rounded-lg border border-slate-200 dark:border-slate-700">
              <table className="w-full text-xs">
                <thead className="bg-slate-50 dark:bg-slate-700/40 text-slate-500 dark:text-slate-400">
                  <tr>
                    <th className="text-left px-3 py-2">Solver</th>
                    <th className="text-right px-3 py-2">Parameters</th>
                    <th className="text-right px-3 py-2">Disk</th>
                    <th className="text-right px-3 py-2">Memory</th>
                    <th className="text-right px-3 py-2">Inference</th>
                    <th className="text-right px-3 py-2">Deploy / problem</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(M).map((m) => (
                    <tr key={m} className="border-t border-slate-100 dark:border-slate-700">
                      <td className="px-3 py-2 font-semibold" style={{ color: COLORS[m] }}>{m}</td>
                      <td className="px-3 py-2 text-right">{M[m].params ? M[m].params.toLocaleString() : "—"}</td>
                      <td className="px-3 py-2 text-right">{M[m].disk_mb ? `${M[m].disk_mb.toFixed(2)} MB` : "—"}</td>
                      <td className="px-3 py-2 text-right">{M[m].rss_mb ? `${Math.round(M[m].rss_mb)} MB` : "—"}</td>
                      <td className="px-3 py-2 text-right">{M[m].infer_ms ? `${Math.round(M[m].infer_ms)} ms` : "—"}</td>
                      <td className="px-3 py-2 text-right font-semibold">{M[m].deploy_s >= 60 ? `${Math.round(M[m].deploy_s / 60)} min` : `${(M[m].deploy_s || 0).toFixed(2)} s`}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className={`text-xs mt-2 ${faint}`}>
              {d.env?.processor ? d.env.processor.split(",")[0] : "project machine"}
              {d.env?.logical_cores ? ` · ${d.env.logical_cores} cores` : ""}{d.env?.torch ? ` · torch ${d.env.torch}` : ""}
              {d.env?.repeats ? ` · ${d.env.repeats} repeats, median` : ""}.
            </p>
          </Card>

          <div className="rounded-2xl p-5 bg-gradient-to-r from-indigo-50 via-white to-white dark:from-indigo-500/10 dark:via-slate-800 dark:to-slate-800 border border-indigo-200 dark:border-indigo-500/30">
            <div className="text-sm font-semibold text-indigo-800 dark:text-indigo-300">What this means for the project</div>
            <p className="text-sm text-slate-700 dark:text-slate-200 mt-1">
              Switch to "Beyond it" and the cheap-and-reliable corner empties out. No single solver is both cheap and trustworthy past the
              training window: the surrogates are fast but collapse, the numerical solvers stay exact but cost 2-4x more. That empty corner is
              exactly what the hybrid targets — run the cheap surrogate while it can be trusted, and buy numerical accuracy only where it is needed.
              The hybrid's own measured position on this trade-off is on the <span className="font-medium">Cost control</span> page.
            </p>
          </div>
        </>
      )}
    </div>
  );
}
