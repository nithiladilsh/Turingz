import { useEffect, useState } from "react";
import { Card, Stat, Banner } from "../components/ui.jsx";
import { LineChart } from "../components/Charts.jsx";
import { API } from "../api.js";

function thresholds(t) {
  const lo = Math.min(0.58, Math.max(0.12, 0.62 - 1.4 * t));
  return [lo, Math.min(0.9, lo + 0.12)];
}

function logTicks(min, max) {
  const out = [];
  const lo = Math.floor(Math.log10(min)), hi = Math.ceil(Math.log10(max));
  for (let e = lo; e <= hi; e++) {
    for (const m of [1, 2, 5]) {
      const v = m * Math.pow(10, e);
      if (v >= min && v <= max) out.push(v);
    }
  }
  return out;
}
const fmtErr = (v) => {
  const p = v * 100;
  return p >= 1 ? `${Math.round(p)}%` : p >= 0.1 ? `${p.toFixed(1)}%` : `${p.toFixed(2)}%`;
};
const fmtSec = (v) => (v >= 1 ? `${v}s` : `${v}s`);
const fmtRel = (v) => `${v}x`;

function Ticks({ xs, ys, sx, sy, w, h, pad, fx, fy }) {
  return (
    <g>
      {xs.map((v) => (
        <g key={`x${v}`}>
          <line x1={sx(v)} x2={sx(v)} y1={h - pad} y2={h - pad + 4} stroke="var(--chart-axis)" />
          <text x={sx(v)} y={h - pad + 15} textAnchor="middle" fontSize="9" fill="var(--chart-axis)">{fx(v)}</text>
        </g>
      ))}
      {ys.map((v) => (
        <g key={`y${v}`}>
          <line x1={pad - 4} x2={pad} y1={sy(v)} y2={sy(v)} stroke="var(--chart-axis)" />
          <text x={pad - 7} y={sy(v) + 3} textAnchor="end" fontSize="9" fill="var(--chart-axis)">{fy(v)}</text>
        </g>
      ))}
    </g>
  );
}

function Bar({ label, value, max, display, color }) {
  const pct = Math.max(2, Math.min(100, (value / (max || 1)) * 100));
  return (
    <div className="flex items-center gap-2 text-xs">
      <div className="w-40 text-slate-500 dark:text-slate-400">{label}</div>
      <div className="flex-1 bg-slate-100 dark:bg-slate-700/40 rounded h-5">
        <div className="h-5 rounded" style={{ width: pct + "%", background: color }} />
      </div>
      <div className="w-16 text-right text-slate-700 dark:text-slate-200">{display}</div>
    </div>
  );
}

function FrontierChart({ frontier, pml, pnum, sel }) {
  const w = 520, h = 300, pad = 46;
  const pts = [...frontier, pml, pnum].filter(Boolean);
  const xs = pts.map((p) => p.cost), ys = pts.map((p) => Math.max(p.error, 5e-5));
  const xmin = Math.min(...xs) * 0.8, xmax = Math.max(...xs) * 1.25;
  const ymin = Math.min(...ys) * 0.6, ymax = Math.max(...ys) * 1.6;
  const L = Math.log10;
  const sx = (v) => pad + ((L(v) - L(xmin)) / (L(xmax) - L(xmin))) * (w - 2 * pad);
  const sy = (v) => h - pad - ((L(Math.max(v, 5e-5)) - L(ymin)) / (L(ymax) - L(ymin))) * (h - 2 * pad);
  const fr = [...frontier].sort((a, b) => a.cost - b.cost);
  const line = fr.map((p, i) => `${i ? "L" : "M"}${sx(p.cost).toFixed(1)} ${sy(p.error).toFixed(1)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full">
      <rect x={pad} y={pad} width={w - 2 * pad} height={h - 2 * pad} fill="var(--chart-surface)" stroke="var(--chart-grid)" />
      <Ticks xs={logTicks(xmin, xmax)} ys={logTicks(ymin, ymax)} sx={sx} sy={sy} w={w} h={h} pad={pad} fx={fmtSec} fy={fmtErr} />
      <path d={line} fill="none" stroke="#059669" strokeWidth="1.5" opacity="0.55" />
      {fr.map((p, i) => <circle key={i} cx={sx(p.cost)} cy={sy(p.error)} r="4" fill="#059669" />)}
      {pml && <rect x={sx(pml.cost) - 5} y={sy(pml.error) - 5} width="10" height="10" fill="#e11d48" />}
      {pnum && <path d={`M ${sx(pnum.cost)} ${sy(pnum.error) - 6} L ${sx(pnum.cost) + 6} ${sy(pnum.error) + 5} L ${sx(pnum.cost) - 6} ${sy(pnum.error) + 5} Z`} fill="#4f46e5" />}
      {sel && <circle cx={sx(sel.cost)} cy={sy(sel.error)} r="9" fill="none" stroke="#111827" strokeWidth="2.5" />}
      <text x={w / 2} y={h - 8} textAnchor="middle" fontSize="10" fill="var(--chart-axis)">cost (seconds, log) — lower is better</text>
      <text x={13} y={h / 2} textAnchor="middle" fontSize="10" fill="var(--chart-axis)" transform={`rotate(-90 13 ${h / 2})`}>error (log) — lower is better</text>
    </svg>
  );
}

function Strip({ flags, label, count }) {
  const w = 520, h = 20, n = flags.length;
  return (
    <div>
      <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400 mb-0.5">
        <span>{label}</span><span>{count} switch{count === 1 ? "" : "es"}</span>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full" preserveAspectRatio="none" style={{ height: 16 }}>
        {flags.map((f, i) => <rect key={i} x={(i / n) * w} y={0} width={w / n + 0.6} height={h} fill={f ? "#e11d48" : "#a7f3d0"} />)}
      </svg>
    </div>
  );
}

function noiseAt(i) { const s = Math.sin(i * 12.9898) * 43758.5453; return (s - Math.floor(s)) * 2 - 1; }
function flipCount(a) { return a.reduce((n, f, i) => n + (i && f !== a[i - 1] ? 1 : 0), 0); }

function CompareChart({ models }) {
  const w = 520, h = 300, pad = 48;
  const names = Object.keys(models);
  const pts = [];
  names.forEach((n) => { models[n].frontier.forEach((p) => pts.push(p)); pts.push(models[n].pure_ml); });
  const xs = pts.map((p) => Math.max(p.rel_cost, 0.02));
  const ys = pts.map((p) => Math.max(p.error, 5e-5));
  const xmin = Math.min(...xs, 0.9) * 0.7, xmax = Math.max(...xs, 1.1) * 1.35;
  const ymin = Math.min(...ys) * 0.6, ymax = Math.max(...ys) * 1.5;
  const L = Math.log10;
  const sx = (v) => pad + ((L(Math.max(v, 0.02)) - L(xmin)) / (L(xmax) - L(xmin))) * (w - 2 * pad);
  const sy = (v) => h - pad - ((L(Math.max(v, 5e-5)) - L(ymin)) / (L(ymax) - L(ymin))) * (h - 2 * pad);
  const colors = { FNO: "#059669", DeepONet: "#e11d48" };
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full">
      <rect x={pad} y={pad} width={w - 2 * pad} height={h - 2 * pad} fill="var(--chart-surface)" stroke="var(--chart-grid)" />
      <Ticks xs={logTicks(xmin, xmax)} ys={logTicks(ymin, ymax)} sx={sx} sy={sy} w={w} h={h} pad={pad} fx={fmtRel} fy={fmtErr} />
      <line x1={sx(1)} x2={sx(1)} y1={pad} y2={h - pad} stroke="#4f46e5" strokeWidth="2" strokeDasharray="5 4" />
      <text x={sx(1) + 5} y={pad + 12} fontSize="9" fill="#4f46e5" fontWeight="700">= numerical cost</text>
      {names.map((n) => {
        const fr = [...models[n].frontier].sort((a, b) => a.rel_cost - b.rel_cost);
        const d = fr.map((p, i) => `${i ? "L" : "M"}${sx(p.rel_cost).toFixed(1)} ${sy(p.error).toFixed(1)}`).join(" ");
        return (
          <g key={n}>
            <path d={d} fill="none" stroke={colors[n] || "#64748b"} strokeWidth="2" />
            {fr.map((p, i) => <circle key={i} cx={sx(p.rel_cost)} cy={sy(p.error)} r="4" fill={colors[n] || "#64748b"} />)}
          </g>
        );
      })}
      <text x={w / 2} y={h - 8} textAnchor="middle" fontSize="10" fill="var(--chart-axis)">cost relative to pure-numerical (log) — left of the line is cheaper</text>
      <text x={13} y={h / 2} textAnchor="middle" fontSize="10" fill="var(--chart-axis)" transform={`rotate(-90 13 ${h / 2})`}>error (log) — lower is better</text>
    </svg>
  );
}

export default function CostControl() {
  const [data, setData] = useState(null);
  const [rob, setRob] = useState(null);
  const [idx, setIdx] = useState(2);
  const [noise, setNoise] = useState(0.06);
  const [regime, setRegime] = useState(null);
  const [cmp, setCmp] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/m3/frontier`).then((r) => r.json()).then((d) => {
      if (d.error) setErr(d.error); else { setData(d); setIdx(Math.min(2, d.frontier.length - 1)); }
    }).catch(() => setErr("Backend not reachable. Start it with: uvicorn main:app --port 8000"));
    fetch(`${API}/api/m3/robustness`).then((r) => r.json()).then(setRob).catch(() => {});
    fetch(`${API}/api/m3/regime`).then((r) => r.json()).then(setRegime).catch(() => {});
    fetch(`${API}/api/m3/frontiers`).then((r) => r.json()).then(setCmp).catch(() => {});
  }, []);

  const muted = "text-slate-500 dark:text-slate-400";
  const faint = "text-slate-400 dark:text-slate-500";
  const sel = data ? data.frontier[idx] : null;
  const [lo, hi] = sel ? thresholds(sel.target) : [0.4, 0.6];
  const pct = (v) => `${(v * 100).toFixed(1)}%`;
  const cheaper = sel && data ? (data.pure_numerical.cost / sel.cost).toFixed(1) : "—";
  const sharper = sel && data ? (data.pure_ml.error / sel.error).toFixed(1) : "—";
  const corrFrac = data && sel
    ? Math.max(0, Math.min(1, (sel.cost - data.pure_ml.cost) / ((data.pure_numerical.cost - data.pure_ml.cost) || 1)))
    : 0;
  const robMax = rob ? Math.max(rob.fixed_err || 0, rob.adaptive_err || 0, 0.001) : 1;
  const oodMax = rob ? Math.max(rob.ood_err || 0, rob.indist_err || 0, 0.001) : 1;

  // live hysteresis illustration
  const N = 120, tArr = [], trust = [];
  for (let i = 0; i < N; i++) {
    const t = (2 * i) / (N - 1); tArr.push(t);
    const base = 0.9 - 0.75 * (i / (N - 1));
    trust.push(Math.max(0, Math.min(1, base + noise * noiseAt(i))));
  }
  const single = trust.map((v) => v < lo);
  const hyst = []; let on = false;
  for (const v of trust) { if (!on && v < lo) on = true; else if (on && v > hi) on = false; hyst.push(on); }
  const singleFlips = flipCount(single), hystFlips = flipCount(hyst);

  return (
    <div>
      <h1 className="text-2xl font-bold text-slate-800 dark:text-slate-100">Cost control — cost-aware adaptive control (Module 3)</h1>
      <p className="text-slate-600 dark:text-slate-300 mt-1 max-w-3xl text-sm">
        Turn one accuracy knob and watch it map to switch thresholds, split the compute effort, and slide along the
        measured cost/accuracy frontier. All frontier numbers are real, measured end-to-end (FNO + M1 trust + M2 coupling).
      </p>
      {err && <div className="mt-3 text-sm text-rose-600 dark:text-rose-300 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 rounded-lg px-3 py-2">{err}</div>}

      <div className="mt-5 grid grid-cols-[320px_1fr] gap-5">
        {/* CONTROLS */}
        <div className="space-y-4 sticky top-6 self-start">
          <Card title="The accuracy knob" subtitle="one target in — a full correction schedule out">
            {data ? (
              <div className="space-y-3">
                <div>
                  <div className={`flex justify-between text-xs ${muted}`}><span>Accuracy target</span><span>{sel.target}</span></div>
                  <input type="range" min="0" max={data.frontier.length - 1} value={idx} onChange={(e) => setIdx(+e.target.value)} className="w-full" />
                  <div className={`flex justify-between text-[11px] ${faint}`}><span>loose (0.30)</span><span>tight (0.01)</span></div>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <Stat label="switch threshold lo" value={lo.toFixed(2)} tone="indigo" />
                  <Stat label="switch threshold hi" value={hi.toFixed(2)} tone="indigo" />
                </div>
                <p className={`text-xs ${muted}`}>thresholds_for_target(): a tighter target lowers the switch threshold, so the controller switches earlier and spends more numerical effort.</p>
              </div>
            ) : <p className={`text-sm ${faint}`}>Loading measured frontier…</p>}
          </Card>

          {sel && (
            <Card title="Operating point">
              <div className="grid grid-cols-1 gap-2">
                <Stat label="cost" value={`${sel.cost.toFixed(2)} s`} tone="green" />
                <Stat label="error" value={pct(sel.error)} tone="slate" />
                <Stat label="target hit-rate" value={`${Math.round(sel.hit_rate * 100)}%`} tone={sel.hit_rate >= 0.9 ? "green" : "red"} />
              </div>
            </Card>
          )}
          {data && (
            <Card title="Where the compute goes" subtitle="derived from measured cost - moves with the knob">
              <div className="flex h-7 rounded overflow-hidden text-[11px] font-medium">
                <div style={{ width: `${(1 - corrFrac) * 100}%`, background: "#059669" }} className="text-white flex items-center justify-center">ML {Math.round((1 - corrFrac) * 100)}%</div>
                <div style={{ width: `${corrFrac * 100}%`, background: "#e11d48" }} className="text-white flex items-center justify-center">num {Math.round(corrFrac * 100)}%</div>
              </div>
              <p className={`text-xs mt-2 ${muted}`}>Cheap ML does most of the work; expensive numerical correction is spent only where needed. Tighten the target and the red share grows.</p>
            </Card>
          )}
        </div>

        {/* RESULTS */}
        <div className="space-y-4">
          <Card title="Measured cost / accuracy frontier"
            subtitle={data ? `real values · source: ${data.source} · black ring = your chosen operating point` : "loading"}>
            {data ? (
              <>
                <FrontierChart frontier={data.frontier} pml={data.pure_ml} pnum={data.pure_numerical} sel={sel} />
                <div className={`text-xs ${faint}`}>green = hybrid frontier (the knob) · red square = pure-ML · blue triangle = pure-numerical</div>
                <div className="mt-3"><Banner ok text={`At this setting: ~${cheaper}x cheaper than pure-numerical and ~${sharper}x more accurate than pure-ML.`} /></div>
              </>
            ) : <p className={`text-sm ${faint}`}>Loading…</p>}
          </Card>

          <div className="grid grid-cols-2 gap-4">
            <Card title="Adaptive vs fixed controller" subtitle="matched correction budget">
              {rob && rob.adaptive_err != null ? (
                <div className="space-y-2">
                  <Bar label={`adaptive (${rob.adaptive_corr} corr)`} value={rob.adaptive_err} max={robMax} display={pct(rob.adaptive_err)} color="#059669" />
                  <Bar label={`fixed (${rob.fixed_corr} corr)`} value={rob.fixed_err} max={robMax} display={pct(rob.fixed_err)} color="#e11d48" />
                  <p className={`text-xs ${muted}`}>Same number of corrections — the adaptive controller spends them where trust is low, so it hits the target while the fixed baseline does not.</p>
                </div>
              ) : <p className={`text-sm ${faint}`}>—</p>}
            </Card>

            <Card title="Robustness — in-dist vs OOD" subtitle="mean error, loose target">
              {rob && rob.indist_err != null ? (
                <div className="space-y-2">
                  <Bar label="in-distribution" value={rob.indist_err} max={oodMax} display={pct(rob.indist_err)} color="#4f46e5" />
                  <Bar label="out-of-distribution" value={rob.ood_err} max={oodMax} display={pct(rob.ood_err)} color="#d97706" />
                  <p className={`text-xs ${muted}`}>OOD inputs are harder, but the controller still holds a low, bounded error — the operating curve degrades gracefully.</p>
                </div>
              ) : <p className={`text-sm ${faint}`}>—</p>}
            </Card>
          </div>

          <Card title="Why the deadband — hysteresis stops chattering"
            subtitle="a noisy trust signal · red dashed = θ_lo · green dashed = θ_hi">
            <LineChart h={180} xr={[0, 2]} yr={[0, 1]} xlabel="time t"
              series={[
                { x: tArr, y: trust, color: "#4f46e5", width: 2 },
                { x: [0, 2], y: [lo, lo], color: "#e11d48", dashed: true, width: 1 },
                { x: [0, 2], y: [hi, hi], color: "#059669", dashed: true, width: 1 },
              ]} />
            <div className="mt-3 space-y-2">
              <Strip flags={single} label="naive single threshold (correct when trust < θ_lo)" count={singleFlips} />
              <Strip flags={hyst} label="hysteresis deadband (my controller)" count={hystFlips} />
              <div className={`text-[11px] ${faint}`}>green = running ML · red = numerical correction</div>
            </div>
            <div className="mt-3">
              <div className={`flex justify-between text-xs ${muted}`}><span>Trust-signal noise</span><span>{noise.toFixed(2)}</span></div>
              <input type="range" min="0" max="0.1" step="0.01" value={noise} onChange={(e) => setNoise(+e.target.value)} className="w-full" />
            </div>
            <p className={`text-sm mt-2 text-slate-600 dark:text-slate-300`}>
              Turn up the noise: the naive threshold flips on and off <b>{singleFlips}</b> times (wasted corrections), while the deadband switches just <b>{hystFlips}</b> — it only stops correcting once trust climbs clear of θ_hi. That gap is the anti-chatter mechanism.
            </p>
          </Card>

          {cmp && cmp.FNO && cmp.DeepONet && (
            <Card title="Does the hybrid always pay off? FNO vs DeepONet (both measured)"
              subtitle="each frontier normalised by its own pure-numerical baseline, so the two runs are comparable">
              <CompareChart models={cmp} />
              <div className="flex items-center justify-center gap-5 text-[11px] text-slate-400 dark:text-slate-500">
                <span className="flex items-center gap-1"><span className="w-3 h-1 rounded-full bg-emerald-600" /> FNO</span>
                <span className="flex items-center gap-1"><span className="w-3 h-1 rounded-full bg-rose-600" /> DeepONet</span>
                <span className="flex items-center gap-1"><span className="w-4 border-t-2 border-dashed border-indigo-600" /> numerical cost (1.0x)</span>
              </div>
              <p className="text-sm mt-3 text-slate-600 dark:text-slate-300">
                Left of the dashed line the hybrid is cheaper than simply running the numerical solver; right of it you would be better off not using the hybrid at all.
                <b> FNO sits at 0.28-0.52x</b> numerical cost at 3-5% error. <b>DeepONet sits at 1.26-3.13x</b> and stays stuck near 23% error - more expensive
                AND less accurate than the numerical solver, with target hit-rate collapsing to 0%. The controller does not create accuracy; it protects a
                surrogate that is already worth trusting.
              </p>
            </Card>
          )}

          {regime && regime.surrogates && (
            <Card title="Where the controller applies - the operating regime"
              subtitle="measured deployment cost and in-window accuracy for all three surrogates">
              <div className="overflow-hidden rounded-lg border border-slate-200 dark:border-slate-700">
                <table className="w-full text-xs">
                  <thead className="bg-slate-50 dark:bg-slate-700/40 text-slate-500 dark:text-slate-400">
                    <tr>
                      <th className="text-left px-3 py-2">Surrogate</th>
                      <th className="text-right px-3 py-2">Deploy cost</th>
                      <th className="text-right px-3 py-2">In-window error</th>
                      <th className="text-center px-3 py-2">Amortized?</th>
                      <th className="text-center px-3 py-2">Accurate enough?</th>
                      <th className="text-center px-3 py-2">Hybrid pays off?</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[["FNO", true, true], ["DeepONet", true, false], ["PINN", false, true]].map(([m, amort, acc]) => {
                      const v = regime.surrogates[m] || {};
                      const ok = amort && acc;
                      const cost = v.deploy_s == null ? "-" : (v.deploy_s >= 100 ? `${Math.round(v.deploy_s)} s` : `${v.deploy_s.toFixed(2)} s`);
                      return (
                        <tr key={m} className="border-t border-slate-100 dark:border-slate-700">
                          <td className="px-3 py-2 font-medium text-slate-700 dark:text-slate-200">{m}</td>
                          <td className="px-3 py-2 text-right">{cost}</td>
                          <td className="px-3 py-2 text-right">{v.err_in == null ? "-" : `${v.err_in.toFixed(2)}%`}</td>
                          <td className={`px-3 py-2 text-center font-semibold ${amort ? "text-emerald-600" : "text-rose-600"}`}>{amort ? "yes" : "no"}</td>
                          <td className={`px-3 py-2 text-center font-semibold ${acc ? "text-emerald-600" : "text-rose-600"}`}>{acc ? "yes" : "no"}</td>
                          <td className={`px-3 py-2 text-center font-bold ${ok ? "text-emerald-600" : "text-rose-600"}`}>{ok ? "YES" : "NO"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p className={`text-xs mt-2 ${muted}`}>
                The controller needs a surrogate that is <b>both</b> amortized (one cheap forward pass) <b>and</b> accurate in-window (worth trusting before it drifts).
                FNO is the only one of the three that is both, which is why the frontier above is measured on FNO. DeepONet is cheap but ~29% wrong in-window, so there is
                nothing worth trusting; PINN is accurate but re-optimises per instance at ~2114 s, roughly 950x the numerical solver, so there is no cheap path to protect.
                That is a stated precondition of the method, not a gap in it.
              </p>
            </Card>
          )}

          <Card title="Why this is the novelty (in code)">
            <div className="text-sm text-slate-600 dark:text-slate-300 space-y-2">
              <p>No prior hybrid solver has an accuracy-budget knob. Mine maps a target directly to a correction schedule:</p>
              <div className="font-mono text-xs bg-slate-50 dark:bg-slate-700/40 rounded-lg px-3 py-2">
                thresholds_for_target({sel ? sel.target : "target"}) → θ_lo = {lo.toFixed(2)}, θ_hi = {hi.toFixed(2)}
              </div>
              <p className={muted}>Then the controller corrects only while trust &lt; θ_lo and releases at θ_hi (the deadband above), spending numerical effort only when needed. The frontier is the measured proof it pays off.</p>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
