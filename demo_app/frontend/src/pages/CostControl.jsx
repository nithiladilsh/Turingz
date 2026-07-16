import { useEffect, useState } from "react";
import { Card, Stat, Banner } from "../components/ui.jsx";
import { LineChart } from "../components/Charts.jsx";
import { API } from "../api.js";

function thresholds(t) {
  const lo = Math.min(0.58, Math.max(0.12, 0.62 - 1.4 * t));
  return [lo, Math.min(0.9, lo + 0.12)];
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

export default function CostControl() {
  const [data, setData] = useState(null);
  const [rob, setRob] = useState(null);
  const [idx, setIdx] = useState(2);
  const [noise, setNoise] = useState(0.06);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/m3/frontier`).then((r) => r.json()).then((d) => {
      if (d.error) setErr(d.error); else { setData(d); setIdx(Math.min(2, d.frontier.length - 1)); }
    }).catch(() => setErr("Backend not reachable. Start it with: uvicorn main:app --port 8000"));
    fetch(`${API}/api/m3/robustness`).then((r) => r.json()).then(setRob).catch(() => {});
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
        <div className="space-y-4">
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

          {data && (
            <Card title="Where the compute goes (this setting)" subtitle="derived from the measured cost — moves with the knob">
              <div className="flex h-7 rounded overflow-hidden text-[11px] font-medium">
                <div style={{ width: `${(1 - corrFrac) * 100}%`, background: "#059669" }} className="text-white flex items-center justify-center">ML {Math.round((1 - corrFrac) * 100)}%</div>
                <div style={{ width: `${corrFrac * 100}%`, background: "#e11d48" }} className="text-white flex items-center justify-center">numerical {Math.round(corrFrac * 100)}%</div>
              </div>
              <p className={`text-xs mt-2 ${muted}`}>Cheap ML does most of the work; expensive numerical correction is spent only where needed. Tighten the target and the red numerical share grows.</p>
            </Card>
          )}

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
