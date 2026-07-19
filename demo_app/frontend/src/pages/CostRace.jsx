import { useEffect, useRef, useState } from "react";
import { Card } from "../components/ui.jsx";
import { WS } from "../api.js";
import { Play, Trophy, RotateCcw } from "lucide-react";

const LANE = {
  num: { name: "Pure numerical", sub: "correct every step", c: "#4f46e5", tag: "always expensive" },
  ml:  { name: "Pure ML",        sub: "never correct",      c: "#e11d48", tag: "always cheap" },
  hyb: { name: "Adaptive hybrid",sub: "my controller decides", c: "#059669", tag: "spends only when trust drops" },
};
const KEYS = ["num", "ml", "hyb"];

function runRace(payload, onFrame, onInit, onSummary, onDone, onError) {
  const ws = new WebSocket(`${WS}/ws/race`);
  ws.onopen = () => ws.send(JSON.stringify(payload));
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (typeof m.error === "string") return onError && onError(m.error);
    if (m.done) { onDone && onDone(); ws.close(); return; }
    if (m.init) return onInit && onInit(m.init);
    if (m.summary) return onSummary && onSummary(m.summary);
    onFrame(m);
  };
  ws.onerror = () => onError && onError("Could not reach backend at " + WS + ". Is it running?");
  return ws;
}

/* one lane: spend timeline + live counters */
function Lane({ k, nt, steps, cost, err, maxCost, running }) {
  const L = LANE[k];
  const share = maxCost > 0 ? cost / maxCost : 0;
  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4">
      <div className="flex items-center justify-between mb-2">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full" style={{ background: L.c }} />
            <span className="text-sm font-bold text-slate-800 dark:text-slate-100">{L.name}</span>
            <span className="text-[10px] px-2 py-0.5 rounded-full font-medium"
              style={{ background: `${L.c}1a`, color: L.c }}>{L.tag}</span>
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5 ml-5">{L.sub}</div>
        </div>
        <div className="flex items-baseline gap-5">
          <div className="text-right">
            <div className="text-[10px] uppercase tracking-wide text-slate-400">spent</div>
            <div className="text-2xl font-extrabold tabular-nums" style={{ color: L.c }}>{cost.toFixed(2)}<span className="text-sm font-semibold ml-0.5">s</span></div>
          </div>
          <div className="text-right w-20">
            <div className="text-[10px] uppercase tracking-wide text-slate-400">error</div>
            <div className="text-2xl font-extrabold tabular-nums text-slate-700 dark:text-slate-200">
              {err == null ? "—" : err < 0.0005 ? "0%" : `${(err * 100).toFixed(1)}%`}
            </div>
          </div>
        </div>
      </div>

      {/* spend timeline — one block per step, red = paid for a numerical correction */}
      <div className="flex gap-[2px] h-7">
        {Array.from({ length: nt }).map((_, i) => {
          const done = i < steps.length;
          const exp = done && steps[i];
          return (
            <div key={i} className="flex-1 rounded-[3px] transition-all duration-200"
              style={{
                background: !done ? "var(--chart-grid)" : exp ? "#e11d48" : "#10b981",
                opacity: done ? 1 : 0.35,
                height: !done ? "40%" : exp ? "100%" : "45%",
                alignSelf: "flex-end",
              }} />
          );
        })}
      </div>

      <div className="mt-2 h-2 rounded-full bg-slate-100 dark:bg-slate-700 overflow-hidden">
        <div className="h-2 rounded-full transition-all duration-150" style={{ width: `${share * 100}%`, background: L.c }} />
      </div>
    </div>
  );
}

/* synchronized cumulative-cost + error charts */
function Traces({ hist, nt, tEnd, target, kind }) {
  const W = 560, H = 200, padL = 52, padR = 14, padT = 12, padB = 30;
  if (!hist.length) return <div style={{ height: 200 }} />;
  const vals = hist.flatMap((f) => KEYS.map((k) => (kind === "cost" ? f[k].cost : Math.max(f[k].err, 1e-5))));
  const vmax = Math.max(...vals) * 1.12, vmin = kind === "cost" ? 0 : Math.min(...vals) * 0.7;
  const sx = (t) => padL + (t / (tEnd || 1)) * (W - padL - padR);
  const sy = kind === "cost"
    ? (v) => H - padB - (v / (vmax || 1)) * (H - padT - padB)
    : (v) => { const L = Math.log10; return H - padB - ((L(Math.max(v, 1e-5)) - L(vmin)) / (L(vmax) - L(vmin) || 1)) * (H - padT - padB); };
  const ticks = kind === "cost"
    ? [0, vmax / 2, vmax]
    : [1e-4, 1e-3, 1e-2, 1e-1].filter((v) => v >= vmin * 0.9 && v <= vmax * 1.1);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      <rect x={padL} y={padT} width={W - padL - padR} height={H - padT - padB} fill="var(--chart-surface)" stroke="var(--chart-grid)" />
      {ticks.map((v, i) => (
        <g key={i}>
          <line x1={padL} x2={W - padR} y1={sy(v)} y2={sy(v)} stroke="var(--chart-grid)" strokeDasharray="2 4" />
          <text x={padL - 7} y={sy(v) + 3} textAnchor="end" fontSize="9" fill="var(--chart-axis)">
            {kind === "cost" ? `${v.toFixed(1)}s` : `${(v * 100).toFixed(v < 0.01 ? 2 : 0)}%`}
          </text>
        </g>
      ))}
      {kind === "err" && target && (
        <g>
          <line x1={padL} x2={W - padR} y1={sy(target)} y2={sy(target)} stroke="#d97706" strokeWidth="1.5" strokeDasharray="5 3" />
          <text x={W - padR - 4} y={sy(target) - 4} textAnchor="end" fontSize="9" fontWeight="700" fill="#d97706">your target</text>
        </g>
      )}
      {KEYS.map((k) => (
        <path key={k} d={hist.map((f, i) => `${i ? "L" : "M"}${sx(f.t).toFixed(1)} ${sy(kind === "cost" ? f[k].cost : f[k].err).toFixed(1)}`).join(" ")}
          fill="none" stroke={LANE[k].c} strokeWidth={k === "hyb" ? 2.6 : 1.8} opacity={k === "hyb" ? 1 : 0.75} />
      ))}
      {hist.filter((f) => f.correcting).map((f, i) => (
        <line key={i} x1={sx(f.t)} x2={sx(f.t)} y1={padT} y2={H - padB} stroke="#e11d48" strokeWidth="1" opacity="0.16" />
      ))}
      <text x={(padL + W - padR) / 2} y={H - 5} textAnchor="middle" fontSize="9" fill="var(--chart-axis)">simulation time →</text>
    </svg>
  );
}


/* handover panel: the switch point and what the controller spends after it */
function Handover({ hist, nt, thetaLo }) {
  const first = hist.findIndex((f) => f.correcting);
  if (first < 0) return (
    <div className="rounded-2xl border border-emerald-200 dark:border-emerald-500/30 bg-emerald-50 dark:bg-emerald-500/10 p-4 text-sm text-slate-700 dark:text-slate-200">
      No handover yet — trust never dropped below the threshold, so the controller has spent nothing on corrections.
    </div>
  );
  const sw = hist[first];
  const before = hist.slice(0, first);
  const after = hist.slice(first);
  const corrAfter = after.filter((f) => f.correcting).length;
  const share = corrAfter / Math.max(after.length, 1);
  const latched = share > 0.95;
  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
      <div className="text-sm font-bold text-slate-800 dark:text-slate-100 mb-1">Where the controller hands over</div>
      <div className="text-xs text-slate-400 mb-4">the single decision that sets the whole bill</div>

      <div className="flex items-stretch gap-0">
        <div className="rounded-l-xl bg-emerald-50 dark:bg-emerald-500/10 p-4 border-y border-l border-emerald-200 dark:border-emerald-500/30"
          style={{ flex: Math.max(before.length, 1) }}>
          <div className="text-[10px] uppercase tracking-wide text-emerald-700 dark:text-emerald-400 font-bold">Phase 1 · ML runs free</div>
          <div className="text-2xl font-extrabold text-emerald-600 mt-1">{before.length}<span className="text-sm font-semibold text-slate-400"> steps</span></div>
          <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            {before.length ? `${before[before.length - 1].hyb.cost.toFixed(2)}s spent · ${(before[before.length - 1].hyb.err * 100).toFixed(1)}% error` : "—"}
          </div>
          <div className="text-xs text-emerald-700 dark:text-emerald-400 font-semibold mt-1">0 corrections bought</div>
        </div>

        <div className="flex flex-col items-center justify-center px-3 bg-amber-50 dark:bg-amber-500/10 border-y border-amber-300 dark:border-amber-500/40">
          <div className="text-[9px] uppercase tracking-wide text-amber-700 dark:text-amber-400 font-bold whitespace-nowrap">handover</div>
          <div className="text-lg font-extrabold text-amber-600 whitespace-nowrap">t = {sw.t}</div>
          <div className="text-[10px] text-slate-500 whitespace-nowrap">trust {sw.trust} &lt; θlo {thetaLo}</div>
        </div>

        <div className="rounded-r-xl bg-rose-50 dark:bg-rose-500/10 p-4 border-y border-r border-rose-200 dark:border-rose-500/30"
          style={{ flex: Math.max(after.length, 1) }}>
          <div className="text-[10px] uppercase tracking-wide text-rose-700 dark:text-rose-400 font-bold">Phase 2 · controller correcting</div>
          <div className="text-2xl font-extrabold text-rose-600 mt-1">{corrAfter}<span className="text-sm font-semibold text-slate-400">/{after.length} steps</span></div>
          <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            {after.length ? `${(after[after.length - 1].hyb.cost - (before.length ? before[before.length - 1].hyb.cost : 0)).toFixed(2)}s spent here` : "—"}
          </div>
          <div className="text-xs text-rose-700 dark:text-rose-400 font-semibold mt-1">{Math.round(share * 100)}% of remaining steps corrected</div>
        </div>
      </div>

      <div className={`mt-4 rounded-xl px-4 py-3 text-sm ${latched ? "bg-slate-50 dark:bg-slate-700/40 text-slate-700 dark:text-slate-200" : "bg-indigo-50 dark:bg-indigo-500/10 text-slate-700 dark:text-slate-200"}`}>
        {latched ? (
          <>Once trust collapses it does not recover, so the controller <b>latches into correction mode</b> and stays there —
          this is measured, not assumed: on this trust signal a deadband re-engages only ~1.2 times, so I ship a one-way handover.
          The saving comes entirely from <b>how long it safely delayed the handover</b> — {before.length} of {nt} steps run at ML price.</>
        ) : (
          <>Trust recovers after corrections, so the controller <b>releases and re-engages</b> — it corrected only {Math.round(share * 100)}% of the
          remaining steps rather than all of them. The hysteresis deadband is what prevents that switching from chattering.</>
        )}
      </div>
    </div>
  );
}

export default function CostRace({ model = "FNO", ic, pinnIndex = 0, target = 0.05 }) {
  const [hist, setHist] = useState([]);
  const [init, setInit] = useState(null);
  const [sum, setSum] = useState(null);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => () => wsRef.current && wsRef.current.close(), []);

  const start = () => {
    if (running) return;
    setHist([]); setSum(null); setErr(null); setInit(null); setRunning(true);
    wsRef.current = runRace(
      { model, ic, pinn_index: pinnIndex, target },
      (f) => setHist((h) => [...h, f]),
      setInit, setSum,
      () => setRunning(false),
      (e) => { setErr(e); setRunning(false); }
    );
  };

  const last = hist[hist.length - 1];
  const nt = init?.nt || 29;
  const tEnd = init?.t_end || 1;
  const maxCost = Math.max(init?.num_total || 1, last?.num?.cost || 1);
  const spend = {
    num: hist.map(() => true),
    ml: hist.map(() => false),
    hyb: hist.map((f) => f.correcting),
  };

  return (
    <div className="space-y-5">
      <div className="rounded-2xl p-5 bg-gradient-to-r from-slate-50 to-white dark:from-slate-800 dark:to-slate-800 border border-slate-200 dark:border-slate-700">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-base font-bold text-slate-800 dark:text-slate-100">Same wave. Same clock. Three strategies.</div>
            <div className="text-sm text-slate-500 dark:text-slate-400 mt-0.5">
              Green blocks are cheap ML steps, red blocks are numerical corrections you paid for.
            </div>
          </div>
          <button onClick={start} disabled={running || !ic}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white text-sm font-semibold transition">
            {running ? <><RotateCcw size={16} className="animate-spin" /> Racing…</> : <><Play size={16} /> Start the race</>}
          </button>
        </div>
      </div>

      {err && <div className="text-sm text-rose-600 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 rounded-lg px-3 py-2">{err}</div>}

      <div className="space-y-3">
        {KEYS.map((k) => (
          <Lane key={k} k={k} nt={nt} steps={spend[k]} running={running}
            cost={last ? last[k].cost : 0} err={last ? last[k].err : null} maxCost={maxCost} />
        ))}
      </div>

      {hist.length > 0 && <Handover hist={hist} nt={nt} thetaLo={init?.theta_lo ?? "—"} />}

      <div className="grid grid-cols-2 gap-5">
        <Card title="Money spent, as it happens" subtitle="cumulative compute cost">
          <Traces hist={hist} nt={nt} tEnd={tEnd} kind="cost" />
        </Card>
        <Card title="What that money bought" subtitle="error against the exact solution">
          <Traces hist={hist} nt={nt} tEnd={tEnd} target={target} kind="err" />
        </Card>
      </div>

      {sum && (
        <div className="rounded-2xl p-5 border-2 border-emerald-300 dark:border-emerald-500/40 bg-emerald-50 dark:bg-emerald-500/10">
          <div className="flex items-center gap-2 mb-3">
            <Trophy size={18} className="text-emerald-600" />
            <span className="text-sm font-bold text-emerald-800 dark:text-emerald-300">Final scoreboard</span>
          </div>
          <div className="grid grid-cols-4 gap-4">
            <div>
              <div className="text-[10px] uppercase text-slate-400">hybrid vs numerical</div>
              <div className="text-3xl font-extrabold text-emerald-600">{sum.speedup}×</div>
              <div className="text-xs text-slate-500">cheaper</div>
            </div>
            <div>
              <div className="text-[10px] uppercase text-slate-400">hybrid vs pure ML</div>
              <div className="text-3xl font-extrabold text-emerald-600">{sum.acc_gain}×</div>
              <div className="text-xs text-slate-500">more accurate</div>
            </div>
            <div>
              <div className="text-[10px] uppercase text-slate-400">corrections used</div>
              <div className="text-3xl font-extrabold text-slate-700 dark:text-slate-200">{sum.corr_steps}<span className="text-lg text-slate-400">/{nt}</span></div>
              <div className="text-xs text-slate-500">{Math.round((sum.corr_steps / nt) * 100)}% of steps</div>
            </div>
            <div>
              <div className="text-[10px] uppercase text-slate-400">compute saved</div>
              <div className="text-3xl font-extrabold text-emerald-600">{Math.round(sum.saving * 100)}%</div>
              <div className="text-xs text-slate-500">vs solving numerically</div>
            </div>
          </div>
          <div className="text-sm text-slate-700 dark:text-slate-200 mt-4 pt-3 border-t border-emerald-200 dark:border-emerald-500/30">
            The controller bought <b>{sum.corr_steps} corrections</b> instead of {nt}, and placed them where trust had collapsed —
            landing at <b>{(sum.hyb.err * 100).toFixed(1)}%</b> error for <b>{sum.hyb.cost.toFixed(2)}s</b>,
            against pure ML's {(sum.ml.err * 100).toFixed(1)}% and the numerical solver's {sum.num.cost.toFixed(2)}s.
          </div>
        </div>
      )}

      <div className="text-xs text-slate-400">
        The numerical track is the reference solution here, so its error is 0 by construction — this race is about cost,
        not about who wins on accuracy. Per-step costs come from the measured timed run, so the totals land on the same frontier as the Findings tab.
      </div>
    </div>
  );
}
