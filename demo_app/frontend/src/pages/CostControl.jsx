import { useEffect, useRef, useState } from "react";
import { Card } from "../components/ui.jsx";
import { API, WS, getMeta, buildIC, pinnIC } from "../api.js";
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

/* ---------------- page ---------------- */
export default function CostControl() {
  const [tab, setTab] = useState("findings");
  const [cmp, setCmp] = useState(null);
  const [rob, setRob] = useState(null);
  const [regime, setRegime] = useState(null);
  const [pick, setPick] = useState("FNO");
  const [idx, setIdx] = useState(2);
  const [meta, setMeta] = useState(null);
  const [modes, setModes] = useState(2);
  const [amp, setAmp] = useState(1.0);
  const [model, setModel] = useState("FNO");
  const [pidx, setPidx] = useState(0);
  const [ic, setIc] = useState(null);
  const [frame, setFrame] = useState(null);
  const [hist, setHist] = useState([]);
  const [sum, setSum] = useState(null);
  const [running, setRunning] = useState(false);
  const wsRef = useRef(null);
  const [err, setErr] = useState(null);
  const p = useDraw(1400, tab);

  useEffect(() => {
    fetch(`${API}/api/m3/frontiers`).then((r) => r.json()).then(setCmp)
      .catch(() => setErr("Backend not reachable — start it with: uvicorn main:app"));
    fetch(`${API}/api/m3/robustness`).then((r) => r.json()).then(setRob).catch(() => {});
    fetch(`${API}/api/m3/regime`).then((r) => r.json()).then(setRegime).catch(() => {});
    getMeta().then(setMeta).catch(() => {});
  }, []);

  useEffect(() => {
    if (!meta) return;
    if (model === "PINN") pinnIC(pidx).then((d) => setIc(d.ic)).catch(() => {});
    else buildIC(modes, amp).then((d) => setIc(d.ic)).catch(() => {});
  }, [meta, model, modes, amp, pidx]);

  function run() {
    if (wsRef.current) wsRef.current.close();
    setHist([]); setFrame(null); setSum(null); setRunning(true);
    wsRef.current = runCostControl(
      model === "PINN"
        ? { model, pinn_index: pidx, target: sel?.target ?? 0.05 }
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
      </div>

      {err && <div className="text-sm text-rose-600 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 rounded-lg px-3 py-2">{err}</div>}

      {tab === "findings" && cmp && F && D && (
        <>
          <div className="grid grid-cols-3 gap-4">
            <RegimeCard name="FNO" best active={pick === "FNO"} onClick={() => setPick("FNO")}
              a={{ k: "cost", v: `${fLo.toFixed(2)}×` }} b={{ k: "error", v: pct(fE1) }} c={{ k: "hit-rate", v: "100%" }} />
            <RegimeCard name="DeepONet" ok="dominated" active={pick === "DeepONet"} onClick={() => setPick("DeepONet")}
              a={{ k: "cost", v: `${dLo.toFixed(2)}×` }} b={{ k: "error", v: pct(dE1) }} c={{ k: "hit-rate", v: "0%" }} />
            <RegimeCard name="PINN" ok="not amortized" active={pick === "PINN"} onClick={() => setPick("PINN")}
              a={{ k: "cost", v: `${pinnX}×` }} b={{ k: "error", v: "8.0%" }} c={{ k: "hit-rate", v: "80%" }}
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

      {tab === "live" && (
        <>
          <div className="grid grid-cols-4 gap-4">
            <Tile label="time" value={frame ? `t = ${frame.t.toFixed(2)}` : "—"} sub={running ? "running…" : "press run"} />
            <Tile label="trust" value={frame ? frame.trust.toFixed(2) : "—"}
              tone={frame ? (frame.correcting ? "red" : "green") : "slate"}
              sub={frame ? (frame.correcting ? "correcting" : "trusting ML") : "—"} />
            <Tile label="cost so far" value={frame ? `${frame.cost_s.toFixed(2)} s` : "—"}
              tone={model === "PINN" ? "red" : "green"}
              sub={model === "PINN" ? "excludes 2114 s retrain" : (frame ? `${frame.ml_steps} ML · ${frame.corr_steps} numerical` : "—")} />
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
                    <div className="flex justify-between text-[11px] text-slate-500 dark:text-slate-400"><span>modes</span><span>{modes}</span></div>
                    <input type="range" min="1" max="4" value={modes} onChange={(e) => setModes(+e.target.value)} className="w-full" />
                    <div className="flex justify-between text-[11px] text-slate-500 dark:text-slate-400 mt-2"><span>amplitude</span><span>{amp.toFixed(1)}</span></div>
                    <input type="range" min="0.5" max="1.5" step="0.1" value={amp} onChange={(e) => setAmp(+e.target.value)} className="w-full" />
                    {modes > 4 && <div className="text-[11px] text-amber-600 mt-1">above 4 = out-of-distribution</div>}
                  </>
                )}
              </Card>

              <Card title="3 · Accuracy target (Module 3)"
                subtitle="the accuracy you ask for — the controller turns it into when to correct">
                <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400"><span>loose 0.30</span><span>tight 0.01</span></div>
                <input type="range" min="0" max={Math.max(0, fr.length - 1)} value={idx} onChange={(e) => setIdx(+e.target.value)} className="w-full" />
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
              <Card title="Trust vs the deadband" subtitle="red bands = steps where it paid for numerical">
                <TrustTrace hist={hist} lo={lo} />
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
                <div className="text-white grid place-items-center" style={{ width: `${(sum.ml_steps / (sum.ml_steps + sum.corr_steps)) * 100}%`, background: "#059669" }}>
                  ML {Math.round((sum.ml_steps / (sum.ml_steps + sum.corr_steps)) * 100)}%
                </div>
                <div className="text-white grid place-items-center" style={{ width: `${(sum.corr_steps / (sum.ml_steps + sum.corr_steps)) * 100}%`, background: "#e11d48" }}>
                  num {Math.round((sum.corr_steps / (sum.ml_steps + sum.corr_steps)) * 100)}%
                </div>
              </div>
              <div className="text-sm text-slate-700 dark:text-slate-200 mt-3">
                <b>{sum.cost_s.toFixed(2)} s</b> ({sum.rel_cost.toFixed(2)}× the numerical solver) at <b>{pct(sum.error)}</b> error
                {sum.switch_t != null && <> · first correction at t = {sum.switch_t.toFixed(2)}</>}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
