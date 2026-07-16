import { useEffect, useRef, useState } from "react";
import { Card, Stat, Banner } from "../components/ui.jsx";
import { LineChart } from "../components/Charts.jsx";
import { getMeta, buildIC, pinnIC, getCouplingMeta, runCoupling } from "../api.js";
import { CP_XS, CP_FRAMES, CP_STATS as S } from "../couplingData.js";
import {
  Zap, ArrowLeftRight, Anchor, TrendingDown, Clock, Target, ShieldCheck,
  HelpCircle, ArrowRight, Play,
} from "lucide-react";

/* ---- auto-looping three-curve handoff: true / pure ML / hybrid ----
   real committed FNO prediction + the hybrid computed by the VERIFIED restart */
function HandoffChart({ frame, switched }) {
  const W = 580, H = 300, padX = 18, padT = 18, padB = 28;
  const sx = (x) => padX + ((x + 1) / 2) * (W - 2 * padX);
  const sy = (v) => H - padB - ((v + 1.4) / 2.8) * (H - padT - padB);
  const path = (arr) =>
    arr.map((v, i) => `${i ? "L" : "M"}${sx(CP_XS[i]).toFixed(1)} ${sy(v).toFixed(1)}`).join(" ");
  let band = `M${sx(CP_XS[0]).toFixed(1)} ${sy(frame.true[0]).toFixed(1)}`;
  CP_XS.forEach((x, i) => (band += `L${sx(x).toFixed(1)} ${sy(frame.true[i]).toFixed(1)}`));
  for (let i = CP_XS.length - 1; i >= 0; i--)
    band += `L${sx(CP_XS[i]).toFixed(1)} ${sy(frame.ml[i]).toFixed(1)}`;
  band += "Z";
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {[-1, -0.5, 0, 0.5, 1].map((v) => (
        <line key={v} x1={sx(-1)} x2={sx(1)} y1={sy(v)} y2={sy(v)}
          stroke="var(--chart-grid)" strokeWidth="1" />
      ))}
      <path d={band} fill="#fb7185" fillOpacity="0.22" />
      <path d={path(frame.true)} fill="none" stroke="#94a3b8" strokeWidth="2" strokeDasharray="5 4" />
      <path d={path(frame.ml)} fill="none" stroke="#e11d48" strokeWidth="2.5" strokeLinecap="round" />
      <path d={path(frame.hy)} fill="none" stroke="#4f46e5" strokeWidth="3" strokeLinecap="round" />
      <text x={sx(-1) + 4} y={padT + 4} fontSize="11" fill="#fb7185" fontWeight="700">
        shaded = pure ML&apos;s error {switched ? "— the hybrid escaped it" : ""}
      </text>
    </svg>
  );
}

function Metric({ icon: Icon, tone, tag, value, label }) {
  const c = {
    rose: ["bg-rose-50 dark:bg-rose-500/10 border-rose-100 dark:border-rose-500/20",
      "bg-rose-100 dark:bg-rose-500/20 text-rose-600 dark:text-rose-400",
      "text-rose-600 dark:text-rose-400"],
    indigo: ["bg-indigo-50 dark:bg-indigo-500/10 border-indigo-100 dark:border-indigo-500/20",
      "bg-indigo-100 dark:bg-indigo-500/20 text-indigo-600 dark:text-indigo-400",
      "text-indigo-600 dark:text-indigo-400"],
    emerald: ["bg-emerald-50 dark:bg-emerald-500/10 border-emerald-100 dark:border-emerald-500/20",
      "bg-emerald-100 dark:bg-emerald-500/20 text-emerald-600 dark:text-emerald-400",
      "text-emerald-600 dark:text-emerald-400"],
    amber: ["bg-amber-50 dark:bg-amber-500/10 border-amber-100 dark:border-amber-500/20",
      "bg-amber-100 dark:bg-amber-500/20 text-amber-600 dark:text-amber-400",
      "text-amber-600 dark:text-amber-400"],
  }[tone];
  return (
    <div className={`rounded-2xl border p-4 hover:-translate-y-1 transition-transform ${c[0]}`}>
      <div className="flex items-center justify-between">
        <div className={`w-9 h-9 rounded-xl grid place-items-center ${c[1]}`}><Icon size={18} /></div>
        {tag && <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">{tag}</span>}
      </div>
      <div className={`text-3xl font-extrabold mt-3 ${c[2]}`}>{value}</div>
      <div className="text-xs text-slate-600 dark:text-slate-300 mt-1 leading-snug">{label}</div>
    </div>
  );
}

function Step({ n, icon: Icon, tone, title, sub }) {
  const c = { indigo: "bg-indigo-600", violet: "bg-violet-600", fuchsia: "bg-fuchsia-600" }[tone];
  return (
    <div className="flex-1 min-w-[150px] rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50/60 dark:bg-slate-700/30 p-3 flex items-center gap-3">
      <div className={`relative w-10 h-10 rounded-xl grid place-items-center text-white shrink-0 ${c}`}>
        <Icon size={18} />
        <span className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-white dark:bg-slate-800 text-[11px] font-bold grid place-items-center text-slate-700 dark:text-slate-200 border border-slate-200 dark:border-slate-600">{n}</span>
      </div>
      <div>
        <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">{title}</div>
        <div className="text-[11px] text-slate-500 dark:text-slate-400 leading-tight">{sub}</div>
      </div>
    </div>
  );
}

const MODELS = ["FNO", "DeepONet", "PINN"];
const inactiveBtn = "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700";

export default function CouplingPage() {
  /* ---- auto-loop ---- */
  const [i, setI] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setI((k) => (k + 1) % CP_FRAMES.length), 120);
    return () => clearInterval(id);
  }, []);
  const frame = CP_FRAMES[i];
  const switched = frame.t >= S.switchT;
  const upto = CP_FRAMES.slice(0, i + 1);

  /* ---- live run state ---- */
  const [meta, setMeta] = useState(null);
  const [cmeta, setCmeta] = useState(null);
  const [model, setModel] = useState("FNO");
  const [modes, setModes] = useState(4);
  const [amplitude, setAmplitude] = useState(1.0);
  const [pinnIndex, setPinnIndex] = useState(0);
  const [switchMode, setSwitchMode] = useState("manual");
  const [ts, setTs] = useState(1.0);
  const [ic, setIc] = useState(null);
  const [lf, setLf] = useState(null);
  const [hist, setHist] = useState([]);
  const [summary, setSummary] = useState(null);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => {
    getMeta().then(setMeta).catch(() => {});
    getCouplingMeta().then(setCmeta).catch(() => {});
  }, []);
  useEffect(() => {
    if (!meta) return;
    if (model === "PINN") pinnIC(pinnIndex).then((d) => setIc(d.ic)).catch(() => {});
    else buildIC(modes, amplitude).then((d) => setIc(d.ic)).catch(() => {});
  }, [meta, model, modes, amplitude, pinnIndex]);

  function run() {
    if (wsRef.current) wsRef.current.close();
    setHist([]); setLf(null); setSummary(null); setErr(null); setRunning(true);
    const payload = model === "PINN"
      ? { model, pinn_index: pinnIndex, switch_mode: switchMode, t_s: ts }
      : { model, ic, switch_mode: switchMode, t_s: ts };
    wsRef.current = runCoupling(payload,
      (f) => { setLf(f); setHist((h) => [...h, f]); },
      (s) => setSummary(s),
      () => setRunning(false),
      (e) => { setErr(e); setRunning(false); });
  }

  const x = meta?.x || [];
  const muted = "text-slate-500 dark:text-slate-400";
  const boundaryTs = cmeta?.viability?.boundary_t_s ?? S.boundary;
  const lyr = lf ? [Math.min(...lf.true, ...lf.hybrid, -1.1), Math.max(...lf.true, ...lf.hybrid, 1.1)] : [-1.1, 1.1];

  return (
    <div className="space-y-7">
      {/* HEADER */}
      <div>
        <span className="text-xs font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">Hybrid components · Module 2</span>
        <h1 className="text-3xl font-extrabold text-slate-800 dark:text-slate-100 mt-1">Coupling — the verified handoff</h1>
        <p className="text-slate-500 dark:text-slate-400 mt-1 max-w-2xl">
          The ML model is fast but fails in the future; the numerical solver never fails but is slow. So —{" "}
          <span className="font-medium text-slate-700 dark:text-slate-200">can we hand the wave from one to the other, mid-flight, without breaking it?</span>{" "}
          Here is the mechanism, the proof, and its limits.
        </p>
      </div>

      {/* HOW IT WORKS */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
        <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-4">
          The handoff — three steps
        </div>
        <div className="flex items-stretch gap-2 flex-wrap">
          <Step n="1" icon={Zap} tone="indigo" title="ML races ahead" sub="the FNO predicts the whole trajectory in milliseconds" />
          <ArrowRight size={18} className="self-center shrink-0 text-slate-300 dark:text-slate-600" />
          <Step n="2" icon={ArrowLeftRight} tone="violet" title="Hand over the state" sub="at the switch, the ML field becomes the solver's initial condition — zero jump" />
          <ArrowRight size={18} className="self-center shrink-0 text-slate-300 dark:text-slate-600" />
          <Step n="3" icon={Anchor} tone="fuchsia" title="Re-anchor & carry home" sub="the verified numerical solver continues to the horizon" />
        </div>

        <div className="mt-5 rounded-xl bg-slate-50 dark:bg-slate-700/40 p-4">
          <div className="text-center font-mono text-[15px] text-slate-700 dark:text-slate-200 leading-relaxed">
            u_hybrid(t) = <span className="text-rose-600 dark:text-rose-400 font-semibold">u_ML(t)</span> for t &lt; t_s ·{" "}
            <span className="text-indigo-600 dark:text-indigo-400 font-semibold">u_num(t; seeded from u_ML(t_s))</span> for t ≥ t_s
          </div>
          <div className="flex items-center justify-center gap-6 mt-3 text-xs">
            <span className="flex items-center gap-1.5 text-slate-500 dark:text-slate-400">
              <span className="w-2.5 h-2.5 rounded bg-rose-500" />
              <span className="font-semibold text-rose-600 dark:text-rose-400">fast</span> — while trusted
            </span>
            <span className="flex items-center gap-1.5 text-slate-500 dark:text-slate-400">
              <span className="w-2.5 h-2.5 rounded bg-indigo-500" />
              <span className="font-semibold text-indigo-600 dark:text-indigo-400">exact restart</span> — proven bit-for-bit = the production solver (rel diff 0.0)
            </span>
          </div>
        </div>
      </div>

      {/* HERO ANIMATION */}
      <div className="grid grid-cols-[1fr_230px] gap-5 items-stretch">
        <div className="relative rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4 overflow-hidden">
          <div className="absolute -top-8 -right-8 w-40 h-40 rounded-full bg-indigo-100/50 dark:bg-indigo-500/10 blur-2xl" />
          <div className="relative">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">Watch the handoff rescue the trajectory</div>
                <div className="text-xs text-slate-500 dark:text-slate-400">
                  real held-out wave · committed FNO prediction · hybrid computed by the verified restart
                </div>
              </div>
              <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-1 rounded-full ${switched
                ? "bg-indigo-100 dark:bg-indigo-500/20 text-indigo-600 dark:text-indigo-400 animate-pulse"
                : "bg-rose-100 dark:bg-rose-500/20 text-rose-600 dark:text-rose-400"}`}>
                {switched ? "re-anchored — numerical" : "ML rolling"}
              </span>
            </div>
            <HandoffChart frame={frame} switched={switched} />
            <div className="flex items-center justify-center gap-5 text-[11px] text-slate-400 dark:text-slate-500">
              <span className="flex items-center gap-1"><span className="w-3 h-1 rounded-full bg-slate-400" /> true</span>
              <span className="flex items-center gap-1"><span className="w-3 h-1 rounded-full bg-rose-500" /> pure ML (keeps drifting)</span>
              <span className="flex items-center gap-1"><span className="w-3 h-1 rounded-full bg-indigo-500" /> hybrid (handed off at t_s = {S.switchT.toFixed(2)})</span>
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4 text-center">
            <div className="text-xs text-slate-500 dark:text-slate-400">time</div>
            <div className="text-2xl font-bold text-slate-800 dark:text-slate-100">t = {frame.t.toFixed(2)}</div>
          </div>
          <div className="rounded-2xl border border-rose-200 dark:border-rose-500/30 bg-rose-50 dark:bg-rose-500/10 p-4 text-center">
            <div className="text-xs text-rose-500 dark:text-rose-400">pure ML error</div>
            <div className="text-4xl font-extrabold text-rose-600 dark:text-rose-400 mt-1">{(frame.eml * 100).toFixed(0)}%</div>
            <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">keeps growing</div>
          </div>
          <div className="flex-1 rounded-2xl border border-indigo-200 dark:border-indigo-500/30 bg-indigo-50 dark:bg-indigo-500/10 p-4 text-center grid place-items-center">
            <div>
              <div className="text-xs text-indigo-500 dark:text-indigo-400">hybrid error</div>
              <div className="text-4xl font-extrabold text-indigo-600 dark:text-indigo-400 mt-1">{(frame.ehy * 100).toFixed(1)}%</div>
              <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">{switched ? "pinned by the re-anchor · jump = 0" : "same as ML until the switch"}</div>
            </div>
          </div>
        </div>
      </div>

      {/* PROGRESSIVE ERROR CURVES */}
      <Card title="Error over time, drawn live"
        subtitle="identical until the switch — then the hybrid is pinned back while pure ML keeps drifting">
        <LineChart
          series={[
            { x: upto.map((f) => f.t), y: upto.map((f) => f.eml), color: "#e11d48", width: 2 },
            { x: upto.map((f) => f.t), y: upto.map((f) => f.ehy), color: "#4f46e5", width: 2.5 },
          ]}
          xr={[0, 2]} yr={[0, Math.max(...CP_FRAMES.map((f) => f.eml)) * 1.08]}
          vline={S.switchT} h={185} xlabel="t" ylabel="relative L2 error" />
      </Card>

      {/* FINDINGS */}
      <div>
        <h2 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3 flex items-center gap-2">
          <HelpCircle size={15} /> The evidence · measured, not asserted
        </h2>
        <div className="grid grid-cols-4 gap-4">
          <Metric icon={TrendingDown} tone="emerald" tag="main result" value={`${S.benefit}%`}
            label={`error cut: ${S.fnoTail}% → ${S.hybTail}% on all 20 held-out waves (switch at t = 1)`} />
          <Metric icon={Clock} tone="amber" tag="the limit" value={`t ≈ ${S.boundary}`}
            label={`switch later than this and the handoff stops helping — it equals the FNO's own reliable horizon (${S.horizon})`} />
          <Metric icon={Target} tone="indigo" tag="the law" value={`~${S.oracle}`}
            label="restarting from the TRUE state is near-perfect — all hybrid error is inherited from the ML state, none from my continuation" />
          <Metric icon={ShieldCheck} tone="rose" tag="fidelity" value={`Re ≈ ${S.reCell}`}
            label="beyond this sharpness an approximate restart fails or blows up — mine is exact, so it doesn't" />
        </div>
      </div>

      {/* VERDICT */}
      <div className="rounded-2xl p-5 bg-gradient-to-r from-indigo-50 via-white to-white dark:from-indigo-500/10 dark:via-slate-800 dark:to-slate-800 border border-indigo-200 dark:border-indigo-500/30">
        <div className="text-sm font-semibold text-indigo-800 dark:text-indigo-300">Answer</div>
        <p className="text-sm text-slate-700 dark:text-slate-200 mt-1">
          Yes — the handoff is <span className="font-medium">continuous (jump = 0), verified (bit-for-bit the
          production solver), and works for any of the three models</span> ({S.transfer.FNO}% / {S.transfer.PINN}% / {S.transfer.DeepONet}% benefit for FNO / PINN / DeepONet).
          Its accuracy is bounded by the ML state it inherits — which is exactly why{" "}
          <span className="font-semibold">Module 1 decides when</span> to fire it and{" "}
          <span className="font-semibold">Module 3 decides how much</span> it may cost.
        </p>
        <p className="text-xs text-slate-400 dark:text-slate-500 mt-2">
          This page dissects the mechanism — you control the switch and may deliberately switch too late.
          The Hybrid engine page is the opposite: you set an accuracy target and the system decides.
        </p>
      </div>

      {/* RUN IT YOURSELF */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
        <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-1 flex items-center gap-2">
          <Play size={14} /> Run it yourself — live, with the real M2Coupling
        </div>
        <p className={`text-xs ${muted} mb-4`}>
          Pick a model and a wave, choose the switch — manual (drag past {boundaryTs.toFixed(2)} and watch the
          benefit die) or Module 1&apos;s trust signal — and the backend runs the same adapter Module 3 calls.
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
                <label className={`block ${muted}`}>
                  modes: {modes}
                  <input type="range" min="1" max="8" value={modes} onChange={(e) => setModes(+e.target.value)} className="w-full" />
                </label>
                <label className={`block ${muted}`}>
                  amplitude: {amplitude.toFixed(2)}
                  <input type="range" min="0.2" max="1.5" step="0.05" value={amplitude} onChange={(e) => setAmplitude(+e.target.value)} className="w-full" />
                </label>
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
                switch time t_s = {ts.toFixed(2)}
                {ts > boundaryTs && <span className="ml-2 text-amber-600 dark:text-amber-400">past the viability boundary</span>}
                <input type="range" min="0.5" max="1.9" step="0.05" value={ts} onChange={(e) => setTs(+e.target.value)} className="w-full" />
              </label>
            )}
            <button onClick={run} disabled={running || !ic}
              className="w-full px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium text-sm transition">
              {running ? "Running…" : "Run the hybrid"}
            </button>
            {summary && (
              <div className="space-y-2">
                <div className="grid grid-cols-2 gap-2">
                  <Stat label="pure-ML error [1,2]" value={`${(summary.ml_tail_1_2 * 100).toFixed(1)}%`} tone="red" />
                  <Stat label="hybrid error [1,2]" value={`${(summary.hybrid_tail_1_2 * 100).toFixed(1)}%`} tone="green" />
                  <Stat label="benefit" value={summary.benefit != null ? `${(summary.benefit * 100).toFixed(0)}%` : "—"} tone="indigo" />
                  <Stat label="numerical work" value={`${(summary.numerical_fraction * 100).toFixed(0)}%`} />
                </div>
                <Banner ok={summary.handoff_jump === 0}
                  text={`handoff jump = ${summary.handoff_jump} — continuous by construction`} />
              </div>
            )}
          </div>
          <div className="space-y-3">
            <Card title={lf ? `t = ${lf.t.toFixed(2)}${lf.switched ? " — numerical continuation active" : " — ML rolling"}` : "run to start"}>
              <LineChart
                series={[
                  { x, y: lf ? lf.true : [], color: "#94a3b8", dashed: true },
                  { x, y: lf ? lf.ml : [], color: "#e11d48" },
                  { x, y: lf ? lf.hybrid : [], color: "#4f46e5", width: 2.5 },
                ]}
                xr={[-1, 1]} yr={lyr} h={190} xlabel="x" ylabel="u(x, t)" />
            </Card>
            <div className="grid grid-cols-2 gap-3">
              <Card title="error over time">
                <LineChart
                  series={[
                    { x: hist.map((f) => f.t), y: hist.map((f) => f.ml_err), color: "#e11d48" },
                    { x: hist.map((f) => f.t), y: hist.map((f) => f.hybrid_err), color: "#4f46e5", width: 2.5 },
                  ]}
                  xr={[0, 2]} yr={[0, Math.max(0.3, ...hist.map((f) => f.ml_err))]}
                  vline={lf?.switch_t ?? null} h={150} xlabel="t" />
              </Card>
              <Card title="when does the handoff help?">
                <LineChart
                  series={[{ x: S.sweep.map((r) => r.ts), y: S.sweep.map((r) => r.benefit), color: "#4f46e5", width: 2.5 }]}
                  xr={[1.0, 1.8]} yr={[0, 1]} vline={boundaryTs} h={150}
                  xlabel="switch time t_s" ylabel="benefit" />
              </Card>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
