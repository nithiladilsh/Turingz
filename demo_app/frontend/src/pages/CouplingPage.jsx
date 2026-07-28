import { useEffect, useRef, useState } from "react";
import { Card, Stat, Banner } from "../components/ui.jsx";
import { LineChart } from "../components/Charts.jsx";
import { getMeta, buildIC, pinnIC, runCoupling } from "../api.js";
import {
  RL_XS, RL_T, RL_FRAMES, RL_ERR_ML, RL_SWITCHES, RL_META,
} from "../couplingData.js";
import { Play, GitCommitHorizontal, Microscope, Code2 } from "lucide-react";

/* =====================================================================
   Module 2 · THE BATON PASS — a relay between two solvers.
   The page is built around one interaction: YOU drag the handoff point.
   ===================================================================== */

const MODELS = ["FNO", "DeepONet", "PINN"];
const inactiveBtn = "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700";

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
    agg: ["AGGREGATE n = 20", "bg-indigo-100 dark:bg-indigo-500/20 text-indigo-700 dark:text-indigo-300"],
  }[kind];
  return <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full ${c[1]}`}>{c[0]}</span>;
}

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
  const [modes, setModes] = useState(4);
  const [amplitude, setAmplitude] = useState(1.0);
  const [pinnIndex, setPinnIndex] = useState(0);
  const [switchMode, setSwitchMode] = useState("manual");
  const [lts, setLts] = useState(1.0);
  const [ic, setIc] = useState(null);
  const [lf, setLf] = useState(null);
  const [hist, setHist] = useState([]);
  const [summary, setSummary] = useState(null);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => { getMeta().then(setMeta).catch(() => {}); }, []);
  useEffect(() => {
    if (!meta) return;
    if (model === "PINN") pinnIC(pinnIndex).then((d) => setIc(d.ic)).catch(() => {});
    else buildIC(modes, amplitude).then((d) => setIc(d.ic)).catch(() => {});
  }, [meta, model, modes, amplitude, pinnIndex]);

  function run() {
    if (wsRef.current) wsRef.current.close();
    setHist([]); setLf(null); setSummary(null); setErr(null); setRunning(true);
    const payload = model === "PINN"
      ? { model, pinn_index: pinnIndex, switch_mode: switchMode, t_s: lts }
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
      {/* HEADER — its own identity: the relay */}
      <div className="rounded-2xl p-6 bg-gradient-to-r from-rose-50 via-white to-indigo-50 dark:from-rose-500/10 dark:via-slate-800 dark:to-indigo-500/10 border border-slate-200 dark:border-slate-700">
        <span className="text-xs font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">
          Hybrid components · Module 2
        </span>
        <h1 className="text-3xl font-extrabold mt-1">
          <span className="text-rose-600 dark:text-rose-400">Coupling</span>
          <span className="text-slate-400 dark:text-slate-500 mx-2">—</span>
          <span className="text-slate-800 dark:text-slate-100">the baton pass</span>
        </h1>
        <p className="text-slate-500 dark:text-slate-400 mt-1 max-w-3xl text-sm">
          One trajectory, two runners. The <span className="font-semibold text-rose-600 dark:text-rose-400">fast ML model</span> carries
          the wave while it can be trusted; my verified handoff passes it — mid-flight, zero jump — to the{" "}
          <span className="font-semibold text-indigo-600 dark:text-indigo-400">numerical solver</span> that carries it home.{" "}
          <span className="font-medium text-slate-700 dark:text-slate-200">You hold the baton: drag the handoff and watch what one decision does.</span>
        </p>
      </div>

      {/* TABS */}
      <div className="flex gap-2">
        {[["story", "Story"], ["evidence", "Evidence"], ["live", "Try it live"]].map(([v, label]) => (
          <button key={v} onClick={() => setTab(v)}
            className={`px-4 py-2 rounded-xl text-sm font-medium border transition ${tab === v
              ? "bg-indigo-600 text-white border-indigo-600"
              : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700"}`}>
            {label}
          </button>
        ))}
      </div>

      {tab === "story" && (<div className="space-y-6">
        <div className="flex items-center gap-2"><Badge kind="rep" /><span className="text-xs text-slate-400 dark:text-slate-500">one held-out wave, interactive — aggregate numbers are on the Evidence tab</span></div>
      {/* ===== THE RELAY TIMELINE — the page's centrepiece ===== */}
      <div className="rounded-2xl border-2 border-indigo-200 dark:border-indigo-500/30 bg-white dark:bg-slate-800 p-5">
        <div className="flex items-center justify-between mb-3">
          <div className="text-sm font-semibold text-slate-800 dark:text-slate-100 flex items-center gap-2">
            <GitCommitHorizontal size={16} className="text-indigo-500" />
            Who carries the wave — drag the handoff t_s = {sw.ts.toFixed(1)}
          </div>
          <span className={`text-[11px] font-bold uppercase tracking-wider px-2.5 py-1 rounded-full ${vChip}`}>
            {verdict}{verdict === "exceeds 10% error criterion" && " — still improves, but the state was already too degraded"}
          </span>
        </div>

        {/* ownership bar with playhead baton */}
        <div className="relative h-9 rounded-lg overflow-hidden flex text-[11px] font-bold text-white select-none">
          <div className="bg-rose-500/90 grid place-items-center transition-all duration-300"
            style={{ width: `${sw.i0f * 100}%` }}>ML — fast</div>
          <div className="bg-indigo-600 grid place-items-center flex-1 transition-all duration-300">
            numerical — verified restart
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

        {/* consequences of the chosen handoff — updates instantly */}
        <div className="grid grid-cols-4 gap-3 mt-3">
          <div className="rounded-xl bg-rose-50 dark:bg-rose-500/10 border border-rose-100 dark:border-rose-500/20 px-3 py-2 text-center">
            <div className="text-[10px] uppercase tracking-wide text-rose-500">pure-ML error · never hand off</div>
            <div className="text-xl font-extrabold text-rose-600 dark:text-rose-400">{(sw.mlTail * 100).toFixed(1)}%</div>
          </div>
          <div className="rounded-xl bg-indigo-50 dark:bg-indigo-500/10 border border-indigo-100 dark:border-indigo-500/20 px-3 py-2 text-center">
            <div className="text-[10px] uppercase tracking-wide text-indigo-500">hybrid error · hand off here</div>
            <div className="text-xl font-extrabold text-indigo-600 dark:text-indigo-400">{(sw.hyTail * 100).toFixed(1)}%</div>
          </div>
          <div className="rounded-xl bg-slate-50 dark:bg-slate-700/40 border border-slate-100 dark:border-slate-600/40 px-3 py-2 text-center">
            <div className="text-[10px] uppercase tracking-wide text-slate-400">numerical work · the cost</div>
            <div className="text-xl font-extrabold text-slate-700 dark:text-slate-200">{(sw.work * 100).toFixed(0)}%</div>
          </div>
          <div className="rounded-xl bg-emerald-50 dark:bg-emerald-500/10 border border-emerald-100 dark:border-emerald-500/20 px-3 py-2 text-center">
            <div className="text-[10px] uppercase tracking-wide text-emerald-600">jump at handoff</div>
            <div className="text-xl font-extrabold text-emerald-600 dark:text-emerald-400">{sw.jump.toExponential(0)}</div>
          </div>
          <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-2">
            Later hand-offs are <span className="font-medium">cheaper</span> (less numerical work) but{" "}
            <span className="font-medium">less accurate</span> — the state handed over is already degraded.
            The criterion above is about <span className="font-medium">accuracy</span>, not cost: how much that
            accuracy is worth paying for is Module 3&apos;s decision.
          </p>
        </div>
      </div>

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
            <span>— true (dashed)</span>
            <span className="text-rose-500">— pure ML{switched ? " (ghost — what would have happened)" : ""}</span>
            {switched && <span className="text-indigo-500 font-medium">— hybrid</span>}
          </div>
        </div>
        <Card title="The cost of your decision"
          subtitle="red = never hand off · indigo = your relay — identical until t_s, then pinned">
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
      {/* AGGREGATE RESULT — the report headline, kept distinct from the single-wave animation */}
      <div className="rounded-2xl border-2 border-indigo-200 dark:border-indigo-500/30 bg-white dark:bg-slate-800 p-5">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-sm font-semibold text-slate-800 dark:text-slate-100">Primary result — hand-off at t_s = 1.0</span>
          <Badge kind="agg" /><Badge kind="committed" />
        </div>
        <p className="text-xs text-slate-500 dark:text-slate-400 mb-3">
          Mean over 20 held-out waves (the figure above animates one representative wave, so its numbers differ slightly).
        </p>
        <div className="grid grid-cols-4 gap-3">
          <div className="rounded-xl bg-rose-50 dark:bg-rose-500/10 border border-rose-100 dark:border-rose-500/20 px-3 py-2 text-center">
            <div className="text-[10px] uppercase tracking-wide text-rose-500">pure ML tail error</div>
            <div className="text-2xl font-extrabold text-rose-600 dark:text-rose-400">13.44%</div>
          </div>
          <div className="rounded-xl bg-indigo-50 dark:bg-indigo-500/10 border border-indigo-100 dark:border-indigo-500/20 px-3 py-2 text-center">
            <div className="text-[10px] uppercase tracking-wide text-indigo-500">hybrid tail error</div>
            <div className="text-2xl font-extrabold text-indigo-600 dark:text-indigo-400">0.98%</div>
          </div>
          <div className="rounded-xl bg-emerald-50 dark:bg-emerald-500/10 border border-emerald-100 dark:border-emerald-500/20 px-3 py-2 text-center">
            <div className="text-[10px] uppercase tracking-wide text-emerald-600">error reduction</div>
            <div className="text-2xl font-extrabold text-emerald-600 dark:text-emerald-400">91.7%</div>
          </div>
          <div className="rounded-xl bg-slate-50 dark:bg-slate-700/40 border border-slate-100 dark:border-slate-600/40 px-3 py-2 text-center">
            <div className="text-[10px] uppercase tracking-wide text-slate-400">waves improved</div>
            <div className="text-2xl font-extrabold text-slate-700 dark:text-slate-200">20 / 20</div>
          </div>
        </div>
      </div>

      {/* THE LAW — the equality that names the module's finding */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
        <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3 flex items-center gap-2">
          <Microscope size={14} /> The law this page demonstrates — hybrid error ≈ state error at handoff
        </div>
        <div className="flex items-center justify-center gap-6 flex-wrap">
          <div className="text-center">
            <div className="text-[11px] text-slate-500 dark:text-slate-400">ML state error when you handed off</div>
            <div className="text-4xl font-extrabold text-rose-600 dark:text-rose-400">{(sw.es * 100).toFixed(1)}%</div>
          </div>
          <div className="text-4xl font-black text-slate-300 dark:text-slate-600">≈</div>
          <div className="text-center">
            <div className="text-[11px] text-slate-500 dark:text-slate-400">mean hybrid tail error over [t_s, 2]</div>
            <div className="text-4xl font-extrabold text-indigo-600 dark:text-indigo-400">{(sw.hyTail * 100).toFixed(1)}%</div>
          </div>
          <div className="max-w-xs text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
            Drag the slider — the two numbers move <span className="font-semibold">together</span>. The continuation adds
            ~{RL_META.oracle} of its own error (oracle control), so the handoff-state error <span className="font-semibold">dominates</span> the resulting
            hybrid error. You can only anchor what you hand over — so the hand-off time{" "}
            <span className="font-medium text-slate-700 dark:text-slate-200">sets the accuracy ceiling for the
            whole system</span>, and the sweep below is what tells the trust and control layers where that
            ceiling is.
          </div>
        </div>
      </div>

      {/* NOVELTY IN CODE — mirrors the Cost page's card, with M2's receipts */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
        <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-2 flex items-center gap-2">
          <Code2 size={14} /> Why this is trustworthy (in code)
        </div>
        <p className="text-sm text-slate-700 dark:text-slate-200">
          The production solver only knew <span className="font-mono text-[13px]">solve(u0)</span> from t = 0. I made it restartable:
        </p>
        <div className="mt-2 rounded-lg bg-slate-50 dark:bg-slate-700/40 px-3 py-2 font-mono text-[13px] text-slate-700 dark:text-slate-200">
          solve_from(u_ML(t_s), i_start) → verified <span className="text-emerald-600 dark:text-emerald-400 font-semibold">bit-for-bit identical</span> to the team solver (rel diff 0.0)
        </div>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">
          Guarded by 12 automated tests — including a regression check against an oracle-contaminated alternative — and a stress test showing a
          restart that drops one “cosmetic” safety step fails or blows up beyond Re_cell ≈ {RL_META.reCell} while this one holds.
          The exact adapter this page calls is the one Module 3&apos;s runtime executes.
        </p>
      </div>

      </div>)}
      {tab === "live" && (<div className="space-y-6">
        <div className="flex items-center gap-2"><Badge kind="live" /><span className="text-xs text-slate-400 dark:text-slate-500">runs the real M2Coupling adapter with the verified pseudo-spectral restart</span></div>
      {/* RUN IT YOURSELF — real backend */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-5">
        <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-1 flex items-center gap-2">
          <Play size={14} /> Run it yourself — your wave, the real M2Coupling
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
                  <Stat label="benefit" value={summary.benefit != null ? `${(summary.benefit * 100).toFixed(0)}%` : "—"} tone="indigo" />
                  <Stat label="numerical work" value={`${(summary.numerical_fraction * 100).toFixed(0)}%`} />
                </div>
                <Banner ok={summary.handoff_jump === 0}
                  text={`handoff jump = ${summary.handoff_jump} — continuous by construction`} />
              </div>
            )}
          </div>
          <div className="space-y-3">
            <Card title={lf ? `t = ${lf.t.toFixed(2)}${lf.switched ? " — numerical carries it" : " — ML carries it"}` : "run to start"}>
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
        This page dissects the handoff — you control the switch and may deliberately hand off outside the viability criterion.
        The Hybrid engine page is the opposite: you set an accuracy target and the trust + control layer decides for you.
      </p>
      </div>)}
    </div>
  );
}
