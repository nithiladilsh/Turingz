import { useEffect, useRef, useState } from "react";
import { Card } from "../components/ui.jsx";
import { LineChart } from "../components/Charts.jsx";
import { REL } from "../reliabilityData.js";
import { getMeta, buildIC, pinnIC, runReliability } from "../api.js";
import { Play, Trophy, Timer, Activity, Shuffle, ArrowRight } from "lucide-react";

const MODELS = ["FNO", "PINN", "DeepONet"];
const COLOR = { FNO: "#059669", PINN: "#4f46e5", DeepONet: "#e11d48" };
const GEN = {
  FNO: { seen: 12.7, unseen: 14.1, ok: true },
  DeepONet: { seen: 68.5, unseen: 61.1, ok: false },
};

// what each model actually receives as input (shown in the explorer)
const INPUTS = {
  FNO: {
    kind: "wave",
    takes: "a starting wave u₀(x) + the (x, t) grid",
    note: "Encodes u₀ as a channel and returns the whole u(x, t) field in one pass. Build any wave below.",
  },
  DeepONet: {
    kind: "wave",
    takes: "a starting wave u₀(x) (branch) + query points (x, t) (trunk)",
    note: "The branch net reads the whole wave, the trunk net reads each location. Build any wave below.",
  },
  PINN: {
    kind: "index",
    takes: "one pre-trained wave (index) + coordinates (x, t)",
    note: "A PINN is fit to a single initial condition, so it can't take a new wave — pick one it was trained on.",
  },
};

/* animated error-over-time chart for the 3 models, with highlight */
function ErrorChart({ highlight, h = 300 }) {
  const [p, setP] = useState(0);
  useEffect(() => {
    let raf, start;
    const loop = (ts) => {
      if (!start) start = ts;
      const q = Math.min(1, (ts - start) / 1600);
      setP(q);
      if (q < 1) raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);
  const W = 640,
    H = h,
    padL = 40,
    padR = 16,
    padT = 16,
    padB = 30;
  const sx = (t) => padL + (t / 2) * (W - padL - padR);
  const sy = (e) => H - padB - Math.min(1, e / 0.7) * (H - padT - padB);
  const n = Math.max(2, Math.floor(p * REL.t.length));
  const line = (arr) =>
    arr
      .slice(0, n)
      .map(
        (v, i) =>
          `${i ? "L" : "M"}${sx(REL.t[i]).toFixed(1)} ${sy(v).toFixed(1)}`,
      )
      .join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {[0.1, 0.3, 0.5, 0.7].map((g) => (
        <line
          key={g}
          x1={padL}
          x2={W - padR}
          y1={sy(g)}
          y2={sy(g)}
          stroke="var(--chart-grid)"
        />
      ))}
      {[0.2, 0.4, 0.6].map((g) => (
        <text
          key={g}
          x={padL - 6}
          y={sy(g) + 3}
          textAnchor="end"
          fontSize="9"
          fill="var(--chart-axis)"
        >
          {Math.round(g * 100)}%
        </text>
      ))}
      {/* training window */}
      <rect
        x={sx(0)}
        y={padT}
        width={sx(1) - sx(0)}
        height={H - padT - padB}
        fill="#6366f1"
        fillOpacity="0.05"
      />
      <line
        x1={sx(1)}
        x2={sx(1)}
        y1={padT}
        y2={H - padB}
        stroke="#818cf8"
        strokeDasharray="4 3"
      />
      <text
        x={sx(0.5)}
        y={padT + 12}
        textAnchor="middle"
        fontSize="9"
        fill="#818cf8"
        fontWeight="600"
      >
        trained (t ≤ 1)
      </text>
      <text
        x={sx(1.5)}
        y={padT + 12}
        textAnchor="middle"
        fontSize="9"
        fill="#94a3b8"
      >
        extrapolation (t &gt; 1)
      </text>
      {/* fail line */}
      <line
        x1={padL}
        x2={W - padR}
        y1={sy(0.1)}
        y2={sy(0.1)}
        stroke="#f59e0b"
        strokeWidth="1"
        strokeDasharray="3 3"
      />
      <text
        x={W - padR}
        y={sy(0.1) - 4}
        textAnchor="end"
        fontSize="9"
        fill="#d97706"
        fontWeight="600"
      >
        10% fail line
      </text>
      {MODELS.map((m) => {
        const dim = highlight && highlight !== m;
        return (
          <path
            key={m}
            d={line(REL.curves[m])}
            fill="none"
            stroke={COLOR[m]}
            strokeWidth={highlight === m ? 3.5 : 2.5}
            strokeOpacity={dim ? 0.18 : 1}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        );
      })}
      {/* legend (top-left, empty region of the plot) */}
      {MODELS.map((m, i) => {
        const lx = padL + 12,
          ly = padT + 14 + i * 18;
        const dim = highlight && highlight !== m;
        return (
          <g key={m} opacity={dim ? 0.3 : 1}>
            <line
              x1={lx}
              x2={lx + 18}
              y1={ly}
              y2={ly}
              stroke={COLOR[m]}
              strokeWidth="3.5"
              strokeLinecap="round"
            />
            <text
              x={lx + 24}
              y={ly + 4}
              fontSize="11"
              fill={COLOR[m]}
              fontWeight="700"
            >
              {m}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function ScoreCard({ m, best, active, onClick }) {
  const s = REL.summary[m];
  return (
    <button
      onClick={onClick}
      className={`text-left rounded-2xl border p-4 transition-all hover:-translate-y-1 ${
        active ? "border-2 shadow-md" : "border-slate-200 dark:border-slate-700"
      } bg-white dark:bg-slate-800`}
      style={active ? { borderColor: COLOR[m] } : {}}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className="w-3 h-3 rounded-full"
            style={{ background: COLOR[m] }}
          />
          <span className="text-sm font-bold text-slate-800 dark:text-slate-100">
            {m}
          </span>
        </div>
        {best && (
          <span className="inline-flex items-center gap-1 text-[10px] font-bold text-emerald-700 dark:text-emerald-300 bg-emerald-100 dark:bg-emerald-500/20 px-2 py-0.5 rounded-full">
            <Trophy size={11} /> best
          </span>
        )}
      </div>
      <div className="grid grid-cols-3 gap-2 mt-3">
        <div>
          <div className="text-[10px] text-slate-400 dark:text-slate-500">
            in-window
          </div>
          <div className="text-lg font-bold text-slate-700 dark:text-slate-200">
            {Math.round(s.inWindow * 100)}%
          </div>
        </div>
        <div>
          <div className="text-[10px] text-slate-400 dark:text-slate-500">
            extrapolation
          </div>
          <div className="text-lg font-bold" style={{ color: COLOR[m] }}>
            {Math.round(s.extrap * 100)}%
          </div>
        </div>
        <div>
          <div className="text-[10px] text-slate-400 dark:text-slate-500">
            horizon
          </div>
          <div className="text-lg font-bold text-slate-700 dark:text-slate-200">
            {s.horizon.toFixed(2)}
          </div>
        </div>
      </div>
    </button>
  );
}

function HorizonRunway({ m }) {
  const h = REL.summary[m].horizon;
  return (
    <div className="flex items-center gap-3">
      <div className="w-16 text-xs font-semibold" style={{ color: COLOR[m] }}>
        {m}
      </div>
      <div className="flex-1 h-5 rounded-full bg-slate-100 dark:bg-slate-700 relative overflow-hidden">
        <div
          className="h-5 rounded-full transition-all duration-700"
          style={{ width: `${(h / 2) * 100}%`, background: COLOR[m] }}
        />
        <div className="absolute inset-y-0 left-1/2 w-px bg-slate-300 dark:bg-slate-600" />
      </div>
      <div className="w-20 text-right text-xs text-slate-600 dark:text-slate-300">
        trusted to <b>{h.toFixed(2)}</b>
      </div>
    </div>
  );
}

export default function ReliabilityPage() {
  const [tab, setTab] = useState("findings");
  const [highlight, setHighlight] = useState(null);
  // interactive explorer state
  const [meta, setMeta] = useState(null);
  const [model, setModel] = useState("FNO");
  const [modes, setModes] = useState(4);
  const [amplitude, setAmplitude] = useState(1.0);
  const [seed, setSeed] = useState(1);
  const [pinnIndex, setPinnIndex] = useState(0);
  const [ic, setIc] = useState(null);
  const [frame, setFrame] = useState(null);
  const [hist, setHist] = useState([]);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => {
    getMeta()
      .then(setMeta)
      .catch(() =>
        setErr("Backend not reachable — start it with: uvicorn main:app"),
      );
  }, []);
  useEffect(() => {
    if (!meta) return;
    // model or input changed -> abort any run and clear the old results
    if (wsRef.current) wsRef.current.close();
    setRunning(false);
    setFrame(null);
    setHist([]);
    setErr(null);
    if (model === "PINN")
      pinnIC(pinnIndex)
        .then((d) => setIc(d.ic))
        .catch(() => {});
    else
      buildIC(modes, amplitude, 0, seed)
        .then((d) => setIc(d.ic))
        .catch(() => {});
  }, [meta, model, modes, amplitude, seed, pinnIndex]);

  function run() {
    if (wsRef.current) wsRef.current.close();
    setHist([]);
    setFrame(null);
    setErr(null);
    setRunning(true);
    const payload =
      model === "PINN" ? { model, pinn_index: pinnIndex } : { model, ic };
    wsRef.current = runReliability(
      payload,
      (f) => {
        setFrame(f);
        setHist((h) => [...h, f]);
      },
      () => setRunning(false),
      (e) => {
        setErr(e);
        setRunning(false);
      },
    );
  }

  const x = meta?.x || [];
  const muted = "text-slate-500 dark:text-slate-400";
  const inactiveBtn =
    "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700";

  return (
    <div className="space-y-8">
      {/* HEADER */}
      <div>
        <span className="text-xs font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">
          ML model analysis
        </span>
        <h1 className="text-3xl font-extrabold text-slate-800 dark:text-slate-100 mt-1">
          Reliability Analysis
        </h1>
        <p className={`mt-1 ${muted}`}>
          How dependable are the three ML surrogates? Each is trained up to{" "}
          <b>t = 1</b>. We measure their error in-window vs extrapolation, and
          how far each stays reliable (error under 10%).
        </p>
      </div>

      {/* TABS */}
      <div className="inline-flex rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-1">
        {[
          ["findings", "Findings"],
          ["explore", "Try it live"],
        ].map(([id, label]) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`px-4 py-1.5 rounded-lg text-sm font-medium transition ${tab === id ? "bg-indigo-600 text-white" : "text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"}`}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === "findings" && (
        <>
          {/* SCOREBOARD — click to spotlight a model in the chart */}
          <div className="grid grid-cols-3 gap-4">
            {MODELS.map((m) => (
              <ScoreCard
                key={m}
                m={m}
                best={m === "FNO"}
                active={highlight === m}
                onClick={() => setHighlight(highlight === m ? null : m)}
              />
            ))}
          </div>

          {/* CHART + SIDE PANEL side by side */}
          <div className="grid grid-cols-[1.7fr_1fr] gap-5 items-start">
            <Card
              title="Error grows in extrapolation"
              subtitle="accurate while trained (t ≤ 1), then they fan out — click a card above to spotlight one"
            >
              <ErrorChart highlight={highlight} h={250} />
            </Card>

            <div className="flex flex-col gap-5">
              {/* box 1 — reliable horizon */}
              <Card
                title="Reliable horizon"
                subtitle="how far each can be trusted (out of t = 2)"
              >
                <div className="space-y-3 mt-1">
                  {MODELS.map((m) => (
                    <HorizonRunway key={m} m={m} />
                  ))}
                </div>
              </Card>

              {/* box 2 — generalization */}
              <Card
                title="Holds on new inputs?"
                subtitle="extrapolation error — trained (seen) vs unseen"
              >
                <div className="space-y-4 mt-1">
                  {["FNO", "DeepONet"].map((m) => (
                    <div key={m}>
                      <div className="flex justify-between items-center text-xs mb-1.5">
                        <span
                          className="font-semibold"
                          style={{ color: COLOR[m] }}
                        >
                          {m}
                        </span>
                        <span
                          className={`font-medium ${
                            GEN[m].ok
                              ? "text-emerald-600 dark:text-emerald-400"
                              : "text-rose-600 dark:text-rose-400"
                          }`}
                        >
                          {GEN[m].ok ? "generalises ✓" : "degrades ✗"}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 text-[11px] text-slate-500 dark:text-slate-400">
                        <span className="w-16 shrink-0">seen {GEN[m].seen}%</span>
                        <div className="flex-1 h-2 rounded-full bg-slate-100 dark:bg-slate-700">
                          <div
                            className="h-2 rounded-full"
                            style={{
                              width: `${GEN[m].unseen}%`,
                              background: COLOR[m],
                            }}
                          />
                        </div>
                        <span className="w-20 shrink-0 text-right">
                          unseen {GEN[m].unseen}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </Card>
            </div>
          </div>
        </>
      )}

      {tab === "explore" && (
        <div className="space-y-4">
          {/* flow strip: input -> model -> output */}
          <div className="flex items-center gap-3 text-sm rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-4 py-3">
            <span className="inline-flex items-center gap-2 font-semibold text-slate-700 dark:text-slate-200">
              <span className="w-2.5 h-2.5 rounded-full bg-slate-400" /> Input
              <span className="font-normal text-slate-500 dark:text-slate-400">
                {INPUTS[model].takes}
              </span>
            </span>
            <ArrowRight size={16} className="text-slate-400 shrink-0" />
            <span
              className="inline-flex items-center gap-2 font-semibold"
              style={{ color: COLOR[model] }}
            >
              <span
                className="w-2.5 h-2.5 rounded-full"
                style={{ background: COLOR[model] }}
              />
              {model}
            </span>
            <ArrowRight size={16} className="text-slate-400 shrink-0" />
            <span className="inline-flex items-center gap-2 font-semibold text-slate-700 dark:text-slate-200">
              Output
              <span className="font-normal text-slate-500 dark:text-slate-400">
                predicted u(x, t) — compared to the exact answer
              </span>
            </span>
          </div>

          {err && (
            <div className="text-sm text-rose-600 dark:text-rose-300 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 rounded-lg px-3 py-2">
              {err}
            </div>
          )}

          <div className="grid grid-cols-[300px_1fr] gap-5 items-start">
            {/* ---------- CONTROL PANEL ---------- */}
            <Card title="1 · Choose a model">
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-2">
                  {MODELS.map((m) => (
                    <button
                      key={m}
                      onClick={() => setModel(m)}
                      className={`px-2 py-2 rounded-xl text-xs font-bold border transition ${model === m ? "text-white shadow-sm" : inactiveBtn}`}
                      style={
                        model === m
                          ? { background: COLOR[m], borderColor: COLOR[m] }
                          : {}
                      }
                    >
                      {m}
                    </button>
                  ))}
                </div>

                {/* what this model takes */}
                <div className="rounded-xl bg-slate-50 dark:bg-slate-900/40 border border-slate-200 dark:border-slate-700 p-3">
                  <div
                    className="text-[11px] font-semibold uppercase tracking-wide mb-1"
                    style={{ color: COLOR[model] }}
                  >
                    what {model} takes
                  </div>
                  <p className="text-[12px] text-slate-600 dark:text-slate-300 leading-snug">
                    {INPUTS[model].note}
                  </p>
                </div>

                {/* input controls */}
                <div>
                  <div
                    className={`text-xs font-semibold uppercase tracking-wide mb-2 ${muted}`}
                  >
                    2 · Set the input
                  </div>
                  {model === "PINN" ? (
                    <div>
                      <div className={`text-xs mb-1 ${muted}`}>
                        Pick a trained wave
                      </div>
                      <select
                        value={pinnIndex}
                        onChange={(e) => setPinnIndex(+e.target.value)}
                        className="w-full bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-2 text-sm"
                      >
                        {Array.from(
                          { length: meta?.n_pinn_ics || 0 },
                          (_, i) => (
                            <option key={i} value={i}>
                              Trained wave #{i}
                            </option>
                          ),
                        )}
                      </select>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <div>
                        <div
                          className={`flex justify-between text-xs mb-1 ${muted}`}
                        >
                          <span>Wave complexity</span>
                          <span className="font-semibold text-slate-600 dark:text-slate-300">
                            {modes} modes
                          </span>
                        </div>
                        <input
                          type="range"
                          min="1"
                          max="10"
                          value={modes}
                          onChange={(e) => setModes(+e.target.value)}
                          className="w-full accent-indigo-600"
                        />
                      </div>
                      <div>
                        <div
                          className={`flex justify-between text-xs mb-1 ${muted}`}
                        >
                          <span>Amplitude</span>
                          <span className="font-semibold text-slate-600 dark:text-slate-300">
                            {amplitude.toFixed(1)}
                          </span>
                        </div>
                        <input
                          type="range"
                          min="0.4"
                          max="1.5"
                          step="0.1"
                          value={amplitude}
                          onChange={(e) => setAmplitude(+e.target.value)}
                          className="w-full accent-indigo-600"
                        />
                      </div>
                      <button
                        onClick={() => setSeed((s) => s + 1)}
                        className={`w-full inline-flex items-center justify-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium border ${inactiveBtn} hover:bg-slate-100 dark:hover:bg-slate-700`}
                      >
                        <Shuffle size={13} /> New random wave
                      </button>
                    </div>
                  )}
                </div>

                {/* input preview */}
                {ic && (
                  <div>
                    <div className={`text-[11px] mb-1 ${muted}`}>
                      input wave u₀(x) →{" "}
                      {model === "PINN"
                        ? "fixed (per-IC)"
                        : `${x.length}-point vector`}
                    </div>
                    <LineChart
                      series={[{ x, y: ic, color: COLOR[model], width: 2 }]}
                      xr={[-1, 1]}
                      yr={[-1.6, 1.6]}
                      h={100}
                      xlabel="x"
                    />
                  </div>
                )}

                <button
                  onClick={run}
                  disabled={running || !ic}
                  className="w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-indigo-600 text-white text-sm font-semibold hover:bg-indigo-700 disabled:opacity-50"
                >
                  <Play size={15} /> {running ? "Running…" : "3 · Run model"}
                </button>
              </div>
            </Card>

            {/* ---------- RESULTS ---------- */}
            <div className="space-y-4">
              {/* compact metric bar — stays with the charts, no scroll */}
              <div className="grid grid-cols-4 gap-3">
                <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-2.5">
                  <div className="text-[10px] uppercase tracking-wide text-slate-400 dark:text-slate-500">
                    time now
                  </div>
                  <div className="text-xl font-bold text-slate-700 dark:text-slate-200">
                    {frame ? `t = ${frame.t.toFixed(2)}` : "—"}
                  </div>
                  <div className={`text-[10px] ${muted}`}>
                    {frame
                      ? frame.t > 1
                        ? "extrapolating"
                        : "in training window"
                      : "press run"}
                  </div>
                </div>
                <div
                  className="rounded-2xl border px-3 py-2.5"
                  style={{
                    borderColor: frame
                      ? frame.error < 0.1
                        ? "#05966955"
                        : "#e11d4855"
                      : undefined,
                  }}
                >
                  <div className="text-[10px] uppercase tracking-wide text-slate-400 dark:text-slate-500">
                    error now
                  </div>
                  <div
                    className="text-2xl font-extrabold leading-tight"
                    style={{
                      color: frame
                        ? frame.error < 0.1
                          ? "#059669"
                          : frame.error < 0.25
                            ? "#d97706"
                            : "#e11d48"
                        : "#94a3b8",
                    }}
                  >
                    {frame ? `${Math.round(frame.error * 100)}%` : "—"}
                  </div>
                  <div
                    className="text-[10px] font-medium"
                    style={{
                      color: frame
                        ? frame.error < 0.1
                          ? "#059669"
                          : "#e11d48"
                        : "#94a3b8",
                    }}
                  >
                    {frame ? (frame.error < 0.1 ? "reliable" : "unreliable") : "—"}
                  </div>
                </div>
                <div className="rounded-2xl border border-indigo-200 dark:border-indigo-500/30 bg-indigo-50 dark:bg-indigo-500/10 px-3 py-2.5">
                  <div className="inline-flex items-center gap-1 text-[10px] uppercase tracking-wide text-indigo-500 dark:text-indigo-400">
                    <Timer size={11} /> reliable to
                  </div>
                  <div className="text-2xl font-bold text-indigo-700 dark:text-indigo-300 leading-tight">
                    {frame ? `t = ${frame.horizon.toFixed(2)}` : "—"}
                  </div>
                  <div className="text-[10px] text-indigo-500/80 dark:text-indigo-400/80">
                    out of t = 2
                  </div>
                </div>
                <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-2.5">
                  <div className="text-[10px] uppercase tracking-wide text-slate-400 dark:text-slate-500">
                    mean error
                  </div>
                  <div className="flex items-baseline gap-2">
                    <span className="text-sm font-bold text-slate-700 dark:text-slate-200">
                      {frame ? `${Math.round(frame.in_window * 100)}%` : "—"}
                    </span>
                    <span className={`text-[10px] ${muted}`}>in-window</span>
                  </div>
                  <div className="flex items-baseline gap-2">
                    <span
                      className="text-sm font-bold"
                      style={{ color: COLOR[model] }}
                    >
                      {frame ? `${Math.round(frame.extrap * 100)}%` : "—"}
                    </span>
                    <span className={`text-[10px] ${muted}`}>extrapolation</span>
                  </div>
                </div>
              </div>

              {/* two charts side by side — both visible at once */}
              <div className="grid grid-cols-2 gap-4">
                <Card
                  title="Prediction vs true answer"
                  subtitle="coloured = model · dashed = exact (Cole–Hopf)"
                >
                  <LineChart
                    h={230}
                    xr={[-1, 1]}
                    yr={[-1.6, 1.6]}
                    xlabel="x"
                    series={
                      frame
                        ? [
                            {
                              x,
                              y: frame.true,
                              color: "#94a3b8",
                              dashed: true,
                              width: 1.5,
                            },
                            { x, y: frame.u, color: COLOR[model], width: 2.5 },
                          ]
                        : ic
                          ? [{ x, y: ic, color: COLOR[model] + "55", width: 2 }]
                          : []
                    }
                  />
                </Card>

                <Card
                  title="Error over time"
                  subtitle="dashed = training boundary (t = 1) · amber = 10% fail"
                >
                  <LineChart
                    h={230}
                    xr={[0, 2]}
                    yr={[0, 0.7]}
                    hline={0.1}
                    vline={1.0}
                    xlabel="time t"
                    series={[
                      {
                        x: hist.map((f) => f.t),
                        y: hist.map((f) => Math.min(0.7, f.error)),
                        color: COLOR[model],
                        width: 2.5,
                      },
                    ]}
                  />
                </Card>
              </div>
            </div>
          </div>
        </div>
      )}

      {tab === "findings" && (
        <div className="rounded-2xl p-5 bg-gradient-to-r from-emerald-50 via-white to-white dark:from-emerald-500/10 dark:via-slate-800 dark:to-slate-800 border border-emerald-200 dark:border-emerald-500/30">
          <div className="text-sm font-semibold text-emerald-800 dark:text-emerald-300">
            Conclusion
          </div>
          <p className="text-sm text-slate-700 dark:text-slate-200 mt-1">
            <b>FNO is the most dependable surrogate</b> — lowest error, longest
            reliable horizon, and it generalises to unseen inputs. All three are
            accurate in-window but <b>degrade in extrapolation</b> — exactly why
            the hybrid needs a trust-gated switch to the numerical solver.
          </p>
        </div>
      )}
    </div>
  );
}
