import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { X, Table2, Waves, Minus, Plus, ChevronLeft, ChevronRight, Play, Pause } from "lucide-react";
import { datasetSample } from "../api.js";
import { LineChart } from "./Charts.jsx";

const iconBtn = "w-8 h-8 grid place-items-center rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-500 dark:text-slate-400 hover:text-indigo-600 dark:hover:text-indigo-400 hover:border-indigo-300 dark:hover:border-indigo-500 disabled:opacity-40 disabled:pointer-events-none transition";

export default function DatasetModal({ open, onClose }) {
  const [index, setIndex] = useState(0);
  const [tIndex, setTIndex] = useState(0);
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!open) return;
    datasetSample(index, tIndex).then(setData).catch(() => setErr("Backend not reachable. Start it with: uvicorn main:app"));
  }, [open, index, tIndex]);

  useEffect(() => {
    if (!playing) return;
    const maxT = data ? data.nt - 1 : 199;
    if (tIndex >= maxT) { setPlaying(false); return; }
    const id = setTimeout(() => setTIndex((t) => t + 1), 90);
    return () => clearTimeout(id);
  }, [playing, tIndex, data]);

  useEffect(() => { if (!open) setPlaying(false); }, [open]);

  // lock page scroll while the modal is open, so the backdrop always covers
  // the full window with no visible gap at the bottom of a tall page
  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = prev; };
  }, [open]);

  const clampIndex = (v) => Math.max(0, Math.min(999, v));
  // filled-track look: colored up to the thumb, plain track after it
  const rangeStyle = (pct) => ({
    background: `linear-gradient(to right, currentColor ${pct}%, var(--track) ${pct}%)`,
  });

  if (!open) return null;

  return createPortal(
    <div className="fixed inset-0 z-[100] grid place-items-center bg-slate-900/60 backdrop-blur-sm p-6" onClick={onClose}>
      <div
        className="w-full max-w-6xl h-[88vh] bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-2xl flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* HEADER */}
        <div className="flex items-center justify-between px-6 py-4 bg-gradient-to-r from-indigo-50 via-white to-white dark:from-indigo-500/10 dark:via-slate-800 dark:to-slate-800 border-b border-slate-200 dark:border-slate-700 shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl grid place-items-center bg-indigo-100 dark:bg-indigo-500/20 text-indigo-600 dark:text-indigo-400">
              <Table2 size={19} />
            </div>
            <div>
              <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">Raw dataset viewer</div>
              <div className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                data/colehopf/burgers_colehopf.pt — the exact values used to train and evaluate the models
              </div>
            </div>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700 transition">
            <X size={18} />
          </button>
        </div>

        {err && <div className="mx-6 mt-4 text-sm text-rose-600 dark:text-rose-300 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 rounded-lg px-3 py-2 shrink-0">{err}</div>}

        {/* BODY: left = selectors + wave, right = table */}
        <div className="flex flex-1 min-h-0">
          {/* LEFT */}
          <div className="w-120 shrink-0 border-r border-slate-200 dark:border-slate-700 bg-slate-50/60 dark:bg-slate-900/30 overflow-y-auto px-6 py-4 space-y-4">
            {/* IC INDEX */}
            <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4">
              <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 mb-1.5">
                <span>Initial condition</span>
              </div>
              <div className="flex items-center gap-2">
                <button className={iconBtn} disabled={index <= 0} onClick={() => setIndex(clampIndex(index - 1))}><Minus size={14} /></button>
                <input
                  type="number" min="0" max="999" value={index}
                  onChange={(e) => setIndex(clampIndex(+e.target.value || 0))}
                  className="w-16 text-center bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 border border-slate-200 dark:border-slate-700 rounded-lg px-2 py-1.5 text-sm font-semibold focus:outline-none focus:ring-2 focus:ring-indigo-400/50"
                />
                <button className={iconBtn} disabled={index >= 999} onClick={() => setIndex(clampIndex(index + 1))}><Plus size={14} /></button>
                <span className="text-xs text-slate-400 dark:text-slate-500 ml-auto">/ 999</span>
              </div>
              <input
                type="range" min="0" max="999" value={index}
                onChange={(e) => setIndex(+e.target.value)}
                className="w-full mt-2.5 text-indigo-600 dark:text-indigo-400"
                style={rangeStyle((index / 999) * 100)}
              />
            </div>

            {/* TIME STEP */}
            <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4">
              <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 mb-1.5">
                <span>Time step</span>
                <span className="font-medium text-slate-600 dark:text-slate-300">t = {data ? data.t.toFixed(3) : "—"}</span>
              </div>
              <div className="text-[11px] text-slate-400 dark:text-slate-500 mb-1.5">step {tIndex} / {data ? data.nt - 1 : "…"}</div>
              <div className="flex items-center gap-2">
                <button className={iconBtn} onClick={() => setPlaying((p) => !p)} title={playing ? "Pause" : "Play through time"}>
                  {playing ? <Pause size={14} /> : <Play size={14} />}
                </button>
                <button className={iconBtn} disabled={tIndex <= 0} onClick={() => setTIndex((t) => Math.max(0, t - 1))}><ChevronLeft size={14} /></button>
                <input
                  type="range" min="0" max={data ? data.nt - 1 : 199} value={tIndex}
                  onChange={(e) => { setPlaying(false); setTIndex(+e.target.value); }}
                  className="flex-1 text-indigo-600 dark:text-indigo-400"
                  style={rangeStyle((tIndex / (data ? data.nt - 1 : 199)) * 100)}
                />
                <button className={iconBtn} disabled={data && tIndex >= data.nt - 1} onClick={() => setTIndex((t) => Math.min((data ? data.nt : 200) - 1, t + 1))}><ChevronRight size={14} /></button>
              </div>
              <div className="flex justify-between text-[10px] text-slate-400 dark:text-slate-500 mt-1">
                <span>t = 0</span>
                <span className="text-amber-500 dark:text-amber-400 font-medium">trained window ends at t = 1</span>
                <span>t = 2</span>
              </div>
            </div>

            {/* WAVE PREVIEW */}
            {data && (
              <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 px-4 pt-3 pb-1">
                <div className="flex items-center gap-1.5 text-[11px] font-semibold text-slate-500 dark:text-slate-400 mb-1">
                  <Waves size={12} /> u(x, t = {data.t.toFixed(2)})
                </div>
                <LineChart h={150} xr={[-1, 1]} yr={[-1.6, 1.6]}
                  series={[{ x: data.x, y: data.u, color: "#4f46e5", width: 2 }]} />
                <div className="text-[10px] text-slate-400 dark:text-slate-500 text-center pb-1">this is what the table on the right is drawn from</div>
              </div>
            )}
          </div>

          {/* RIGHT: TABLE */}
          <div className="flex-1 min-w-0 overflow-y-auto">
            {data ? (
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-slate-100 dark:bg-slate-900/80 backdrop-blur text-slate-500 dark:text-slate-400 shadow-sm">
                  <tr>
                    <th className="text-left font-semibold px-6 py-2.5 w-16">i</th>
                    <th className="text-left font-semibold px-6 py-2.5">x</th>
                    <th className="text-left font-semibold px-6 py-2.5">u(x, t)</th>
                  </tr>
                </thead>
                <tbody>
                  {data.x.map((xv, i) => (
                    <tr key={i} className="border-t border-slate-100 dark:border-slate-700/60 text-slate-700 dark:text-slate-200 even:bg-slate-50/70 dark:even:bg-slate-900/25 hover:bg-indigo-50/60 dark:hover:bg-indigo-500/10 transition-colors">
                      <td className="px-6 py-1.5 text-slate-400 dark:text-slate-500 tabular-nums">{i}</td>
                      <td className="px-6 py-1.5 tabular-nums">{xv.toFixed(4)}</td>
                      <td className="px-6 py-1.5 tabular-nums font-semibold text-indigo-700 dark:text-indigo-300">{data.u[i].toFixed(5)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="p-10 text-center text-sm text-slate-400 dark:text-slate-500">Loading…</div>
            )}
          </div>
        </div>

        {/* FOOTER */}
        <div className="px-6 py-3 border-t border-slate-200 dark:border-slate-700 text-[11px] text-slate-400 dark:text-slate-500 bg-slate-50/60 dark:bg-slate-900/30 shrink-0">
          {data ? `${data.nx} spatial points shown for this time step · 512 points × 200 steps × 1000 initial conditions = 102.4M values in the full dataset` : ""}
        </div>
      </div>
    </div>,
    document.body
  );
}
