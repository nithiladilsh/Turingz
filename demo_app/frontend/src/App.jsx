import { useState, useEffect } from "react";
import { Sun, Moon } from "lucide-react";
import Overview from "./pages/Overview.jsx";
import TrustPage from "./pages/TrustPage.jsx";
import FDMPage from "./pages/FDMPage.jsx";
import Placeholder from "./pages/Placeholder.jsx";
import CouplingPage from "./pages/CouplingPage.jsx";
import ColeHopfPage from "./pages/ColeHopfPage.jsx";
import RobustnessPage from "./pages/RobustnessPage.jsx";

const SECTIONS = [
  { id: "overview", label: "Overview", group: "Project" },

  { id: "fdm", label: "Finite Difference (FDM)", group: "Numerical solvers" },
  { id: "colehopf", label: "Cole–Hopf", group: "Numerical solvers" },
  { id: "spectral", label: "Spectral", group: "Numerical solvers" },

  { id: "reliability", label: "Reliability analysis", group: "ML model analysis" },
  { id: "robustness", label: "Robustness analysis", group: "ML model analysis" },
  { id: "cost", label: "Cost analysis", group: "ML model analysis" },

  { id: "trust", label: "Trust score", group: "Hybrid components" },
  { id: "coupling", label: "Coupling", group: "Hybrid components" },
  { id: "costcontrol", label: "Cost control", group: "Hybrid components" },

  { id: "hybrid", label: "Hybrid engine", group: "Final · Hybrid engine" },
];

const NOTES = {
  fdm: "The finite-difference (FDM) solver and its experiments: run it on a chosen wave and show the accuracy / error-over-time graphs that justify it against the Cole-Hopf reference.",
  colehopf: "The exact Cole-Hopf solution, used as the ground-truth reference.",
  spectral: "The pseudo-spectral numerical solver.",
  reliability: "Reliability analysis of PINN, FNO and DeepONet: accuracy in-window vs extrapolation, error curves, and which model is most dependable.",
  robustness: "Robustness analysis of the three ML models (behaviour under perturbed / harder inputs).",
  cost: "Cost analysis of the three ML models (speed and compute trade-offs).",
  coupling: "The coupling module: how the ML model and the numerical solver are joined at the hand-off.",
  costcontrol: "The cost-control module: managing compute budget across the hybrid run.",
  hybrid: "The full product: the ML model predicts each step, the trust module scores it, and control switches to the numerical solver when trust drops. Reuses the same wave-builder input as the Trust page.",
};

export default function App() {
  const [active, setActive] = useState("overview");
  const [dark, setDark] = useState(() => {
    const saved = localStorage.getItem("theme");
    if (saved) return saved === "dark";
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  });

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("theme", dark ? "dark" : "light");
  }, [dark]);

  const groups = [...new Set(SECTIONS.map((s) => s.group))];
  const sec = SECTIONS.find((s) => s.id === active);

  return (
    <div className="min-h-screen">
      <aside className="fixed top-0 left-0 h-screen w-64 overflow-y-auto bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 p-4 z-20 flex flex-col">
        <div className="px-2 py-3">
          <div className="text-lg font-bold text-slate-800 dark:text-slate-100">Team Turingz</div>
          <div className="text-xs text-slate-500 dark:text-slate-400">Hybrid ML + Numerical PDE Solver</div>
        </div>
        <nav className="mt-3 space-y-4 flex-1">
          {groups.map((g) => (
            <div key={g}>
              <div className="px-2 text-[11px] uppercase tracking-wide text-slate-400 dark:text-slate-500 mb-1">{g}</div>
              {SECTIONS.filter((s) => s.group === g).map((s) => (
                <button key={s.id} onClick={() => setActive(s.id)}
                  className={`w-full text-left px-3 py-2 rounded-lg text-sm mb-0.5 transition ${
                    active === s.id
                      ? "bg-indigo-600 text-white"
                      : "text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"}`}>
                  {s.label}
                </button>
              ))}
            </div>
          ))}
        </nav>
        <button onClick={() => setDark((d) => !d)}
          className="mt-4 flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 transition">
          {dark ? <Sun size={16} /> : <Moon size={16} />}
          {dark ? "Light mode" : "Dark mode"}
        </button>
      </aside>

      <main className="ml-72 mx-10 p-8">
        {active === "overview" && <Overview go={setActive} />}
        {active === "trust" && <TrustPage />}
        {active === "fdm" && <FDMPage />}
        {active === "coupling" && <CouplingPage />}
        {active === "colehopf" && <ColeHopfPage />}
        {active === "robustness" && <RobustnessPage />}
        {!["overview", "trust", "fdm", "coupling", "colehopf", "robustness"].includes(active) && (
          <Placeholder title={sec.label} group={sec.group} note={NOTES[active]} />
        )}
      </main>
    </div>
  );
}
