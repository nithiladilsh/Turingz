export function Card({ title, subtitle, children, className = "" }) {
  return (
    <div className={`bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm p-5 ${className}`}>
      {title && <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">{title}</h3>}
      {subtitle && <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{subtitle}</p>}
      <div className={title ? "mt-3" : ""}>{children}</div>
    </div>
  );
}

export function Stat({ label, value, tone = "slate" }) {
  const tones = {
    slate: "text-slate-800 dark:text-slate-100",
    green: "text-emerald-600 dark:text-emerald-400",
    red: "text-rose-600 dark:text-rose-400",
    indigo: "text-indigo-600 dark:text-indigo-400",
  };
  return (
    <div className="bg-slate-50 dark:bg-slate-700/40 rounded-xl px-4 py-3">
      <div className="text-xs text-slate-500 dark:text-slate-400">{label}</div>
      <div className={`text-2xl font-semibold mt-0.5 ${tones[tone]}`}>{value}</div>
    </div>
  );
}

/* --- scientific-notation display helpers ---
   Render values like 1e-6 as "10" with a real superscript exponent, using
   Unicode superscript glyphs so they work in both HTML and SVG <text>. */
const _SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
export const supStr = (n) => String(n).split("").map((c) => _SUP[c] ?? c).join("");
// integer exponent -> "10⁻⁶" etc. (for log-axis tick labels)
export const sci10 = (exp) => `10${supStr(exp)}`;
// a number or JS scientific string -> "m×10ⁿ" superscript, dropping a leading ×1
export function eToSup(v) {
  if (v === 0 || v === "0") return "0";
  const s = typeof v === "number" ? v.toExponential(0) : String(v);
  const m = s.match(/^(-?\d+(?:\.\d+)?)[eE]([+-]?\d+)$/);
  if (!m) return s;
  const mant = m[1];
  const base = `10${supStr(parseInt(m[2], 10))}`;
  return mant === "1" || mant === "1.0" ? base : `${mant}×${base}`;
}

export function Banner({ ok, text }) {
  return (
    <div className={`rounded-xl px-4 py-3 text-sm font-medium border ${ok
      ? "bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-500/30"
      : "bg-rose-50 dark:bg-rose-500/10 text-rose-700 dark:text-rose-300 border-rose-200 dark:border-rose-500/30"}`}>
      {text}
    </div>
  );
}
