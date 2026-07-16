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

export function Banner({ ok, text }) {
  return (
    <div className={`rounded-xl px-4 py-3 text-sm font-medium border ${ok
      ? "bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-500/30"
      : "bg-rose-50 dark:bg-rose-500/10 text-rose-700 dark:text-rose-300 border-rose-200 dark:border-rose-500/30"}`}>
      {text}
    </div>
  );
}
