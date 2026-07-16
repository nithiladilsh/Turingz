export function Card({ title, subtitle, children, className = "" }) {
  return (
    <div className={`bg-white rounded-2xl border border-slate-200 shadow-sm p-5 ${className}`}>
      {title && <h3 className="text-sm font-semibold text-slate-800">{title}</h3>}
      {subtitle && <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>}
      <div className={title ? "mt-3" : ""}>{children}</div>
    </div>
  );
}

export function Stat({ label, value, tone = "slate" }) {
  const tones = {
    slate: "text-slate-800", green: "text-emerald-600", red: "text-rose-600", indigo: "text-indigo-600",
  };
  return (
    <div className="bg-slate-50 rounded-xl px-4 py-3">
      <div className="text-xs text-slate-500">{label}</div>
      <div className={`text-2xl font-semibold mt-0.5 ${tones[tone]}`}>{value}</div>
    </div>
  );
}

export function Banner({ ok, text }) {
  return (
    <div className={`rounded-xl px-4 py-3 text-sm font-medium border ${ok
      ? "bg-emerald-50 text-emerald-700 border-emerald-200"
      : "bg-rose-50 text-rose-700 border-rose-200"}`}>
      {text}
    </div>
  );
}
