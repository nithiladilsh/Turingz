function path(xs, ys, xr, yr, w, h, p) {
  const { padL, padR, padT, padB } = p;
  const sx = (v) => padL + ((v - xr[0]) / (xr[1] - xr[0])) * (w - padL - padR);
  const sy = (v) => h - padB - ((v - yr[0]) / (yr[1] - yr[0])) * (h - padT - padB);
  return xs.map((x, i) => `${i ? "L" : "M"}${sx(x).toFixed(1)} ${sy(ys[i]).toFixed(1)}`).join(" ");
}

const fmtTick = (v) => {
  if (Math.abs(v) < 1e-9) return "0";
  if (Number.isInteger(v)) return String(v);
  return String(Math.round(v * 100) / 100);
};
const ticks = (r, n) =>
  Array.from({ length: n + 1 }, (_, i) => r[0] + (i / n) * (r[1] - r[0]));

export function LineChart({ series, xr, yr, w = 460, h = 200, hline, vline, xlabel, ylabel, xticks = 5, yticks = 4 }) {
  const padL = 46, padR = 14, padT = 12, padB = 34;
  const sx = (v) => padL + ((v - xr[0]) / (xr[1] - xr[0])) * (w - padL - padR);
  const sy = (v) => h - padB - ((v - yr[0]) / (yr[1] - yr[0])) * (h - padT - padB);
  const xt = ticks(xr, xticks), yt = ticks(yr, yticks);
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full">
      <rect x={padL} y={padT} width={w - padL - padR} height={h - padT - padB} fill="var(--chart-surface)" stroke="var(--chart-grid)" />

      {/* y gridlines + value labels */}
      {yt.map((v, i) => (
        <g key={"y" + i}>
          {i > 0 && i < yt.length && (
            <line x1={padL} x2={w - padR} y1={sy(v)} y2={sy(v)} stroke="var(--chart-grid)" strokeOpacity="0.5" />
          )}
          <text x={padL - 6} y={sy(v) + 3} textAnchor="end" fontSize="9" fill="var(--chart-axis)">{fmtTick(v)}</text>
        </g>
      ))}

      {/* x tick marks + value labels */}
      {xt.map((v, i) => (
        <g key={"x" + i}>
          <line x1={sx(v)} x2={sx(v)} y1={h - padB} y2={h - padB + 4} stroke="var(--chart-axis)" strokeOpacity="0.6" />
          <text x={sx(v)} y={h - padB + 15} textAnchor="middle" fontSize="9" fill="var(--chart-axis)">{fmtTick(v)}</text>
        </g>
      ))}

      {hline != null && (
        <line x1={padL} x2={w - padR} y1={sy(hline)} y2={sy(hline)} stroke="#f59e0b" strokeDasharray="4 3" />
      )}
      {vline != null && (
        <line x1={sx(vline)} x2={sx(vline)} y1={padT} y2={h - padB} stroke="#e11d48" strokeWidth="1.5" />
      )}
      {series.map((s, i) =>
        s.x.length > 1 ? (
          <path key={i} d={path(s.x, s.y, xr, yr, w, h, { padL, padR, padT, padB })} fill="none"
            stroke={s.color} strokeWidth={s.width || 2} strokeDasharray={s.dashed ? "5 4" : "0"}
            strokeLinejoin="round" strokeLinecap="round" />
        ) : null
      )}
      {xlabel && <text x={(padL + w - padR) / 2} y={h - 4} textAnchor="middle" fontSize="10" fill="var(--chart-axis)">{xlabel}</text>}
      {ylabel && <text x={11} y={(padT + h - padB) / 2} textAnchor="middle" fontSize="10" fill="var(--chart-axis)"
        transform={`rotate(-90 11 ${(padT + h - padB) / 2})`}>{ylabel}</text>}
    </svg>
  );
}

export function Gauge({ value }) {
  const v = Math.max(0, Math.min(1, value));
  const color = v > 0.66 ? "#059669" : v > 0.4 ? "#d97706" : "#e11d48";
  const r = 52, cx = 70, cy = 70, circ = Math.PI * r;
  return (
    <div className="flex items-center gap-4">
      <svg viewBox="0 0 140 88" width="140" height="88">
        <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`} fill="none" stroke="var(--chart-grid)" strokeWidth="12" />
        <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`} fill="none" stroke={color} strokeWidth="12"
          strokeDasharray={`${v * circ} ${circ}`} strokeLinecap="round" />
      </svg>
      <div>
        <div className="text-3xl font-semibold" style={{ color }}>{v.toFixed(2)}</div>
        <div className="text-xs text-slate-500 dark:text-slate-400">trust score</div>
      </div>
    </div>
  );
}
