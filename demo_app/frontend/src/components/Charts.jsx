function path(xs, ys, xr, yr, w, h, pad) {
  const sx = (v) => pad + ((v - xr[0]) / (xr[1] - xr[0])) * (w - 2 * pad);
  const sy = (v) => h - pad - ((v - yr[0]) / (yr[1] - yr[0])) * (h - 2 * pad);
  return xs.map((x, i) => `${i ? "L" : "M"}${sx(x).toFixed(1)} ${sy(ys[i]).toFixed(1)}`).join(" ");
}

export function LineChart({ series, xr, yr, w = 460, h = 200, hline, vline, xlabel, ylabel }) {
  const pad = 34;
  const sx = (v) => pad + ((v - xr[0]) / (xr[1] - xr[0])) * (w - 2 * pad);
  const sy = (v) => h - pad - ((v - yr[0]) / (yr[1] - yr[0])) * (h - 2 * pad);
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full">
      <rect x={pad} y={pad} width={w - 2 * pad} height={h - 2 * pad} fill="#fff" stroke="#eef0f3" />
      {hline != null && (
        <line x1={pad} x2={w - pad} y1={sy(hline)} y2={sy(hline)} stroke="#94a3b8" strokeDasharray="4 3" />
      )}
      {vline != null && (
        <line x1={sx(vline)} x2={sx(vline)} y1={pad} y2={h - pad} stroke="#e11d48" strokeWidth="1.5" />
      )}
      {series.map((s, i) =>
        s.x.length > 1 ? (
          <path key={i} d={path(s.x, s.y, xr, yr, w, h, pad)} fill="none"
            stroke={s.color} strokeWidth={s.width || 2} strokeDasharray={s.dashed ? "5 4" : "0"} />
        ) : null
      )}
      {xlabel && <text x={w / 2} y={h - 6} textAnchor="middle" fontSize="10" fill="#64748b">{xlabel}</text>}
      {ylabel && <text x={10} y={h / 2} textAnchor="middle" fontSize="10" fill="#64748b"
        transform={`rotate(-90 10 ${h / 2})`}>{ylabel}</text>}
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
        <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`} fill="none" stroke="#eef0f3" strokeWidth="12" />
        <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`} fill="none" stroke={color} strokeWidth="12"
          strokeDasharray={`${v * circ} ${circ}`} strokeLinecap="round" />
      </svg>
      <div>
        <div className="text-3xl font-semibold" style={{ color }}>{v.toFixed(2)}</div>
        <div className="text-xs text-slate-500">trust score</div>
      </div>
    </div>
  );
}
