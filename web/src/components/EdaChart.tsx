import type { TimeSeries } from "../lib/siteAssets.ts";

const palette = ["#a7e7b9", "#efc986", "#9fc8ee", "#f19f99", "#c6aeea"];
const width = 680;
const height = 210;
const left = 38;
const right = 12;
const top = 14;
const bottom = 28;

export function EdaChart({ title, data, note }: { title: string; data: TimeSeries; note: string }) {
  const names = Object.keys(data).filter((name) => name !== "t");
  const duration = Math.max(1, data.t.at(-1) ?? 1);
  const maximum = Math.max(1, ...names.flatMap((name) => data[name]));
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const pathFor = (values: number[]) => values.map((value, index) => {
    const x = left + ((data.t[index] ?? 0) / duration) * plotWidth;
    const y = top + (1 - value / maximum) * plotHeight;
    return `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");

  return <div className="eda-chart">
    <h4>{title}</h4>
    <div className="eda-chart-wrap">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title}: ${names.join(", ")} over ${duration.toFixed(0)} seconds`}>
        {[0, 0.5, 1].map((fraction) => {
          const y = top + (1 - fraction) * plotHeight;
          return <g key={fraction}>
            <line x1={left} x2={width - right} y1={y} y2={y} className="chart-grid" />
            <text x={left - 7} y={y + 4} textAnchor="end" className="chart-axis">{Math.round(maximum * fraction)}</text>
          </g>;
        })}
        {names.map((name, index) => <path key={name} d={pathFor(data[name])} fill="none" stroke={palette[index % palette.length]} strokeWidth="1.5" />)}
        <text x={left} y={height - 5} className="chart-axis">0:00</text>
        <text x={width - right} y={height - 5} textAnchor="end" className="chart-axis">{`${Math.floor(duration / 60)}:${String(Math.floor(duration % 60)).padStart(2, "0")}`}</text>
      </svg>
    </div>
    <div className="eda-chart-legend">{names.map((name, index) => <span key={name}><i style={{ background: palette[index % palette.length] }} />{name.replaceAll("_", " ")}</span>)}</div>
    <p>{note}</p>
  </div>;
}
