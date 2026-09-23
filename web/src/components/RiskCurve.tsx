import { useMemo } from "react";
import type { PredictionSource, VideoPrediction } from "../lib/predictions.ts";
import { sourceLabel } from "../lib/predictions.ts";
import { prepareRiskSeries, riskAtTime } from "../lib/risk.ts";

interface Props {
  prediction: VideoPrediction;
  source: PredictionSource;
  durationSec: number;
  currentTime?: number;
  onSeek?: (seconds: number) => void;
}

const width = 1000;
const height = 240;
const left = 42;
const right = 18;
const top = 18;
const bottom = 32;
const plotWidth = width - left - right;
const plotHeight = height - top - bottom;

export function RiskCurve({ prediction, source, durationSec, currentTime, onSeek }: Props) {
  const points = useMemo(() => prepareRiskSeries(prediction.risk), [prediction.risk]);
  const duration = Math.max(1, durationSec, points.at(-1)?.[0] ?? 0);
  const polyline = points
    .map(([time, score]) => `${left + (time / duration) * plotWidth},${top + (1 - score) * plotHeight}`)
    .join(" ");
  const currentRisk = currentTime === undefined ? null : riskAtTime(prediction.risk, currentTime);

  return (
    <section className="panel risk-panel" aria-label="Accident risk curve">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Causal signal</span>
          <h3>Accident risk curve</h3>
        </div>
        <span className="count-pill">{prediction.risk.length} samples</span>
      </div>
      <p className="data-origin">{sourceLabel(source)}</p>
      {points.length === 0 ? (
        <div className="empty-state">No risk curve is available for this video.</div>
      ) : (
        <>
          <div className="chart-wrap">
            <svg
              className="risk-svg"
              viewBox={`0 0 ${width} ${height}`}
              role="img"
              aria-label="Risk score from zero to one over time"
              onClick={(event) => {
                if (!onSeek) return;
                const box = event.currentTarget.getBoundingClientRect();
                const relative = ((event.clientX - box.left) / box.width) * width;
                onSeek(Math.min(duration, Math.max(0, ((relative - left) / plotWidth) * duration)));
              }}
            >
              {[0, 0.5, 1].map((score) => {
                const y = top + (1 - score) * plotHeight;
                return (
                  <g key={score}>
                    <line className="chart-grid" x1={left} y1={y} x2={width - right} y2={y} />
                    <text className="chart-axis" x={left - 10} y={y + 4} textAnchor="end">
                      {score.toFixed(1)}
                    </text>
                  </g>
                );
              })}
              {prediction.events
                .filter((event) => event[2] === "accident")
                .map(([start], index) => (
                  <line
                    className="chart-accident"
                    key={`${start}-${index}`}
                    x1={left + (start / duration) * plotWidth}
                    x2={left + (start / duration) * plotWidth}
                    y1={top}
                    y2={height - bottom}
                  />
                ))}
              <line className="chart-threshold" x1={left} x2={width - right} y1={top + plotHeight / 2} y2={top + plotHeight / 2} />
              <polyline className="chart-line" points={polyline} />
              {currentTime !== undefined && (
                <line
                  className="chart-playhead"
                  x1={left + (currentTime / duration) * plotWidth}
                  x2={left + (currentTime / duration) * plotWidth}
                  y1={top}
                  y2={height - bottom}
                />
              )}
              <text className="chart-axis" x={left} y={height - 7}>0:00</text>
              <text className="chart-axis" x={width - right} y={height - 7} textAnchor="end">
                {`${Math.floor(duration / 60)}:${String(Math.floor(duration % 60)).padStart(2, "0")}`}
              </text>
            </svg>
          </div>
          <div className="chart-note">
            <span><i className="legend-line" /> Risk score</span>
            <span><i className="legend-accident" /> Accident start</span>
            <span>0.5 alarm threshold</span>
            {currentRisk !== null && <strong>At playhead: {currentRisk.toFixed(2)}</strong>}
          </div>
        </>
      )}
    </section>
  );
}
