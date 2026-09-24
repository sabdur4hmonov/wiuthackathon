import { useMemo } from "react";
import type { PredictionSource, VideoPrediction } from "../lib/predictions.ts";
import { sourceLabel } from "../lib/predictions.ts";
import { buildTimeline } from "../lib/timeline.ts";
import { EMPTY_EVENT_MESSAGE } from "../lib/disclosures.ts";

interface Props {
  prediction: VideoPrediction;
  source: PredictionSource;
  durationHint?: number;
  currentTime?: number;
  onSeek?: (seconds: number) => void;
}

function prettyLabel(label: string): string {
  return label.replaceAll("_", " ");
}

export function clock(seconds: number, precise = false): string {
  const hundredths = Math.round(seconds * 100);
  const minutes = Math.floor(hundredths / 6000);
  const remaining = hundredths % 6000;
  const secondsPart = precise
    ? (remaining / 100).toFixed(2).padStart(5, "0")
    : String(Math.floor(remaining / 100)).padStart(2, "0");
  return `${minutes}:${secondsPart}`;
}

export function EventTimeline({
  prediction,
  source,
  durationHint,
  currentTime,
  onSeek,
}: Props) {
  const data = useMemo(
    () => buildTimeline(prediction.events, prediction.risk, durationHint),
    [prediction, durationHint],
  );

  return (
    <section className="panel timeline-panel" aria-label="Event timeline">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Temporal view</span>
          <h3>Event timeline</h3>
        </div>
        <span className="count-pill">{prediction.events.length} events</span>
      </div>
      <p className="data-origin">{sourceLabel(source)}</p>
      {data.lanes.length === 0 ? (
        <div className="empty-state">{EMPTY_EVENT_MESSAGE}</div>
      ) : (
        <>
          <div className="timeline-ruler" aria-hidden="true">
            {[0, 0.25, 0.5, 0.75, 1].map((fraction) => (
              <span key={fraction}>{clock(data.durationSec * fraction)}</span>
            ))}
          </div>
          <div className="timeline-lanes">
            {data.lanes.map((lane) => (
              <div className="timeline-lane" key={lane.label}>
                <div className="lane-label">{prettyLabel(lane.label)}</div>
                <div className="lane-track">
                  {lane.events.map(([start, end], index) => (
                    <button
                      className={`event-segment event-${lane.label}`}
                      key={`${lane.label}-${start}-${end}-${index}`}
                      style={{
                        left: `${(start / data.durationSec) * 100}%`,
                        width: `${((end - start) / data.durationSec) * 100}%`,
                      }}
                      title={`${prettyLabel(lane.label)} · ${start.toFixed(2)}–${end.toFixed(2)} s`}
                      aria-label={`Seek to ${prettyLabel(lane.label)} at ${start.toFixed(2)} seconds; ends at ${end.toFixed(2)} seconds`}
                      onClick={() => onSeek?.(start)}
                    />
                  ))}
                  {currentTime !== undefined && (
                    <span
                      className="timeline-playhead"
                      style={{ left: `${Math.min(100, Math.max(0, (currentTime / data.durationSec) * 100))}%` }}
                      aria-hidden="true"
                    />
                  )}
                </div>
              </div>
            ))}
          </div>
          <div className="event-index">
            {prediction.events
              .map(([start, end, label]) => ({ start, end, label }))
              .sort((a, b) => a.start - b.start)
              .map((event, index) => (
                <button
                  className="event-index-item"
                  key={`${event.label}-${event.start}-${index}`}
                  onClick={() => onSeek?.(event.start)}
                >
                  <span className="event-index-number">{String(index + 1).padStart(2, "0")}</span>
                  <span>{prettyLabel(event.label)}</span>
                  <span className="event-index-time">
                    {clock(event.start, true)} — {clock(event.end, true)}
                  </span>
                </button>
              ))}
          </div>
        </>
      )}
    </section>
  );
}
