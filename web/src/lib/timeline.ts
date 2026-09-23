import {
  OFFICIAL_CLASSES,
  type EventLabel,
  type EventSegment,
  type RiskPoint,
} from "./predictions.ts";

export interface TimelineLane {
  label: EventLabel;
  events: EventSegment[];
}

export interface TimelineData {
  durationSec: number;
  lanes: TimelineLane[];
}

/** Keeps individual events intact, including simultaneous different classes. */
export function buildTimeline(
  events: readonly EventSegment[],
  risk: readonly RiskPoint[] = [],
  durationHint = 0,
): TimelineData {
  let durationSec = Math.max(1, durationHint);
  for (const event of events) durationSec = Math.max(durationSec, event[1]);
  for (const point of risk) durationSec = Math.max(durationSec, point[0]);
  const lanes: TimelineLane[] = OFFICIAL_CLASSES.flatMap((label) => {
    const matching = events
      .filter((event) => event[2] === label)
      .map((event) => [...event] as EventSegment)
      .sort((a, b) => a[0] - b[0]);
    return matching.length > 0 ? [{ label, events: matching }] : [];
  });
  return { durationSec, lanes };
}
