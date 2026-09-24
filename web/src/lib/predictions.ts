/** The organizer's prediction format, plus source information held OUTSIDE it. */
export const OFFICIAL_CLASSES = [
  "accident",
  "near_miss",
  "red_light",
  "wrong_way",
  "illegal_u_turn",
  "stopped_vehicle",
  "jaywalking",
  "failure_to_yield",
  "illegal_turn",
  "solid_line_crossing",
  "stop_line",
  "congestion",
  "road_obstacle",
  "fire_smoke",
] as const;

export type EventLabel = (typeof OFFICIAL_CLASSES)[number];
export type EventSegment = [startSec: number, endSec: number, label: EventLabel];
export type RiskPoint = [tSec: number, score: number];

export interface VideoPrediction {
  events: EventSegment[];
  risk: RiskPoint[];
}

export interface PredictionDocument {
  team?: string;
  videos: Record<string, VideoPrediction>;
}

/** Provenance is never read from organizer JSON or inferred from its filename. */
export type PredictionSource =
  | { kind: "fixture"; label: string }
  | { kind: "sample"; label: string }
  | { kind: "upload"; label: string };

export interface ParsedPredictions {
  document: PredictionDocument;
  source: PredictionSource;
  warnings: string[];
}

export type ParseResult =
  | { ok: true; value: ParsedPredictions }
  | { ok: false; errors: string[] };

const classSet: ReadonlySet<string> = new Set(OFFICIAL_CLASSES);
const unsafeKeys = new Set(["__proto__", "constructor", "prototype"]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function unsafeKeyAt(value: unknown, path = "prediction", seen = new WeakSet<object>()): string | null {
  if (typeof value !== "object" || value === null || seen.has(value)) return null;
  seen.add(value);
  for (const [key, child] of Object.entries(value)) {
    if (unsafeKeys.has(key)) return `${path}[${JSON.stringify(key)}]`;
    const nested = unsafeKeyAt(child, `${path}[${JSON.stringify(key)}]`, seen);
    if (nested) return nested;
  }
  return null;
}

/** Mirrors evaluate.py's event rules; adds finite-number checks for display. */
export function parsePredictions(
  input: unknown,
  source: PredictionSource,
): ParseResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  if (!isRecord(input) || !isRecord(input.videos)) {
    return { ok: false, errors: ['Expected an object with a "videos" object.'] };
  }
  const unsafePath = unsafeKeyAt(input);
  if (unsafePath) return { ok: false, errors: [`Unsafe object key at ${unsafePath}.`] };

  let team: string | undefined;
  if (input.team === undefined) {
    warnings.push('Missing optional "team" name.');
  } else if (typeof input.team === "string") {
    team = input.team;
  } else {
    errors.push('"team" must be a string when present.');
  }

  const videos: Record<string, VideoPrediction> = Object.create(null) as Record<string, VideoPrediction>;
  for (const [filename, rawVideo] of Object.entries(input.videos)) {
    const prefix = `videos[${JSON.stringify(filename)}]`;
    if (!filename || filename.includes("/") || filename.includes("\\")) {
      errors.push(`${prefix}: key must be a file name, not a path.`);
    }
    if (!isRecord(rawVideo) || !Array.isArray(rawVideo.events)) {
      errors.push(`${prefix}: "events" must be an array.`);
      continue;
    }

    const events: EventSegment[] = [];
    rawVideo.events.forEach((rawEvent: unknown, index: number) => {
      const path = `${prefix}.events[${index}]`;
      if (!Array.isArray(rawEvent) || rawEvent.length !== 3) {
        errors.push(`${path}: expected [start_sec, end_sec, label].`);
        return;
      }
      const [start, end, label] = rawEvent;
      if (!isFiniteNumber(start) || !isFiniteNumber(end)) {
        errors.push(`${path}: start and end must be finite numbers.`);
        return;
      }
      // The organizer requires strict inequality; a zero-length event is invalid.
      if (!(0 <= start && start < end)) {
        errors.push(`${path}: require 0 <= start_sec < end_sec.`);
      }
      if (typeof label !== "string" || !classSet.has(label)) {
        errors.push(`${path}: unknown official class ${JSON.stringify(label)}.`);
        return;
      }
      events.push([start, end, label as EventLabel]);
    });

    const byClass = new Map<EventLabel, EventSegment[]>();
    events.forEach((event) => {
      const group = byClass.get(event[2]) ?? [];
      group.push(event);
      byClass.set(event[2], group);
    });
    byClass.forEach((group, label) => {
      group.sort((a, b) => a[0] - b[0]);
      for (let i = 1; i < group.length; i += 1) {
        if (group[i][0] < group[i - 1][1]) {
          errors.push(`${prefix}: overlapping ${label} segments are invalid.`);
        }
      }
    });

    const risk: RiskPoint[] = [];
    if (rawVideo.risk === undefined) {
      warnings.push(`${prefix}: no risk curve; Part B is unavailable.`);
    } else if (!Array.isArray(rawVideo.risk)) {
      errors.push(`${prefix}.risk: expected an array when present.`);
    } else {
      let previousTime = -1;
      rawVideo.risk.forEach((rawPoint: unknown, index: number) => {
        const path = `${prefix}.risk[${index}]`;
        if (
          !Array.isArray(rawPoint) ||
          rawPoint.length !== 2 ||
          !isFiniteNumber(rawPoint[0]) ||
          !isFiniteNumber(rawPoint[1])
        ) {
          errors.push(`${path}: expected [finite time, finite score].`);
          return;
        }
        const [time, score] = rawPoint;
        if (time < previousTime || time < 0) {
          errors.push(`${path}: timestamps must be non-negative and non-decreasing.`);
        }
        if (score < 0 || score > 1) {
          errors.push(`${path}: score must be in [0, 1].`);
        }
        previousTime = time;
        risk.push([time, score]);
      });
      if (risk.length === 0) {
        warnings.push(`${prefix}: empty risk curve; Part B is unavailable.`);
      }
    }
    videos[filename] = { events, risk };
  }

  if (errors.length > 0) return { ok: false, errors };
  return {
    ok: true,
    value: { document: { team, videos }, source, warnings },
  };
}

export function sourceLabel(source: PredictionSource): string {
  if (source.kind === "fixture") return `ILLUSTRATIVE FIXTURE · ${source.label}`;
  if (source.kind === "sample") return `PREDICTION FORMAT/SCHEMA VALIDATED · REAL SAMPLE · DETECTION ACCURACY NOT VALIDATED · ${source.label}`;
  return `LIVE UPLOAD INFERENCE · ${source.label}`;
}

export function isRealModelOutput(source: PredictionSource): boolean {
  return source.kind !== "fixture";
}
