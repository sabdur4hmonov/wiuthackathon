import test from "node:test";
import assert from "node:assert/strict";
import { OFFICIAL_CLASSES, parsePredictions, sourceLabel, type PredictionSource } from "./predictions.ts";
import { buildTimeline } from "./timeline.ts";
import { prepareRiskSeries, riskAtTime } from "./risk.ts";
import { initialUploadState, uploadReducer } from "./uploadSession.ts";
import type { JobView } from "./api.ts";
import { parseSampleCatalog, parseValidatedSample } from "./sampleResults.ts";

const fixture: PredictionSource = { kind: "fixture", label: "invented test data" };
const sample: PredictionSource = { kind: "sample", label: "measured clip 1" };
const base = (events: unknown[] = [], risk: unknown = []) => ({ team: "test", videos: { "clip.mp4": { events, risk } } });

test("all fourteen official labels survive parsing", () => {
  const events = OFFICIAL_CLASSES.map((label, index) => [index, index + 0.5, label]);
  const parsed = parsePredictions(base(events), fixture);
  assert.equal(parsed.ok, true);
  if (parsed.ok) assert.equal(parsed.value.document.videos["clip.mp4"].events.length, 14);
});

test("rejects malformed and same-class overlapping intervals", () => {
  assert.equal(parsePredictions(base([[1, 1, "accident"]]), fixture).ok, false);
  assert.equal(parsePredictions(base([[0, 2, "accident"], [1, 3, "accident"]]), fixture).ok, false);
  assert.equal(parsePredictions(base([[0, 1, "other"]]), fixture).ok, false);
});

test("different classes can overlap and zero risk is valid", () => {
  const parsed = parsePredictions(base([[0, 2, "accident"], [1, 3, "near_miss"]], [[0, 0], [1, 0]]), fixture);
  assert.equal(parsed.ok, true);
  if (parsed.ok) assert.deepEqual(parsed.value.document.videos["clip.mp4"].risk, [[0, 0], [1, 0]]);
});

test("risk is optional but invalid values are rejected", () => {
  const noRisk = parsePredictions({ videos: { "clip.mp4": { events: [] } } }, sample);
  assert.equal(noRisk.ok, true);
  if (noRisk.ok) assert.match(noRisk.value.warnings.join(" "), /no risk curve/);
  assert.equal(parsePredictions(base([], [[0, 0.5], [0.2, 1.1]]), sample).ok, false);
  assert.equal(parsePredictions(base([], [[1, 0.5], [0, 0.2]]), sample).ok, false);
  assert.equal(parsePredictions(base([], [[0, Number.NaN]]), sample).ok, false);
});

test("server paths and wrong top-level shape are refused", () => {
  assert.equal(parsePredictions({ videos: { "C:\\private\\clip.mp4": { events: [] } } }, sample).ok, false);
  assert.equal(parsePredictions({ events: [] }, sample).ok, false);
});

test("prototype keys are rejected without losing normal MP4 filenames", () => {
  for (const unsafe of ["__proto__", "constructor", "prototype"]) {
    const input = JSON.parse(`{"videos":{"${unsafe}":{"events":[]}}}`) as unknown;
    assert.equal(parsePredictions(input, sample).ok, false);
    const nested = JSON.parse(`{"videos":{"clip.mp4":{"events":[],"${unsafe}":{}}}}`) as unknown;
    assert.equal(parsePredictions(nested, sample).ok, false);
  }
  const valid = parsePredictions({ videos: { "__proto__.mp4": { events: [] } } }, sample);
  assert.equal(valid.ok, true);
  if (valid.ok) assert.ok(Object.hasOwn(valid.value.document.videos, "__proto__.mp4"));
});

test("provenance is external to official JSON", () => {
  const input = base();
  const fixtureResult = parsePredictions(input, fixture);
  const sampleResult = parsePredictions(input, sample);
  assert.equal(fixtureResult.ok, true);
  assert.equal(sampleResult.ok, true);
  if (fixtureResult.ok && sampleResult.ok) {
    assert.deepEqual(fixtureResult.value.document, sampleResult.value.document);
    assert.match(sourceLabel(fixtureResult.value.source), /ILLUSTRATIVE/);
    assert.match(sourceLabel(sampleResult.value.source), /VALIDATED REAL SAMPLE/);
  }
});

test("timeline retains separate lanes and long arrays", () => {
  const events: Array<[number, number, "accident" | "near_miss"]> = [[0, 2, "accident"], [1, 3, "near_miss"]];
  const risk = Array.from({ length: 100_000 }, (_, index) => [index / 100, 0] as [number, number]);
  const timeline = buildTimeline(events, risk);
  assert.equal(timeline.lanes.length, 2);
  assert.equal(timeline.durationSec, 999.99);
});

test("risk reduction preserves a sharp spike and lookup is causal", () => {
  const input = Array.from({ length: 2000 }, (_, index) => [index / 10, index === 913 ? 1 : 0] as [number, number]);
  const reduced = prepareRiskSeries(input);
  assert.ok(reduced.length <= 610);
  assert.ok(reduced.some(([time, score]) => time === 91.3 && score === 1));
  assert.equal(riskAtTime(input, 91.29), 0);
  assert.equal(riskAtTime(input, 91.3), 1);
  assert.equal(riskAtTime([[5, 0.8]], 4), null);
});

test("selection cannot replace the video bound to an existing job", () => {
  const a = new File(["a"], "a.mp4", { type: "video/mp4" });
  const b = new File(["b"], "b.mp4", { type: "video/mp4" });
  const jobA: JobView = { job_id: "a", filename: "a.mp4", status: "queued", result_available: false, message: "Queued" };
  let state = uploadReducer(initialUploadState, { type: "select", file: a });
  state = uploadReducer(state, { type: "start", id: 1, file: a });
  state = uploadReducer(state, { type: "created", id: 1, job: jobA });
  state = uploadReducer(state, { type: "select", file: b });
  assert.equal(state.selectedFile, b);
  assert.equal(state.run?.file, a);
  assert.equal(state.run?.job?.filename, "a.mp4");
});

test("stale job A status, result and duration cannot replace job B", () => {
  const a = new File(["a"], "a.mp4");
  const b = new File(["b"], "b.mp4");
  const jobA: JobView = { job_id: "a", filename: "a.mp4", status: "completed", result_available: true, message: "Done" };
  const jobB: JobView = { job_id: "b", filename: "b.mp4", status: "running", result_available: false, message: "Running" };
  const parsed = parsePredictions({ videos: { "a.mp4": { events: [], risk: [] } } }, { kind: "upload", label: "A" });
  assert.equal(parsed.ok, true);
  if (!parsed.ok) return;
  let state = uploadReducer(initialUploadState, { type: "start", id: 1, file: a });
  state = uploadReducer(state, { type: "created", id: 1, job: jobA });
  state = uploadReducer(state, { type: "start", id: 2, file: b });
  state = uploadReducer(state, { type: "created", id: 2, job: jobB });
  state = uploadReducer(state, { type: "status", id: 1, job: jobA });
  state = uploadReducer(state, { type: "result", id: 1, jobId: "a", result: parsed.value });
  state = uploadReducer(state, { type: "duration", id: 1, seconds: 73 });
  assert.equal(state.run?.file, b);
  assert.equal(state.run?.job?.job_id, "b");
  assert.equal(state.run?.result, null);
  assert.equal(state.run?.durationSec, undefined);
});

test("actual media duration scales empty predictions; unavailable duration retains fallback", () => {
  const file = new File(["video"], "clip.mp4");
  let state = uploadReducer(initialUploadState, { type: "start", id: 1, file });
  state = uploadReducer(state, { type: "duration", id: 1, seconds: 72.345 });
  assert.equal(state.run?.durationSec, 72.345);
  assert.equal(buildTimeline([], [], state.run?.durationSec).durationSec, 72.345);
  assert.equal(buildTimeline([[1, 2, "accident"]], [], state.run?.durationSec).durationSec, 72.345);
  state = uploadReducer(state, { type: "duration", id: 1, seconds: Number.POSITIVE_INFINITY });
  assert.equal(state.run?.durationSec, 72.345);
  assert.equal(buildTimeline([], []).durationSec, 1);
});

test("future real samples require a strict catalog and sanitized prediction document", () => {
  const [entry] = parseSampleCatalog({ samples: [{
    id: "road-01", label: "Verified road clip", filename: "road.mp4",
    predictionUrl: "/samples/road.json", videoUrl: "/samples/road.mp4",
  }] });
  const result = parseValidatedSample({ team: "wiut-cv", videos: { "road.mp4": {
    events: [[1.234, 2.345, "accident"]], risk: [[0.0001, 0.25]],
  } } }, entry);
  assert.equal(result.source.kind, "sample");
  assert.equal(result.document.videos["road.mp4"].events[0][0], 1.234);
  assert.throws(() => parseSampleCatalog({ samples: [{ ...entry, predictionUrl: "/samples/../private.json" }] }));
  assert.throws(() => parseValidatedSample({ videos: { "road.mp4": { events: [] } }, log: { path: "C:/private" } }, entry));
  assert.throws(() => parseValidatedSample({ videos: { "other.mp4": { events: [] } } }, entry));
  assert.throws(() => parseValidatedSample({ videos: { "road.mp4": { events: [], risk: [[0, 2]] } } }, entry));
});
