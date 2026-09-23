import test from "node:test";
import assert from "node:assert/strict";
import { OFFICIAL_CLASSES, parsePredictions, sourceLabel, type PredictionSource } from "./predictions.ts";
import { buildTimeline } from "./timeline.ts";
import { prepareRiskSeries, riskAtTime } from "./risk.ts";

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
    assert.match(sourceLabel(sampleResult.value.source), /SAMPLE MODEL OUTPUT/);
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
