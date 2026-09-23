import type { RiskPoint } from "./predictions.ts";

/** Keeps endpoints and local highs/lows when a long curve is reduced for SVG. */
export function prepareRiskSeries(
  input: readonly RiskPoint[],
  maxPoints = 600,
): RiskPoint[] {
  if (input.length <= maxPoints) return input.map((point) => [...point]);
  const bucketCount = Math.max(1, Math.floor(maxPoints / 4));
  const bucketSize = Math.ceil(input.length / bucketCount);
  const selected = new Set<number>([0, input.length - 1]);

  for (let start = 0; start < input.length; start += bucketSize) {
    const end = Math.min(input.length, start + bucketSize);
    let minIndex = start;
    let maxIndex = start;
    for (let index = start + 1; index < end; index += 1) {
      if (input[index][1] < input[minIndex][1]) minIndex = index;
      if (input[index][1] > input[maxIndex][1]) maxIndex = index;
    }
    selected.add(start);
    selected.add(minIndex);
    selected.add(maxIndex);
    selected.add(end - 1);
  }
  return [...selected].sort((a, b) => a - b).map((index) => [...input[index]]);
}

export function riskAtTime(points: readonly RiskPoint[], timeSec: number): number | null {
  if (points.length === 0) return null;
  let low = 0;
  let high = points.length - 1;
  while (low <= high) {
    const mid = (low + high) >>> 1;
    if (points[mid][0] <= timeSec) low = mid + 1;
    else high = mid - 1;
  }
  return high < 0 ? null : points[high][1];
}
