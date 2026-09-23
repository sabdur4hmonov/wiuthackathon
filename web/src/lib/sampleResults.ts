import { parsePredictions, type ParsedPredictions } from "./predictions.ts";

export interface SampleEntry {
  id: string;
  label: string;
  filename: string;
  predictionUrl: string;
  videoUrl?: string;
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function localAsset(url: unknown, extension: string): url is string {
  return typeof url === "string" &&
    new RegExp(`^/samples/[a-zA-Z0-9_./-]+\\.${extension}$`).test(url) &&
    !url.includes("..") && !url.includes("//");
}

export function parseSampleCatalog(raw: unknown): SampleEntry[] {
  if (!record(raw) || Object.keys(raw).join() !== "samples" || !Array.isArray(raw.samples)) {
    throw new Error("Sample catalog is invalid.");
  }
  const ids = new Set<string>();
  return raw.samples.map((item: unknown) => {
    if (!record(item) || Object.keys(item).some((key) => !["id", "label", "filename", "predictionUrl", "videoUrl"].includes(key)) ||
        typeof item.id !== "string" || !/^[a-z0-9-]+$/.test(item.id) || ids.has(item.id) ||
        typeof item.label !== "string" || !item.label.trim() || item.label.length > 100 ||
        typeof item.filename !== "string" || !/^[^/\\:\x00-\x1f]+\.mp4$/i.test(item.filename) ||
        !localAsset(item.predictionUrl, "json") ||
        (item.videoUrl !== undefined && !localAsset(item.videoUrl, "mp4"))) {
      throw new Error("Sample catalog contains an invalid entry.");
    }
    ids.add(item.id);
    return {
      id: item.id,
      label: item.label,
      filename: item.filename,
      predictionUrl: item.predictionUrl,
      ...(item.videoUrl === undefined ? {} : { videoUrl: item.videoUrl }),
    };
  });
}

export function parseValidatedSample(raw: unknown, entry: SampleEntry): ParsedPredictions {
  if (!record(raw) || Object.keys(raw).some((key) => !["team", "videos"].includes(key)) ||
      !record(raw.videos) || Object.keys(raw.videos).length !== 1 ||
      !Object.hasOwn(raw.videos, entry.filename)) {
    throw new Error("Sample predictions must contain only sanitized data for the named video.");
  }
  if (raw.team !== undefined &&
      (typeof raw.team !== "string" || raw.team.length > 100 || /[/\\:\x00-\x1f\x7f]/.test(raw.team))) {
    throw new Error("Sample predictions contain an invalid team name.");
  }
  const video = raw.videos[entry.filename];
  if (!record(video) || Object.keys(video).some((key) => !["events", "risk"].includes(key))) {
    throw new Error("Sample predictions contain unexpected fields.");
  }
  const parsed = parsePredictions(raw, { kind: "sample", label: entry.label });
  if (!parsed.ok) throw new Error(`Sample prediction validation failed: ${parsed.errors.join("; ")}`);
  return parsed.value;
}

export async function fetchSampleCatalog(signal?: AbortSignal): Promise<SampleEntry[]> {
  const response = await fetch("/samples/catalog.json", { signal });
  if (!response.ok) throw new Error("Sample catalog is unavailable.");
  return parseSampleCatalog(await response.json());
}

export async function fetchValidatedSample(entry: SampleEntry, signal?: AbortSignal): Promise<ParsedPredictions> {
  const response = await fetch(entry.predictionUrl, { signal });
  if (!response.ok) throw new Error("Sample predictions are unavailable.");
  return parseValidatedSample(await response.json(), entry);
}
