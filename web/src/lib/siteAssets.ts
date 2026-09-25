import type { EventSegment, RiskPoint } from "./predictions.ts";

export interface EdaClipSummary {
  clip: string;
  duration_sec: number;
  tracks: number;
  detections_per_sample: number;
  by_class_tracks: Record<string, number>;
  pose: { reason: string; drift_px: number };
}

export interface FormatFindings {
  footage: {
    container: string;
    codec: string;
    resolution: string;
    fps: number;
    bitrate_mbps: number;
    clips: Record<string, { sec: number; frames: number; gb: number }>;
  };
  gop: { length_frames: number; length_sec: number; fixed: boolean; display_order: string; b_frames: boolean };
  measured_costs_x_realtime: { budget: string };
  decisions: Array<{ finding: string; decision: string }>;
}

export interface FailureCase {
  id: string;
  clip: string;
  start: number;
  end: number;
  title: string;
  what: string;
  status: string;
  videoUrl: string;
}

export interface TeamMember {
  name: string;
  role?: string;
  links: Array<{ label: string; url: string }>;
}

export interface TeamCredits {
  name: string;
  members: TeamMember[];
  attribution: string;
  source: string;
}

export interface RealSample {
  id: string;
  filename: string;
  durationSec: number;
  events: EventSegment[];
  risk: RiskPoint[];
  alarmThreshold: number;
  annotatedVideoUrl: string;
  heatmapUrl: string;
  trajectoriesUrl: string;
}

export interface SiteAssets {
  team: TeamCredits;
  eda: EdaClipSummary[];
  format: FormatFindings;
  failures: FailureCase[];
  samples: RealSample[];
}

const CLIPS = ["sample_001", "sample_002", "sample_003", "sample_004"] as const;

export function siteAssetUrl(path: string): string {
  return `${import.meta.env.BASE_URL}site_assets/${path}`;
}

async function json<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(siteAssetUrl(path), { signal });
  if (!response.ok) throw new Error(`Site asset unavailable: ${path}`);
  return response.json() as Promise<T>;
}

export async function loadSiteAssets(signal?: AbortSignal): Promise<SiteAssets> {
  const [team, eda, format, rawFailures, ...sampleDocuments] = await Promise.all([
    json<TeamCredits>("team.json", signal),
    json<EdaClipSummary[]>("eda/summary.json", signal),
    json<FormatFindings>("eda/format_findings.json", signal),
    json<Array<Omit<FailureCase, "videoUrl">>>("failures/failures.json", signal),
    ...CLIPS.flatMap((clip) => [
      json<{ clip: string; duration_sec: number; events: EventSegment[] }>(`events/${clip}_events.json`, signal),
      json<{ clip: string; alarm_threshold: number; risk: RiskPoint[] }>(`risk/${clip}_risk.json`, signal),
    ]),
  ]);

  if (!team.name.trim() || team.members.length === 0 || eda.length !== CLIPS.length || rawFailures.length === 0 || format.decisions.length === 0) {
    throw new Error("site_assets contains incomplete website evidence.");
  }

  const samples = CLIPS.map((clip, index): RealSample => {
    const events = sampleDocuments[index * 2] as { clip: string; duration_sec: number; events: EventSegment[] };
    const risk = sampleDocuments[index * 2 + 1] as { clip: string; alarm_threshold: number; risk: RiskPoint[] };
    if (events.clip !== `${clip}.mp4` || risk.clip !== `${clip}.mp4`) {
      throw new Error(`site_assets clip mismatch for ${clip}.`);
    }
    return {
      id: clip,
      filename: `${clip}.mp4`,
      durationSec: events.duration_sec,
      events: events.events,
      risk: risk.risk,
      alarmThreshold: risk.alarm_threshold,
      annotatedVideoUrl: siteAssetUrl(`videos/${clip}_annotated.mp4`),
      heatmapUrl: siteAssetUrl(`eda/${clip}_motion_heatmap.jpg`),
      trajectoriesUrl: siteAssetUrl(`eda/${clip}_trajectories_lanes.jpg`),
    };
  });

  return {
    team,
    eda,
    format,
    samples,
    failures: rawFailures.map((failure) => ({
      ...failure,
      videoUrl: siteAssetUrl(`failures/${failure.id}.mp4`),
    })),
  };
}
