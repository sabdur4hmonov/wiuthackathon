import { useEffect, useRef, useState } from "react";
import type { EventSegment, PredictionSource } from "../lib/predictions.ts";
import { sourceLabel } from "../lib/predictions.ts";

/** Optional visualization data. This is NOT part of official predictions.json. */
export interface TrackBox {
  tSec: number;
  trackId: number;
  label: string;
  /** Normalized image coordinates, each within [0, 1]. */
  x: number;
  y: number;
  width: number;
  height: number;
}

interface Props {
  videoUrl?: string;
  filename?: string;
  events: readonly EventSegment[];
  source: PredictionSource;
  boxes?: readonly TrackBox[];
  seekToSec?: number;
  seekToken?: number;
  onTimeChange?: (seconds: number) => void;
  onDurationChange?: (seconds: number) => void;
}

export function AnnotatedPlayer({
  videoUrl,
  filename,
  events,
  source,
  boxes = [],
  seekToSec,
  seekToken,
  onTimeChange,
  onDurationChange,
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [time, setTime] = useState(0);

  useEffect(() => {
    if (videoRef.current && seekToSec !== undefined) {
      videoRef.current.currentTime = seekToSec;
    }
  }, [seekToSec, seekToken]);

  const active = events.filter(([start, end]) => start <= time && time < end);
  const visibleBoxes = boxes.filter((box) => Math.abs(box.tSec - time) < 0.15);

  return (
    <section className="panel player-panel" aria-label="Annotated video player">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Synchronized playback</span>
          <h3>Video review</h3>
        </div>
        <span className="count-pill">{filename ?? "No video attached"}</span>
      </div>
      <p className="data-origin">{sourceLabel(source)}</p>
      {videoUrl ? (
        <div className="video-stage">
          <video
            ref={videoRef}
            src={videoUrl}
            controls
            playsInline
            onLoadedMetadata={(event) => {
              const seconds = event.currentTarget.duration;
              if (Number.isFinite(seconds) && seconds > 0) onDurationChange?.(seconds);
            }}
            onDurationChange={(event) => {
              const seconds = event.currentTarget.duration;
              if (Number.isFinite(seconds) && seconds > 0) onDurationChange?.(seconds);
            }}
            onTimeUpdate={(event) => {
              const nextTime = event.currentTarget.currentTime;
              setTime(nextTime);
              onTimeChange?.(nextTime);
            }}
          >
            Your browser does not support video playback.
          </video>
          <div className="video-overlay" aria-hidden="true">
            <div className="active-events">
              {active.map(([start, end, label]) => (
                <span key={`${label}-${start}-${end}`}>{label.replaceAll("_", " ")}</span>
              ))}
            </div>
            {visibleBoxes.map((box) => (
              <div
                className="track-box"
                key={`${box.trackId}-${box.tSec}`}
                style={{
                  left: `${box.x * 100}%`,
                  top: `${box.y * 100}%`,
                  width: `${box.width * 100}%`,
                  height: `${box.height * 100}%`,
                }}
              >
                <span>{box.label} #{box.trackId}</span>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="player-placeholder">
          <span className="placeholder-glyph">▶</span>
          <strong>Playback attaches here</strong>
          <p>Upload a video or add a real sample asset. No frames or boxes are simulated.</p>
        </div>
      )}
      <p className="component-footnote">
        Event labels come from prediction intervals. Object boxes appear only when a separate track sidecar is provided.
      </p>
    </section>
  );
}
