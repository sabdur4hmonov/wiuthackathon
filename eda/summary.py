"""Extract measured video metadata and optional validated result/track summaries."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from demo_api.validation import sanitized_prediction
from demo_api.harness_adapter import _reject_constant, _unique_safe_object

MAX_EDA_FRAMES = 150_000


def video_metadata(path: Path, capture_factory=None) -> dict:
    import cv2

    capture = (capture_factory or cv2.VideoCapture)(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Video could not be decoded.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        reported = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(reported) or reported < 0:
            raise ValueError("Video timing could not be established.")
        count = 0
        width = height = 0
        last_ms = -1.0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            count += 1
            if count > MAX_EDA_FRAMES:
                raise ValueError("Video exceeds the bounded EDA frame limit.")
            current_height, current_width = frame.shape[:2]
            if current_width <= 0 or current_height <= 0 or (width and (width, height) != (current_width, current_height)):
                raise ValueError("Video resolution could not be established consistently.")
            width, height = current_width, current_height
            position_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            if not math.isfinite(position_ms) or position_ms < last_ms:
                raise ValueError("Video timestamps could not be established.")
            last_ms = position_ms
        if count == 0 or (count > 1 and last_ms <= 0):
            raise ValueError("Video contains no verifiable decoded frames.")
        duration = max(count / fps, last_ms / 1000 + 1 / fps)
        return {
            "width": width,
            "height": height,
            "fps": float(fps),
            "decoded_frame_count": count,
            "reported_frame_count": int(reported) if reported > 0 and abs(reported - count) <= 1 else None,
            "decoded_duration_sec": duration,
        }
    finally:
        capture.release()


def event_distribution(source: Path, filename: str) -> dict[str, int]:
    raw = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_unique_safe_object, parse_constant=_reject_constant)
    clean = sanitized_prediction(raw, filename)
    return dict(sorted(Counter(event[2] for event in clean["videos"][filename]["events"]).items()))


def track_hooks(source: Path, filename: str, duration_sec: float) -> dict:
    """Optional real-track sidecar: normalized ground points, not invented boxes."""
    raw = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_unique_safe_object, parse_constant=_reject_constant)
    if not isinstance(raw, dict) or set(raw) != {"video", "points"} or raw["video"] != filename or not isinstance(raw["points"], list):
        raise ValueError("Invalid track sidecar.")
    tracks: dict[int, list[list[float]]] = defaultdict(list)
    bins: dict[int, set[int]] = defaultdict(set)
    for point in raw["points"]:
        if not isinstance(point, dict) or set(point) != {"t_sec", "track_id", "x", "y"}:
            raise ValueError("Invalid track observation.")
        t, track_id, x, y = (point[key] for key in ("t_sec", "track_id", "x", "y"))
        if (type(track_id) is not int or track_id < 0 or
            any(type(value) not in (int, float) or not math.isfinite(value) for value in (t, x, y)) or
            not 0 <= t <= duration_sec or not 0 <= x <= 1 or not 0 <= y <= 1):
            raise ValueError("Invalid track observation.")
        tracks[track_id].append([float(t), float(x), float(y)])
        bins[int(t // 5)].add(track_id)
    return {
        "trajectories": [{"track_id": track_id, "points": sorted(points)} for track_id, points in sorted(tracks.items())],
        "unique_tracks_per_5s": [[index * 5, len(track_ids)] for index, track_ids in sorted(bins.items())],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evidence-only EDA; no data is bundled.")
    parser.add_argument("--video", type=Path, help="Authorized local MP4")
    parser.add_argument("--pred", type=Path, help="Optional sanitized prediction JSON")
    parser.add_argument("--tracks", type=Path, help="Optional measured track sidecar JSON")
    args = parser.parse_args()
    if args.video is None:
        print("real data required: provide an authorized --video")
        return 2
    try:
        metadata = video_metadata(args.video)
        report = {
            "source_status": "local_video_provenance_unverified",
            "video": args.video.name,
            "metadata": metadata,
            "event_distribution": event_distribution(args.pred, args.video.name) if args.pred else None,
            "track_hooks": track_hooks(args.tracks, args.video.name, metadata["decoded_duration_sec"]) if args.tracks else None,
        }
    except Exception:
        print("EDA failed: real decodable data and valid optional sidecars are required.")
        return 1
    print(json.dumps(report, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
