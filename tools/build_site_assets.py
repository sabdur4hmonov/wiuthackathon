"""Build site_assets/ for the team website from the real sample clips.

    python tools/build_site_assets.py [--skip-video]

Needs the perception cache (Stage 1 run on samples/sample_00*.mp4) and
predictions_samples.json (events + risk curves from the harness). Writes
site_assets/ -- see site_assets/README.md for every file. --skip-video
rebuilds everything except the (slow: full 4K decode) annotated videos.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))
sys.path.insert(0, str(ROOT / "tools"))

from src import cache as cache_mod  # noqa: E402
from src import pipeline  # noqa: E402
from src.budget import Budget, probe_duration  # noqa: E402
from src.config import perception_config_hash  # noqa: E402
from src.rules.util import box_heights, sustained_speed  # noqa: E402
from src.rules.zoneindex import zone_index  # noqa: E402
from src.tracks import COL  # noqa: E402

OUT = ROOT / "site_assets"
CLIPS = ["sample_001", "sample_002", "sample_003", "sample_004"]
COCO = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
WEB_WIDTH = 854

# Honest failure cases, chosen from the labelling review and CP3/CP4 analysis.
FAILURES = [
    {"id": "moped_rider", "clip": "sample_002", "start": 169.4, "end": 175.9,
     "title": "A moped rider flagged as a jaywalker (CP3)",
     "what": "The full-rate CP3 pipeline reported a jaywalking event here. It is a moped "
             "rider: the detector boxed the person but not the moped, so the 'person inside "
             "a vehicle box' rider test had nothing to test against.",
     "status": "The keyframe pipeline no longer reports it -- but only because fewer samples "
               "fall past the 2 s minimum, not because it now understands riders. The same "
               "miss-boxed rider can still fire."},
    {"id": "far_kerb_pedestrians", "clip": "sample_001", "start": 173.7, "end": 180.2,
     "title": "Pedestrians at the far kerb are too small to judge",
     "what": "People on the far side of the avenue are under ~30 px tall at 960 px. "
             "Their ground point jitters across the kerb line by more than the one-body-height "
             "margin the jaywalking rule demands, so the rule is either silent (as here) or "
             "fires on someone standing on the pavement.",
     "status": "Kept quiet on purpose (biased toward not firing). A real far-side "
               "jaywalker can be missed."},
    {"id": "platoon_aliasing", "clip": "sample_002", "start": 44.0, "end": 56.0,
     "title": "Why wrong-way detection is off at keyframe rate",
     "what": "Stage 1 samples one frame every 0.5 s. A platoon of cars moving about one "
             "car-gap per sample aliases: the tracker hops back to the car behind each "
             "sample, and the track drifts BACKWARDS (150-200 px/s the wrong way, while "
             "the real cars move at 550-680 px/s). That produced 11-13 false wrong-way "
             "events over the four clips.",
     "status": "wrong_way and congestion only run when samples are <= 0.2 s apart "
               "(the optional dense mode); at keyframe rate they stay silent."},
]


def _tracks_zones(clip: str):
    video = ROOT / "samples" / f"{clip}.mp4"
    tracks = cache_mod.load(cache_mod.video_hash(video), perception_config_hash())
    if tracks is None:
        raise SystemExit(f"no cached tracks for {video}: run Stage 1 first")
    b = Budget(duration=tracks.duration, fps=tracks.fps, n_frames=tracks.n_frames, t0=0.0)
    zones = pipeline._align_zones(str(video), tracks, pipeline._get_zones(tracks, False), b, False)
    return video, tracks, zones


def _frame(video: Path, t: float, width: int = 960) -> np.ndarray:
    from src.avdecode import KeyframeReader

    with KeyframeReader(video, 29.97, "NONKEY", width) as r:
        best = None
        for idx, img in r.frames():
            best = img
            if idx / 29.97 >= t:
                break
    return best


def eda(clip: str, video: Path, tracks, zones, out: Path) -> dict:
    import cv2

    d = tracks.data
    fps = tracks.fps
    t = d[:, COL["t_sec"]]
    cls = d[:, COL["cls"]].astype(int)
    grid = np.unique(t)
    counts = {"t": [round(float(x), 2) for x in grid]}
    pos = np.searchsorted(grid, t)
    for k, name in COCO.items():
        c = np.bincount(pos[cls == k], minlength=grid.size)
        if c.sum():
            counts[name] = c.tolist()
    (out / f"{clip}_object_counts.json").write_text(json.dumps(counts))

    veh = tracks.vehicles()
    vi = zone_index(veh, zones)
    groups = {}
    for li, lane in enumerate(zones.lanes):
        groups.setdefault(lane.direction_group or "other", []).append(li)
    vpos = np.searchsorted(grid, veh.data[:, COL["t_sec"]])
    dens = {"t": counts["t"]}
    for gname, members in groups.items():
        m = np.isin(vi.lane_idx, members)
        dens[gname] = np.bincount(vpos[m], minlength=grid.size).tolist()
    (out / f"{clip}_density.json").write_text(json.dumps(dens))

    # Motion heatmap: ground points of MOVING objects, over a real frame.
    base = _frame(video, tracks.duration / 2)
    h, w = base.shape[:2]
    scale = w / tracks.width
    moving = sustained_speed(tracks) > 0.5 * box_heights(d)
    gx, gy = d[moving, COL["gx"]] * scale, d[moving, COL["gy"]] * scale
    heat, _, _ = np.histogram2d(gy, gx, bins=[h // 8, w // 8], range=[[0, h], [0, w]])
    heat = cv2.GaussianBlur(heat.astype(np.float32), (0, 0), 1.5)
    heat = cv2.resize(heat / max(heat.max(), 1e-6), (w, h))
    colour = cv2.applyColorMap((np.sqrt(heat) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    blend = np.where(heat[..., None] > 0.02, cv2.addWeighted(base, 0.35, colour, 0.65, 0), base // 2)
    cv2.imwrite(str(out / f"{clip}_motion_heatmap.jpg"), blend, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # Trajectories coloured by heading, lanes and their authored directions.
    img = (base * 0.55).astype(np.uint8)
    ids = d[:, COL["track_id"]].astype(int)
    for tid in np.unique(ids[ids >= 0]):
        m = np.flatnonzero(ids == tid)
        if m.size < 4:
            continue
        pts = np.stack([d[m, COL["gx"]], d[m, COL["gy"]]], 1) * scale
        if cls[m[0]] == 0:
            col = (230, 230, 230)
        else:
            dx, dy = pts[-1] - pts[0]
            hue = int((np.degrees(np.arctan2(dy, dx)) % 360) / 2)
            col = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
        cv2.polylines(img, [pts.astype(np.int32)], False, col, 1, cv2.LINE_AA)
    for lane in zones.lanes:
        poly = (np.asarray(lane.polygon, np.float32) * scale).astype(np.int32)
        cv2.polylines(img, [poly], True, (0, 255, 255), 1, cv2.LINE_AA)
        c = poly.mean(0)
        dvec = np.asarray(lane.direction, float)
        cv2.arrowedLine(img, tuple(int(v) for v in c), tuple(int(v) for v in c + 60 * dvec),
                        (0, 255, 255), 2, tipLength=0.35)
    cv2.imwrite(str(out / f"{clip}_trajectories_lanes.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])

    return {"clip": clip, "duration_sec": round(tracks.duration, 1),
            "tracks": int(len(np.unique(ids[ids >= 0]))),
            "detections_per_sample": round(len(d) / max(1, grid.size), 1),
            "by_class_tracks": {COCO.get(k, str(k)): int(len(np.unique(ids[(cls == k) & (ids >= 0)])))
                                for k in COCO},
            "pose": tracks.pose}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-video", action="store_true")
    args = ap.parse_args()
    import app as demo_app   # demo/app.py: the same drawing as the live demo

    pred = json.loads((ROOT / "predictions_samples.json").read_text())["videos"]
    for sub in ("videos", "events", "risk", "event_clips", "failures", "eda"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    summary = []
    for clip in CLIPS:
        video, tracks, zones = _tracks_zones(clip)
        dur, fps, n = probe_duration(video)
        entry = pred[f"{clip}.mp4"]
        events, risk = entry["events"], entry["risk"]
        (OUT / "events" / f"{clip}_events.json").write_text(json.dumps(
            {"clip": f"{clip}.mp4", "duration_sec": round(dur, 3), "events": events}, indent=1))
        # 5 Hz is plenty for a web chart (the harness curve is per frame).
        step = max(1, int(round(fps / 5)))
        (OUT / "risk" / f"{clip}_risk.json").write_text(json.dumps(
            {"clip": f"{clip}.mp4", "alarm_threshold": 0.5, "risk": risk[::step]}))
        summary.append(eda(clip, video, tracks, zones, OUT / "eda"))
        print(f"{clip}: events/risk/eda written", flush=True)
        if args.skip_video:
            continue
        demo_app.OUT_WIDTH = WEB_WIDTH
        part_a = {"tracks": tracks, "zones": zones, "events": events}
        tmp = OUT / "videos" / f"_{clip}_tmp"
        tmp.mkdir(exist_ok=True)
        annotated, clips = demo_app.render(str(video), tmp, part_a, risk, fps, print)
        final = OUT / "videos" / f"{clip}_annotated.mp4"
        Path(annotated).replace(final)
        for i, c in enumerate(clips):
            Path(c).replace(OUT / "event_clips" / f"{clip}_{Path(c).name}")
        for f in FAILURES:
            if f["clip"] == clip:
                demo_app._cut(final, OUT / "failures" / f"{f['id']}.mp4", f["start"], f["end"])
        for p in tmp.iterdir():
            p.unlink()
        tmp.rmdir()
        print(f"{clip}: annotated video and clips written", flush=True)
    (OUT / "failures" / "failures.json").write_text(json.dumps(FAILURES, indent=1))
    (OUT / "eda" / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
