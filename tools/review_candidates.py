"""Build a review page for candidate events: short clips, yes / no / fix.

    python tools/review_candidates.py samples/sample_00*.mp4 \
        --seed labels/gt_approx_cp3.json --out labels/review [--proxies]

then open labels/review/index.html in a browser (no server needed).

Candidates per clip, merged where they overlap in time:
  * every event the shipped pipeline predicts (solution.detect_events path,
    from the perception cache -- run Stage 1 first);
  * every event in --seed (e.g. the CP3 events);
  * RECALL PROBES: the same rules with relaxed thresholds (shorter, closer to
    the kerb, lower confidence), capped per clip. These are the near misses a
    labeller should look at so that a missed event can be caught without
    watching the whole clip.

For each candidate a ~H.264 clip (960 wide, every 2nd frame, the window
[start - 2 s, end + 2 s]) is cut from the ORIGINAL file -- browsers cannot play
the 10-bit 4:2:2 source -- with the source timestamp burned in, the crossing
polygons drawn, and the candidate's tracked boxes outlined.

--proxies also writes a full-length 960-wide H.264 proxy of each clip, so
missed events elsewhere can be found with tools/label.html (it decodes the
whole 4K file: ~1-3x realtime per clip).

The page exports ground_truth.json in evaluate.py's shape.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import cache as cache_mod  # noqa: E402
from src import pipeline  # noqa: E402
from src.budget import Budget, probe_duration  # noqa: E402
from src.config import perception_config_hash  # noqa: E402
from src.postprocess import finalise  # noqa: E402
from src.rules import frames_to_seconds, run_rules  # noqa: E402
from src.rules import jaywalking, stopped_vehicle  # noqa: E402
from src.thresholds import TH  # noqa: E402
from src.tracks import COL  # noqa: E402

PAD_SEC = 2.0
OUT_WIDTH = 960
PROBES_PER_CLIP = 8

# Recall probes: the shipped rules with every gate loosened toward firing.
RELAXED = {
    "jaywalking": (jaywalking.detect, dataclasses.replace(
        TH.jaywalking, min_duration_sec=1.0, kerb_margin_L=0.5, crossing_margin_L=0.25,
        min_confidence=0.35)),
    "stopped_vehicle": (stopped_vehicle.detect, dataclasses.replace(
        TH.stopped_vehicle, min_duration_sec=6.0, min_confidence=0.30)),
}


def _tracks_and_zones(video: Path):
    tracks = cache_mod.load(cache_mod.video_hash(video), perception_config_hash())
    if tracks is None:
        raise SystemExit(f"no cached tracks for {video}: run Stage 1 (solution.detect_events) first")
    budget = Budget(duration=tracks.duration, fps=tracks.fps, n_frames=tracks.n_frames, t0=0.0)
    zones = pipeline._align_zones(str(video), tracks, pipeline._get_zones(tracks, False),
                                  budget, False)
    return tracks, zones


def _overlap(a, b) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def candidates_for(video: Path, seed: dict) -> tuple[list[dict], object, object]:
    tracks, zones = _tracks_and_zones(video)
    fps, step = tracks.fps, tracks.frame_stride
    raw = run_rules(tracks, zones)
    shipped = finalise(raw, tracks.duration)
    ids_by_event = {}
    for r in raw:
        ids_by_event.setdefault(r.label, []).append((r.start, r.end, tuple(r.track_ids)))

    found: list[dict] = []

    def add(start, end, label, source, ids=()):
        for c in found:
            if c["label"] == label and _overlap((c["start"], c["end"]), (start, end)) > 0:
                c["start"], c["end"] = min(c["start"], start), max(c["end"], end)
                c["sources"] = sorted(set(c["sources"]) | {source})
                c["track_ids"] = sorted(set(c["track_ids"]) | set(ids))
                return
        found.append({"start": start, "end": end, "label": label, "sources": [source],
                      "track_ids": sorted(set(ids))})

    for s, e, lab in shipped:
        ids = [i for rs, re, ii in ids_by_event.get(lab, []) if _overlap((rs, re), (s, e)) > 0
               for i in ii]
        add(s, e, lab, "predicted", ids)
    for s, e, lab in seed.get(video.name, {}).get("events", []):
        add(s, e, lab, "seed")
    for lab, (fn, th) in RELAXED.items():
        segs = [frames_to_seconds(sg, lab, fps, step) for sg in fn(tracks, zones, th)]
        segs = [sg for sg in segs if not any(_overlap((sg.start, sg.end), (c["start"], c["end"])) > 0
                                             and c["label"] == lab for c in found)]
        segs.sort(key=lambda sg: sg.end - sg.start, reverse=True)
        for sg in segs[:PROBES_PER_CLIP]:
            add(sg.start, sg.end, lab, "recall probe", sg.track_ids)
    found.sort(key=lambda c: c["start"])
    return found, tracks, zones


def _draw(img, t_abs, tracks, zones, ids, scale):
    import cv2

    for poly in zones.crossings:
        pts = (np.asarray(poly.polygon, dtype=np.float32) * scale).astype(np.int32)
        cv2.polylines(img, [pts], True, (255, 200, 0), 1, cv2.LINE_AA)
    d = tracks.data
    near = np.abs(d[:, COL["t_sec"]] - t_abs) <= (tracks.frame_stride / tracks.fps) / 2 + 1e-3
    for i in np.flatnonzero(near & np.isin(d[:, COL["track_id"]].astype(int), list(ids))):
        x1, y1, x2, y2 = (d[i, [COL["x1"], COL["y1"], COL["x2"], COL["y2"]]] * scale).astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    label = f"t = {t_abs:7.2f} s"
    cv2.putText(img, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)


def render_clip(video: Path, start: float, end: float, out_mp4: Path, out_jpg: Path,
                tracks, zones, ids) -> float:
    """Cut [start, end] from the original file to a browser-playable H.264 clip.

    Every 2nd source frame, at exactly half the source rate, so playback time
    t in the clip is source time t0 + t. Returns t0, the source time of the
    first frame written (the review page adds it to the video's currentTime).
    """
    import av
    import cv2

    with av.open(str(video)) as src, av.open(str(out_mp4), "w") as dst:
        s = src.streams.video[0]
        s.thread_type = "AUTO"                      # full decode here: frame threads help
        tb, st = float(s.time_base), s.start_time or 0
        fps = float(s.average_rate)
        w = OUT_WIDTH
        h = int(round(s.codec_context.height * w / s.codec_context.width / 2)) * 2
        scale = w / s.codec_context.width
        o = dst.add_stream("libx264", rate=s.average_rate / 2)
        o.width, o.height, o.pix_fmt = w, h, "yuv420p"
        o.options = {"crf": "30", "preset": "veryfast"}
        src.seek(int(max(0.0, start) / tb) + st, stream=s, backward=True, any_frame=False)
        k, mid, t0 = 0, None, None
        for f in src.decode(s):
            t = (f.pts - st) * tb
            if t < start:
                continue
            if t > end:
                break
            k += 1
            if not k % 2:
                continue
            if t0 is None:
                t0 = t
            img = f.to_ndarray(format="bgr24", width=w, height=h, interpolation="AREA")
            _draw(img, t, tracks, zones, ids, scale)
            if mid is None and t >= (start + end) / 2:
                mid = img.copy()
            for pkt in o.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
                dst.mux(pkt)
        for pkt in o.encode():
            dst.mux(pkt)
    if mid is not None:
        cv2.imwrite(str(out_jpg), mid, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return start if t0 is None else t0


def render_proxy(video: Path, out_mp4: Path) -> None:
    """Full-length 960-wide H.264 proxy, same frame rate, for tools/label.html."""
    import av

    with av.open(str(video)) as src, av.open(str(out_mp4), "w") as dst:
        s = src.streams.video[0]
        s.thread_type = "AUTO"
        w = OUT_WIDTH
        h = int(round(s.codec_context.height * w / s.codec_context.width / 2)) * 2
        o = dst.add_stream("libx264", rate=s.average_rate)
        o.width, o.height, o.pix_fmt = w, h, "yuv420p"
        o.options = {"crf": "28", "preset": "veryfast"}
        for f in src.decode(s):
            g = f.reformat(width=w, height=h, format="yuv420p", interpolation="AREA")
            g.pts = None
            for pkt in o.encode(g):
                dst.mux(pkt)
        for pkt in o.encode():
            dst.mux(pkt)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--seed", type=Path, help="ground_truth.json-shaped events to review as well")
    ap.add_argument("--out", type=Path, default=ROOT / "labels" / "review")
    ap.add_argument("--proxies", action="store_true", help="also write full-length H.264 proxies")
    args = ap.parse_args()

    seed = json.loads(args.seed.read_text()) if args.seed else {}
    media = args.out / "media"
    media.mkdir(parents=True, exist_ok=True)
    clips, cands = {}, []
    for video in args.videos:
        dur, fps, _n = probe_duration(video)
        clips[video.name] = {"duration": round(dur, 3), "fps": round(fps, 3), "proxy": None}
        found, tracks, zones = candidates_for(video, seed)
        print(f"{video.name}: {len(found)} candidates", flush=True)
        for c in found:
            cid = f"{video.stem}_{c['label']}_{c['start']:07.1f}".replace(".", "_")
            ws, we = max(0.0, c["start"] - PAD_SEC), min(dur, c["end"] + PAD_SEC)
            mp4, jpg = media / f"{cid}.mp4", media / f"{cid}.jpg"
            t0 = render_clip(video, ws, we, mp4, jpg, tracks, zones, c["track_ids"])
            cands.append({"id": cid, "clip": video.name, "start": round(c["start"], 3),
                          "end": round(c["end"], 3), "label": c["label"], "sources": c["sources"],
                          "window_start": round(t0, 4), "video": f"media/{mp4.name}",
                          "thumb": f"media/{jpg.name}"})
            print(f"   {c['label']:<16} {c['start']:7.1f}-{c['end']:7.1f}  {', '.join(c['sources'])}",
                  flush=True)
        if args.proxies:
            px = media / f"{video.stem}_proxy.mp4"
            if not px.exists():
                print(f"   proxy {px.name} ...", flush=True)
                render_proxy(video, px)
            clips[video.name]["proxy"] = f"media/{px.name}"

    data = {"clips": clips, "candidates": cands}
    (args.out / "data.js").write_text("window.REVIEW = " + json.dumps(data, indent=1) + ";\n",
                                      encoding="utf-8")
    shutil.copyfile(ROOT / "tools" / "review.html", args.out / "index.html")
    print(f"wrote {args.out / 'index.html'} ({len(cands)} candidates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
