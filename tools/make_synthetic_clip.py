#!/usr/bin/env python3
"""Generate a synthetic road-camera clip, for timing only.

    python tools/make_synthetic_clip.py --out samples/_synthetic.mp4 \
        --seconds 60 --fps 25 --width 1920 --height 1080

This exists so the decode floor and detector throughput can be measured before
the real clips are in hand. It is NOT a stand-in for the real videos: the scene
is fake, so detection COUNTS from it mean nothing. Only the timings transfer,
and only roughly -- a synthetic frame with a handful of moving rectangles
compresses far better than real CCTV, so its decode cost is a LOWER bound on
the real one.

Re-measure with tools/bench.py on a real clip as soon as one exists.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def make(out: Path, seconds: float, fps: float, width: int, height: int,
         n_objects: int, seed: int = 1337) -> Path:
    rng = np.random.default_rng(seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"cannot open writer for {out}")

    n_frames = int(round(seconds * fps))
    # A static "road" background plus rectangles that drive along it. Enough
    # structure that the encoder cannot collapse the whole clip to nothing.
    bg = np.full((height, width, 3), 60, dtype=np.uint8)
    cv2.rectangle(bg, (0, int(height * 0.45)), (width, height), (78, 78, 82), -1)
    for y in range(int(height * 0.45), height, 60):
        cv2.line(bg, (0, y), (width, y), (92, 92, 96), 1)
    noise = rng.integers(0, 14, (height, width, 3), dtype=np.uint8)
    bg = cv2.add(bg, noise)

    objs = []
    for _ in range(n_objects):
        objs.append({
            "x": float(rng.uniform(0, width)),
            "y": float(rng.uniform(height * 0.5, height * 0.95)),
            "vx": float(rng.uniform(-9, 9)) or 5.0,
            "w": float(rng.uniform(55, 160)),
            "h": float(rng.uniform(40, 95)),
            "c": tuple(int(v) for v in rng.integers(40, 235, 3)),
        })

    for _ in range(n_frames):
        frame = bg.copy()
        for o in objs:
            o["x"] += o["vx"]
            if o["x"] < -o["w"]:
                o["x"] = width + o["w"]
            if o["x"] > width + o["w"]:
                o["x"] = -o["w"]
            x1, y1 = int(o["x"]), int(o["y"])
            cv2.rectangle(frame, (x1, y1), (x1 + int(o["w"]), y1 + int(o["h"])),
                          o["c"], -1)
            cv2.rectangle(frame, (x1, y1), (x1 + int(o["w"]), y1 + int(o["h"])),
                          (20, 20, 20), 2)
        writer.write(frame)
    writer.release()

    cap = cv2.VideoCapture(str(out))
    got = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    got_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    mb = out.stat().st_size / (1 << 20)
    print(f"wrote {out}: {got} frames @ {got_fps:.2f} fps "
          f"({got / got_fps:.1f}s), {width}x{height}, {mb:.1f} MB")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=Path("samples/_synthetic.mp4"), type=Path)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--objects", type=int, default=14)
    args = ap.parse_args()
    make(args.out, args.seconds, args.fps, args.width, args.height, args.objects)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
