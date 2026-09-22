#!/usr/bin/env python3
"""Dump representative frames from a sample video as PNGs, for hand-annotation.

    python tools/extract_frames.py --video samples/clip1.mp4 --out frames/

Writes the first frame plus `--count` frames spread evenly through the clip.
The first frame is always included because it is the one to annotate against:
zones.json records which frame its coordinates came from, and a fixed camera
means frame 0 is as good as any -- but pick a LATER frame if frame 0 is dark,
compressed or has a vehicle parked over the stop line.

PNG, not JPEG: the annotation step reads pixel coordinates off this image and
JPEG ringing around painted lines makes the lines harder to trace by eye.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def extract(video: Path, out_dir: Path, count: int) -> list[Path]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{video.name}: {n_frames} frames, {fps:.2f} fps, {w}x{h}, "
          f"{n_frames / fps if fps else 0:.1f}s")

    if n_frames <= 0:
        raise SystemExit(f"{video} reports {n_frames} frames; is it readable?")

    # Frame 0, then evenly spread; avoid the very last frame, which is often a
    # partial or duplicated frame in re-encoded clips.
    wanted = [0]
    if count > 1:
        span = n_frames - 1
        wanted += [int(round(span * k / count)) for k in range(1, count)]
    wanted = sorted(set(i for i in wanted if 0 <= i < n_frames))

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for idx in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"  ! frame {idx} unreadable, skipped", file=sys.stderr)
            continue
        p = out_dir / f"{video.stem}_f{idx:06d}.png"
        cv2.imwrite(str(p), frame)
        written.append(p)
        print(f"  wrote {p}  (t={idx / fps if fps else 0:.2f}s)")
    cap.release()

    if written:
        print(f"\nAnnotate against: {written[0]}")
        print("  open tools/annotate.html in a browser, load that frame, draw, "
              "then Download zones.json over config/zones.json")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", default=Path("frames"), type=Path)
    ap.add_argument("--count", type=int, default=6,
                    help="how many frames total (default 6)")
    args = ap.parse_args()
    extract(args.video, args.out, max(1, args.count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
