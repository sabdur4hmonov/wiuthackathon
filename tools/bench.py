#!/usr/bin/env python3
"""Measure where the budget goes, in seconds per second of video.

    python tools/bench.py --video samples/clip1.mp4

Reports, for one clip:

  decode-floor   a bare cv2 read() loop over every frame and nothing else,
                 reported as the MEDIAN of several passes. This is the hard
                 floor for Part B: the harness re-decodes the whole video and
                 calls step() on every frame, so Part B can never cost less
                 than this no matter what step() does.
  step-cost      RiskEstimator.step() timed in ISOLATION, not by differencing
                 two decode passes. A single decode pass varies by ~20% run to
                 run, which is thousands of times step()'s own cost, so the
                 subtraction would report jitter as signal.
  part-a         detect_events end to end, with the cache bypassed.

Everything is printed as a multiple of the video's duration, because that is
the unit the harness budget is written in: Part A + Part B must together stay
under 3.0x, or the video scores as empty for BOTH parts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.budget import probe_duration  # noqa: E402


def _decode_once(video: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(video))
    n = 0
    t0 = time.perf_counter()
    while True:
        ok, _frame = cap.read()
        if not ok:
            break
        n += 1
    el = time.perf_counter() - t0
    cap.release()
    return el, n


def bench_decode(video: Path, duration: float, repeats: int = 3) -> dict:
    """Pure decode: read every frame, touch nothing.

    Repeated and reported as a MEDIAN. A single decode pass varies by ~20% run
    to run on a loaded desktop, which is larger than most of the differences we
    care about -- and much larger than step()'s own cost, so measuring that by
    subtracting two decode runs would report noise as signal.
    """
    runs = [_decode_once(video) for _ in range(max(1, repeats))]
    times = sorted(r[0] for r in runs)
    n = runs[0][1]
    med = times[len(times) // 2]
    return {"frames": n, "sec": med, "sec_min": times[0], "sec_max": times[-1],
            "runs": len(times),
            "x_realtime": med / duration if duration else None,
            "ms_per_frame": 1000.0 * med / n if n else None}


def bench_step(video: Path, n_calls: int = 100_000) -> dict:
    """Cost of RiskEstimator.step() ALONE, with no decode in the loop.

    Measured directly rather than by differencing two decode passes: step() is
    sub-microsecond by design, which is three orders of magnitude below decode
    jitter. The number that matters is whether step() is negligible against the
    decode floor, and only an isolated measurement can show that.
    """
    import numpy as np

    from src.risk import RiskEstimator

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        frame = np.zeros((h or 1080, w or 1920, 3), dtype=np.uint8)

    est = RiskEstimator()
    est.reset({"video_id": video.name, "fps": fps, "width": w, "height": h,
               "n_frames": n_frames})
    for i in range(2000):                     # warm up the branch predictor
        est.step(frame, i / fps)

    est.reset({"video_id": video.name, "fps": fps, "width": w, "height": h,
               "n_frames": n_frames})
    t0 = time.perf_counter()
    for i in range(n_calls):
        est.step(frame, i / fps)
    el = time.perf_counter() - t0
    per = el / n_calls
    return {"calls": n_calls, "sec": el, "us_per_frame": 1e6 * per,
            "sec_for_this_clip": per * n_frames}


def bench_part_a(video: Path, duration: float, use_cache: bool) -> dict:
    from src.pipeline import detect_events

    t0 = time.perf_counter()
    events = detect_events(str(video), verbose=True, use_cache=use_cache)
    el = time.perf_counter() - t0
    return {"sec": el, "x_realtime": el / duration if duration else None,
            "n_events": len(events)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--skip-part-a", action="store_true")
    ap.add_argument("--decode-repeats", type=int, default=3,
                    help="decode passes to median over (default 3)")
    ap.add_argument("--use-cache", action="store_true",
                    help="measure the cached path instead of a cold run")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    if not args.video.exists():
        raise SystemExit(f"no such video: {args.video}")

    duration, fps, n_frames = probe_duration(args.video)
    budget = 3.0 * duration
    size_mb = args.video.stat().st_size / (1 << 20)
    print(f"=== {args.video.name} ===")
    print(f"  {n_frames} frames @ {fps:.2f} fps = {duration:.2f}s "
          f"(harness formula: n_frames / fps), {size_mb:.1f} MB")
    print(f"  harness budget for Part A + Part B: {budget:.1f}s (3.0x)\n")

    rep: dict = {"video": args.video.name, "duration_sec": duration, "fps": fps,
                 "n_frames": n_frames, "budget_sec": budget}

    print(f"[1/3] pure decode floor, median of {args.decode_repeats} runs "
          f"(Part B can never beat this) ...")
    rep["decode_floor"] = bench_decode(args.video, duration, args.decode_repeats)
    d = rep["decode_floor"]
    print(f"      {d['sec']:.2f}s  =  {d['x_realtime']:.3f}x realtime  "
          f"({d['ms_per_frame']:.2f} ms/frame over {d['frames']} frames)")
    print(f"      spread across runs: {d['sec_min']:.2f}s .. {d['sec_max']:.2f}s\n")

    print("[2/3] RiskEstimator.step() in isolation ...")
    rep["step_cost"] = bench_step(args.video)
    s = rep["step_cost"]
    print(f"      {s['us_per_frame']:.3f} us/frame over {s['calls']} calls")
    print(f"      = {s['sec_for_this_clip']:.4f}s for this clip's "
          f"{d['frames']} frames "
          f"({100.0 * s['sec_for_this_clip'] / d['sec']:.3f}% of the decode floor)\n")

    # Part B's real cost is the floor plus step()'s own, not a difference of
    # two noisy decode passes.
    part_b_sec = d["sec"] + s["sec_for_this_clip"]
    rep["risk_passthrough"] = {
        "sec": part_b_sec,
        "x_realtime": part_b_sec / duration if duration else None,
    }

    if not args.skip_part_a:
        print("[3/3] Part A end to end ...")
        rep["part_a"] = bench_part_a(args.video, duration, args.use_cache)
        a = rep["part_a"]
        print(f"      {a['sec']:.2f}s  =  {a['x_realtime']:.3f}x realtime, "
              f"{a['n_events']} events\n")
    else:
        rep["part_a"] = None

    # -- verdict -----------------------------------------------------------
    a_x = rep["part_a"]["x_realtime"] if rep["part_a"] else 0.0
    b_x = rep["risk_passthrough"]["x_realtime"]
    total = a_x + b_x
    rep["total_x_realtime"] = total
    rep["headroom_x"] = 3.0 - total
    print("=== verdict ===")
    print(f"  Part A  {a_x:.3f}x  +  Part B  {b_x:.3f}x  =  {total:.3f}x "
          f"of a 3.000x budget")
    if total >= 3.0:
        print(f"  ! OVER BUDGET by {total - 3.0:.3f}x — every video would score as EMPTY")
    elif total >= 2.4:
        print(f"  ! only {3.0 - total:.3f}x headroom; the eval box may be slower")
    else:
        print(f"  OK — {3.0 - total:.3f}x headroom")

    if args.json:
        args.json.write_text(json.dumps(rep, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
