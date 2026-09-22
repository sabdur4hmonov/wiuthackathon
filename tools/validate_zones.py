#!/usr/bin/env python3
"""Fail loudly on a zones.json that is still half-authored.

    python tools/validate_zones.py [--zones config/zones.json]

Exit 0 only when every geometric field is filled and every cross-reference
resolves. Exit 1 otherwise, listing each problem.

This exists because the failure mode it prevents is silent: rules built on a
partly drawn scene do not crash, they emit confident nonsense. Nonsense events
score 0 for their class AND enlarge Part A's class set, so they cost more than
emitting nothing at all.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.zones import load_zones, validate_raw  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zones", default=Path("config/zones.json"), type=Path)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not args.zones.exists():
        print(f"ERROR: {args.zones} does not exist", file=sys.stderr)
        return 1
    try:
        raw = json.loads(args.zones.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"ERROR: {args.zones} is not valid JSON: {e}", file=sys.stderr)
        return 1

    errors = validate_raw(raw)
    if errors:
        print(f"{args.zones}: NOT AUTHORED — {len(errors)} problem(s)\n")
        for e in errors:
            print(f"  - {e}")
        print("\nDraw the missing geometry:")
        print("  1. python tools/extract_frames.py --video samples/<clip>.mp4 --out frames/")
        print("  2. open tools/annotate.html in a browser, load the frame, draw, download")
        print("  3. move the downloaded zones.json over config/zones.json")
        print("  4. python tools/draw_zones.py --image <frame> --out check.png  # look at it")
        return 1

    z = load_zones(args.zones)
    if not args.quiet:
        print(f"{args.zones}: VALID")
        print(f"  {z.summary()}")
        for ln in z.lanes:
            print(f"  lane {ln.id:<14} dir=({ln.direction[0]:+.2f}, "
                  f"{ln.direction[1]:+.2f})  "
                  f"manoeuvres={sorted(ln.permitted_manoeuvres) or '[]'}  "
                  f"stop_line={ln.governed_by_stop_line}")
        for sl in z.stop_lines:
            print(f"  stop_line {sl.id:<12} approach_side={sl.approach_side:+d}  "
                  f"governs={list(sl.governs_lanes)}")
        if not z.has_traffic_light:
            print("  note: no traffic_lights authored — red_light / stop_line rules "
                  "must infer signal phase from traffic behaviour")
        if z.notes:
            print(f"  notes: {z.notes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
