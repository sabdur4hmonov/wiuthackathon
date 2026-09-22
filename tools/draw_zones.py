#!/usr/bin/env python3
"""Render zones.json over a frame so the geometry can be checked by eye.

    python tools/draw_zones.py --image frames/clip1_f000000.png \
        --zones config/zones.json --out check.png

Do not trust config/zones.json until you have looked at this output. The
validator only proves the file is complete and self-consistent; it cannot tell
you that the polygon you called "lane 2" is actually the pavement.

Runs on an unvalidated file deliberately (--strict to opt in), because its main
use is mid-authoring, when the file is still half-drawn.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.zones import ZonesError, validate_raw  # noqa: E402

# BGR
COLOURS = {
    "carriageway": (255, 156, 91),
    "lanes": (142, 207, 62),
    "stop_lines": (72, 182, 255),
    "crossings": (234, 146, 199),
    "signal_queue_zones": (107, 159, 255),
    "traffic_lights": (107, 107, 255),
    "lane_markings": (230, 233, 230),
}


def _pts(raw) -> np.ndarray | None:
    if raw is None:
        return None
    try:
        return np.asarray(raw, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return None


def render(image: Path, zones_path: Path, out: Path, strict: bool) -> None:
    img = cv2.imread(str(image))
    if img is None:
        raise SystemExit(f"cannot read image {image}")
    h, w = img.shape[:2]

    raw = json.loads(zones_path.read_text(encoding="utf-8"))
    errors = validate_raw(raw)
    if errors and strict:
        raise SystemExit("zones.json is not fully authored:\n  - "
                         + "\n  - ".join(errors))

    aa = raw.get("authored_against") or {}
    aw, ah = aa.get("image_width"), aa.get("image_height")
    sx = w / aw if aw else 1.0
    sy = h / ah if ah else 1.0
    if (aw and ah) and (aw != w or ah != h):
        print(f"! zones authored against {aw}x{ah}, image is {w}x{h}; "
              f"rescaling by ({sx:.3f}, {sy:.3f})")

    overlay = img.copy()
    drawn = 0

    def scale(p: np.ndarray) -> np.ndarray:
        return (p * np.array([sx, sy], dtype=np.float32)).astype(np.int32)

    def label_at(pt, text, colour):
        x, y = int(pt[0]), int(pt[1])
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x + 4, y - th - 8), (x + tw + 10, y - 2), (0, 0, 0), -1)
        cv2.putText(img, text, (x + 7, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, colour, 1, cv2.LINE_AA)

    # -- filled polygons ---------------------------------------------------
    for key in ("carriageway", "crossings", "signal_queue_zones", "lanes"):
        for item in raw.get(key) or []:
            p = _pts(item.get("polygon"))
            if p is None:
                continue
            q = scale(p)
            c = COLOURS[key]
            cv2.fillPoly(overlay, [q], c)
            cv2.polylines(img, [q], True, c, 2, cv2.LINE_AA)
            label_at(q[0], str(item.get("id", key)), c)
            drawn += 1

    cv2.addWeighted(overlay, 0.22, img, 0.78, 0, img)

    # -- lane direction arrows --------------------------------------------
    for item in raw.get("lanes") or []:
        f_, t_ = item.get("direction_from"), item.get("direction_to")
        if not f_ or not t_:
            continue
        a = scale(np.asarray([f_], dtype=np.float32))[0]
        b = scale(np.asarray([t_], dtype=np.float32))[0]
        cv2.arrowedLine(img, tuple(a), tuple(b), COLOURS["lanes"], 3,
                        cv2.LINE_AA, tipLength=0.25)

    # -- stop lines --------------------------------------------------------
    for item in raw.get("stop_lines") or []:
        s = _pts(item.get("segment"))
        if s is None:
            continue
        q = scale(s)
        c = COLOURS["stop_lines"]
        cv2.line(img, tuple(q[0]), tuple(q[1]), c, 4, cv2.LINE_AA)
        # Tick marking the approach side, so a flipped sign is visible.
        side = item.get("approach_side")
        if side in (-1, 1):
            mid = q.mean(axis=0)
            d = q[1] - q[0]
            n = np.array([-d[1], d[0]], dtype=np.float32)
            n = n / (np.linalg.norm(n) + 1e-6) * 34.0 * float(side)
            tip = (mid + n).astype(np.int32)
            cv2.arrowedLine(img, tuple(mid.astype(np.int32)), tuple(tip),
                            c, 2, cv2.LINE_AA, tipLength=0.35)
            label_at(tip, f"approach {side:+d}", c)
        label_at(q[0], str(item.get("id", "stopline")), c)
        drawn += 1

    # -- lane markings -----------------------------------------------------
    for item in raw.get("lane_markings") or []:
        s = _pts(item.get("segment"))
        if s is None:
            continue
        q = scale(s)
        style = item.get("style")
        c = COLOURS["lane_markings"]
        if style == "dashed":
            _dashed(img, q[0], q[1], c)
        elif style == "double_solid":
            cv2.line(img, tuple(q[0]), tuple(q[1]), c, 2, cv2.LINE_AA)
            off = np.array([4, 0], dtype=np.int32)
            cv2.line(img, tuple(q[0] + off), tuple(q[1] + off), c, 2, cv2.LINE_AA)
        else:
            cv2.line(img, tuple(q[0]), tuple(q[1]), c, 3, cv2.LINE_AA)
        label_at(q[0], f"{item.get('id', 'mark')} [{style}]", c)
        drawn += 1

    # -- traffic lights ----------------------------------------------------
    for item in raw.get("traffic_lights") or []:
        roi = item.get("roi")
        if not roi:
            continue
        x1, y1, x2, y2 = (float(v) for v in roi)
        p1 = (int(x1 * sx), int(y1 * sy))
        p2 = (int(x2 * sx), int(y2 * sy))
        c = COLOURS["traffic_lights"]
        cv2.rectangle(img, p1, p2, c, 2)
        label_at(p1, str(item.get("id", "signal")), c)
        drawn += 1

    # -- banner ------------------------------------------------------------
    msg = (f"{drawn} shapes" if not errors
           else f"{drawn} shapes — {len(errors)} UNAUTHORED FIELD(S)")
    colour = (142, 207, 62) if not errors else (72, 182, 255)
    cv2.rectangle(img, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(img, f"{zones_path.name}: {msg}", (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)

    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"wrote {out}  ({drawn} shapes drawn)")
    if errors:
        print(f"\n{len(errors)} field(s) still un-authored:")
        for e in errors[:15]:
            print(f"  - {e}")
        if len(errors) > 15:
            print(f"  ... and {len(errors) - 15} more")


def _dashed(img, a, b, colour, dash=12):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    n = int(max(1, np.linalg.norm(b - a) // dash))
    for k in range(0, n, 2):
        p = a + (b - a) * (k / n)
        q = a + (b - a) * (min(k + 1, n) / n)
        cv2.line(img, tuple(p.astype(np.int32)), tuple(q.astype(np.int32)),
                 colour, 2, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--zones", default=Path("config/zones.json"), type=Path)
    ap.add_argument("--out", default=Path("check.png"), type=Path)
    ap.add_argument("--strict", action="store_true",
                    help="refuse to render a file that still has TODOs")
    args = ap.parse_args()
    render(args.image, args.zones, args.out, args.strict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
