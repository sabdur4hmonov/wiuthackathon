"""Report a clip's GOP structure: keyframe interval and whether it has B-frames.

    python tools/gop_probe.py samples/sample_001.mp4 [more.mp4 ...]

The keyframe interval comes from demuxing every packet (no decoding, a few
seconds for a 6 GB file). Frame types come from decoding the first three GOPs.
Part A decodes keyframes only (src/perception.py), so the keyframe interval is
the tracker's sample period.
"""
from __future__ import annotations

import sys
from collections import Counter

import av
import numpy as np

TYPE = {0: "?", 1: "I", 2: "P", 3: "B", 4: "S", 5: "SI", 6: "SP", 7: "BI"}


def probe(path: str) -> dict:
    with av.open(path) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate)
        keys, n, reordered = [], 0, 0
        for pkt in c.demux(s):
            if pkt.size == 0:
                continue
            if pkt.is_keyframe:
                keys.append(n)
            if pkt.pts is not None and pkt.dts is not None and pkt.pts != pkt.dts:
                reordered += 1
            n += 1
        has_b = bool(s.codec_context.has_b_frames)
    gaps = np.diff(keys) if len(keys) > 1 else np.array([n])

    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        stop = keys[3] if len(keys) > 3 else 90
        types = [TYPE.get(int(f.pict_type), "?")
                 for i, f in zip(range(stop), c.decode(s))]
    return {
        "path": path, "fps": fps, "packets": n, "keyframes": len(keys),
        "gop_min": int(gaps.min()), "gop_median": float(np.median(gaps)),
        "gop_max": int(gaps.max()), "gop_sec": float(np.median(gaps)) / fps,
        "has_b_frames": has_b, "reordered_packets": reordered,
        "types": "".join(types), "type_counts": dict(Counter(types)),
    }


def main() -> None:
    for path in sys.argv[1:]:
        r = probe(path)
        print(f"{r['path']}: {r['fps']:.3f} fps, {r['packets']} frames, "
              f"{r['keyframes']} keyframes")
        print(f"  GOP {r['gop_min']}/{r['gop_median']:.0f}/{r['gop_max']} frames "
              f"(min/median/max) = {r['gop_sec']:.3f} s")
        print(f"  B-frames: {r['has_b_frames']}  (packets with pts != dts: "
              f"{r['reordered_packets']})")
        print(f"  first frames, display order: {r['types']}  {r['type_counts']}")


if __name__ == "__main__":
    main()
