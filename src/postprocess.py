"""Stage 3 -- raw segments in, submittable events out.

Order matters and is deliberate:

  1. clamp to [0, duration]      -- the harness clamps end to duration and drops
                                    anything with start >= end; do it ourselves
                                    so we know what survives.
  2. merge same-class fragments  -- a tracker ID switch mid-event splits one
                                    real event into two. Two halves each score
                                    ~0.5 IoU against the truth and can both miss
                                    the 0.7 threshold, so merging is worth real
                                    points.
  3. drop blips                  -- a sub-second segment is almost always
                                    detector noise, and each one is a pure FP.
  4. enforce no same-class overlap.

Step 4 is the one that must never be left to the harness. run_submission.py
resolves a same-class overlap by KEEPING THE EARLIER-STARTING SEGMENT and
silently dropping the other (clean_events, "same-class overlaps: keep the
earlier-starting segment"). That can delete a real, well-localised detection in
favour of a long noisy one that happened to start first. Merging the pair
ourselves keeps the evidence and costs nothing.

Different classes MAY overlap -- the task spec is explicit that a wrong_way
vehicle causing an accident is two events -- so overlap handling is strictly
per-class.
"""
from __future__ import annotations

from .config import CFG, PostConfig
from .rules import RawSegment


def clamp(segments: list[RawSegment], duration: float) -> list[RawSegment]:
    """Clip to the video, drop anything with no positive length left."""
    out = []
    for s in segments:
        start = max(0.0, float(s.start))
        end = min(float(duration), float(s.end))
        if end - start <= 0.0:
            continue
        out.append(RawSegment(start, end, s.label, s.score, s.track_ids, s.debug))
    return out


def merge_same_class(segments: list[RawSegment], gap: float) -> list[RawSegment]:
    """Union same-class segments that overlap or sit within `gap` of each other.

    Merging rather than discarding is the whole point: two fragments of one
    event become one segment whose IoU against the truth is better than either
    half's, instead of one TP-candidate plus one guaranteed FP.
    """
    by_label: dict[str, list[RawSegment]] = {}
    for s in segments:
        by_label.setdefault(s.label, []).append(s)

    out: list[RawSegment] = []
    for label, group in by_label.items():
        group.sort(key=lambda s: (s.start, s.end))
        cur = group[0]
        acc_ids = list(cur.track_ids)
        best = cur.score
        for nxt in group[1:]:
            if nxt.start - cur.end <= gap:
                cur = RawSegment(cur.start, max(cur.end, nxt.end), label,
                                 max(best, nxt.score))
                best = cur.score
                acc_ids.extend(nxt.track_ids)
            else:
                out.append(RawSegment(cur.start, cur.end, label, best,
                                      tuple(dict.fromkeys(acc_ids))))
                cur, best, acc_ids = nxt, nxt.score, list(nxt.track_ids)
        out.append(RawSegment(cur.start, cur.end, label, best,
                              tuple(dict.fromkeys(acc_ids))))
    return out


def drop_blips(segments: list[RawSegment], min_duration: float) -> list[RawSegment]:
    return [s for s in segments if (s.end - s.start) >= min_duration]


def enforce_no_overlap(segments: list[RawSegment]) -> list[RawSegment]:
    """Guarantee no two same-class segments overlap, by merging any that do.

    Runs after blip removal because removing a short segment can never create
    an overlap, but merging can create a segment long enough to touch the next
    one. Idempotent: a second pass changes nothing.
    """
    by_label: dict[str, list[RawSegment]] = {}
    for s in segments:
        by_label.setdefault(s.label, []).append(s)

    out: list[RawSegment] = []
    for label, group in by_label.items():
        group.sort(key=lambda s: (s.start, s.end))
        cur = group[0]
        for nxt in group[1:]:
            if nxt.start < cur.end:                      # strict: touching is legal
                cur = RawSegment(cur.start, max(cur.end, nxt.end), label,
                                 max(cur.score, nxt.score),
                                 tuple(dict.fromkeys(cur.track_ids + nxt.track_ids)))
            else:
                out.append(cur)
                cur = nxt
        out.append(cur)
    return out


def assert_no_overlap(events: list[list]) -> None:
    """Fail loudly if the invariant broke. Called before returning to the harness.

    evaluate.py's validator uses `s2 < e1` on sorted same-class segments, so
    touching segments (s2 == e1) are legal. This mirrors that exactly.
    """
    by_label: dict[str, list[tuple[float, float]]] = {}
    for s, e, label in events:
        by_label.setdefault(label, []).append((float(s), float(e)))
    for label, ss in by_label.items():
        ss.sort()
        for (s1, e1), (s2, e2) in zip(ss, ss[1:]):
            if s2 < e1:
                raise AssertionError(
                    f"overlapping {label!r} segments [{s1}, {e1}] and [{s2}, {e2}] "
                    f"survived post-processing"
                )


def finalise(segments: list[RawSegment], duration: float,
             cfg: PostConfig | None = None) -> list[list]:
    """The whole of Stage 3. Returns harness-ready [[start, end, label], ...]."""
    cfg = cfg or CFG.post
    segs = clamp(segments, duration)
    segs = merge_same_class(segs, cfg.merge_gap_sec)
    segs = drop_blips(segs, cfg.min_duration_sec)
    segs = enforce_no_overlap(segs)
    events = [s.as_event() for s in segs]
    events.sort(key=lambda e: (e[0], e[1], e[2]))
    assert_no_overlap(events)
    return events
