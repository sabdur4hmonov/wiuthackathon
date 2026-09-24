"""Flag windows a PERSON should look at for stopped_vehicle / wrong_way / congestion.

A SEARCH AID FOR MANUAL REVIEW, NOT A DETECTOR. Nothing here is registered as
a rule, nothing in src/ imports it, and it never produces events: it produces
places to look, deliberately far looser than the shipped rules, ranked so a
reviewer can scan ~10-15 windows per clip instead of 5 minutes of raw video.
tests/test_scan_windows.py pins that separation.

Used by tools/review_candidates.py --scan, which renders each window like any
other candidate (short clip, source time burned in, the flagged tracks boxed).

What is flagged, per clip:

  stationary   A vehicle track whose net movement over 2 s stays under
               STATIONARY_SPEED_L_S for at least STATIONARY_MIN_SEC anywhere
               on the carriageway -- inside queue zones too, confidence and
               kerb gates off (the rule wants 10 s, outside queues, off the
               kerb). Windows of different ids at the same spot are merged
               (an id switch mid-stop). Stops inside one signal queue zone at
               the same time are ONE window ("queue") -- those are normally a
               red phase; only the longest few are kept, because a queue that
               outlasts a red phase is exactly what a human should see.
               Outside-queue stops rank first.

  direction    The wrong_way rule with every threshold loosened, and its
               keyframe-rate gate lifted. At 0.5 s per sample moving platoons
               ALIAS (a track hops back one car per sample and drifts
               backwards), so expect most of these to be that: the card says
               so. Ranked by distance travelled against the lane.

  slow traffic The congestion rule, loosened and ungated the same way.

Caps: CAPS per kind per clip. Long windows are rendered sped up (the page
shows the factor and still converts video time to source time exactly).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from src import geometry as g
from src.rules import congestion, frames_to_seconds, wrong_way
from src.rules.util import box_heights, runs_to_segments, sustained_speed
from src.rules.zoneindex import zone_index
from src.thresholds import TH
from src.tracks import COL

STATIONARY_MIN_SEC = 5.0          # the rule: 10 s
STATIONARY_SPEED_L_S = 0.6        # same as the rule: box heights per second over 2 s
STATIONARY_GAP_SEC = 2.0
SAME_SPOT_L = 1.5                 # merge windows of different ids this close (box heights)
QUEUE_WINDOWS_KEPT = 2            # longest queue-zone windows kept per clip
CAPS = {"stationary": 6, "direction": 4, "slow traffic": 3}

RELAXED_WRONG_WAY = dataclasses.replace(
    TH.wrong_way, min_angle_deg=120.0, min_speed_L_s=0.3, min_duration_sec=1.5,
    min_progress_L=1.0, min_confidence=0.30, max_sample_sec=10.0)
RELAXED_CONGESTION = dataclasses.replace(
    TH.congestion, min_vehicles=3, slow_fraction=0.6, min_duration_sec=15.0,
    max_sample_sec=10.0)


@dataclass
class Window:
    start: float
    end: float
    label: str            # the class a reviewer should check for
    kind: str             # "stationary" | "direction" | "slow traffic"
    reason: str
    track_ids: tuple[int, ...]
    rank: float           # higher = look first

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def _queue_of(pt, zones) -> str | None:
    for q in zones.signal_queue_zones:
        if g.point_in_polygon(pt, q.polygon):
            return q.id
    return None


def stationary_windows(tracks, zones) -> list[Window]:
    veh = tracks.vehicles()
    if zones is None or len(veh) == 0:
        return []
    d = veh.data
    fps, step = veh.fps or 25.0, max(1, veh.frame_stride)
    idx = zone_index(veh, zones)
    size = box_heights(d)
    ids = d[:, COL["track_id"]].astype(np.int64)
    frames = d[:, COL["frame_idx"]].astype(np.int64)
    ok = ((sustained_speed(veh) <= STATIONARY_SPEED_L_S * size)
          & idx.on_carriageway & (ids >= 0))

    raw = []   # (start, end, x, y, L, tid)
    for tid in np.unique(ids[ok]):
        sel = np.flatnonzero(ids == tid)
        sel = sel[np.argsort(frames[sel], kind="stable")]
        for s, e in runs_to_segments(frames[sel], ok[sel], fps, STATIONARY_MIN_SEC,
                                     STATIONARY_GAP_SEC, step):
            m = sel[(frames[sel] >= s) & (frames[sel] <= e)]
            raw.append([s / fps, (e + step) / fps, float(np.median(d[m, COL["gx"]])),
                        float(np.median(d[m, COL["gy"]])), float(np.median(size[m])), {int(tid)}])

    # Same spot, overlapping in time: one stop seen under two ids.
    merged: list[list] = []
    for r in sorted(raw, key=lambda r: r[0]):
        for m in merged:
            if (min(m[1], r[1]) > max(m[0], r[0]) - STATIONARY_GAP_SEC
                    and np.hypot(m[2] - r[2], m[3] - r[3]) <= SAME_SPOT_L * max(m[4], r[4])):
                m[0], m[1] = min(m[0], r[0]), max(m[1], r[1])
                m[5] |= r[5]
                break
        else:
            merged.append(r)

    out: list[Window] = []
    queues: dict[str, list[list]] = {}
    for s, e, x, y, _L, tids in merged:
        q = _queue_of((x, y), zones)
        if q is None:
            out.append(Window(s, e, "stopped_vehicle", "stationary",
                              f"vehicle near-stationary {e - s:.0f} s OUTSIDE any queue zone "
                              f"(rule needs {TH.stopped_vehicle.min_duration_sec:.0f} s and more gates)",
                              tuple(sorted(tids)), 1000.0 + (e - s)))
        else:
            queues.setdefault(q, []).append([s, e, tids])
    for q, items in queues.items():
        items.sort(key=lambda it: it[0])
        groups: list[list] = []
        for s, e, tids in items:
            if groups and s <= groups[-1][1]:
                groups[-1][1] = max(groups[-1][1], e)
                groups[-1][2] |= tids
            else:
                groups.append([s, e, set(tids)])
        for s, e, tids in groups:
            out.append(Window(s, e, "stopped_vehicle", "stationary",
                              f"{len(tids)} vehicle(s) stationary in queue zone {q} for {e - s:.0f} s "
                              f"(normally a red phase; look for a stop that outlasts it, or congestion)",
                              tuple(sorted(tids)), e - s))
    inside = sorted((w for w in out if w.rank < 1000), key=lambda w: -w.rank)[:QUEUE_WINDOWS_KEPT]
    outside = [w for w in out if w.rank >= 1000]
    return outside + inside


def direction_windows(tracks, zones) -> list[Window]:
    if zones is None:
        return []
    out = []
    for sg in wrong_way.detect(tracks, zones, RELAXED_WRONG_WAY):
        r = frames_to_seconds(sg, "wrong_way", tracks.fps, tracks.frame_stride)
        against = float(sg.debug.get("against_L", 0.0))
        out.append(Window(r.start, r.end, "wrong_way", "direction",
                          f"track moved {against:.1f} box heights against lane {sg.debug.get('lane', '?')}"
                          f" -- at keyframe rate this is often a platoon ALIASING, check the traffic",
                          tuple(int(i) for i in sg.track_ids), against))
    return out


def congestion_windows(tracks, zones) -> list[Window]:
    if zones is None:
        return []
    out = []
    for sg in congestion.detect(tracks, zones, RELAXED_CONGESTION):
        r = frames_to_seconds(sg, "congestion", tracks.fps, tracks.frame_stride)
        group = sg.debug.get("direction", sg.debug.get("group", "?"))
        out.append(Window(r.start, r.end, "congestion", "slow traffic",
                          f"most vehicles of direction {group} slow for {r.end - r.start:.0f} s "
                          f"outside queue zones (rule needs {TH.congestion.min_duration_sec:.0f} s)",
                          (), r.end - r.start))
    return out


def scan(tracks, zones) -> list[Window]:
    """Every flagged window for one clip, capped per kind, look-first order."""
    found = []
    for kind, fn in (("stationary", stationary_windows), ("direction", direction_windows),
                     ("slow traffic", congestion_windows)):
        ws = sorted(fn(tracks, zones), key=lambda w: -w.rank)[:CAPS[kind]]
        found.extend(ws)
    return sorted(found, key=lambda w: w.start)
