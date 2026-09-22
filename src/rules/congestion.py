"""congestion -- standstill or crawling traffic across ALL lanes of one
direction of travel.

TWO THINGS MAKE THIS HARD.

1. "All lanes of a direction" is not in the zones schema. Lanes carry a
   direction vector but no grouping, so directions are recovered by clustering
   those vectors: lanes whose arrows agree within a tolerance are one
   direction. On a four-way intersection that yields the four expected groups.
   It also means a lane whose arrow was authored sloppily lands in the wrong
   group, which is a real failure mode to watch for on the hand-drawn config.

2. A NORMAL RED-LIGHT QUEUE LOOKS EXACTLY LIKE CONGESTION. Same stopped
   vehicles, same lanes, often the same duration -- a red phase can run 60-90 s,
   so no duration threshold separates them. Duration is a weak secondary guard
   here, nothing more.

   What separates them is WHERE the stopped traffic is. Vehicles inside a
   signal_queue_zone are excluded from the statistics entirely. A queue waiting
   on red sits inside that zone, so its lanes have no measurable vehicles and
   cannot be judged congested. A genuine jam backs up past the zone and is
   counted. This is the whole discriminator, and it depends completely on the
   queue zones being drawn generously in the real zones.json -- the CP0 schema
   note already says to err large, and this rule is why.

   The honest limit: a red phase long enough to back traffic up beyond the
   queue zone is indistinguishable from congestion by this rule, and arguably
   by a human annotator too.
"""
from __future__ import annotations

import numpy as np

from ..geometry import angle_between
from ..thresholds import TH, CongestionThresholds
from ..tracks import COL
from . import FrameSegment, rule
from .util import runs_to_segments
from .zoneindex import zone_index


def group_lanes_by_direction(zones, tolerance_deg: float) -> list[list[int]]:
    """Cluster lane indices by their authored direction vector.

    Greedy: each lane joins the first group whose representative arrow it
    agrees with. Deterministic in zones.json order.
    """
    groups: list[list[int]] = []
    reps: list[tuple[float, float]] = []
    for i, lane in enumerate(zones.lanes):
        placed = False
        for gi, rep in enumerate(reps):
            if np.degrees(angle_between(lane.direction, rep)) <= tolerance_deg:
                groups[gi].append(i)
                placed = True
                break
        if not placed:
            groups.append([i])
            reps.append(lane.direction)
    return groups


@rule("congestion")
def detect(tracks, zones, th: CongestionThresholds | None = None
           ) -> list[FrameSegment]:
    th = th or TH.congestion
    if zones is None or len(tracks) == 0 or not zones.lanes:
        return []

    veh = tracks.vehicles()
    if len(veh) == 0:
        return []

    data = veh.data
    idx = zone_index(veh, zones)
    frames = data[:, COL["frame_idx"]].astype(np.int64)
    speed = data[:, COL["speed"]]

    # Vehicles lawfully waiting at a signal are invisible to this rule.
    countable = (idx.lane_idx >= 0) & (~idx.in_signal_queue)
    if not countable.any():
        return []

    grid = np.unique(frames)
    if grid.size == 0:
        return []
    frame_pos = {int(f): k for k, f in enumerate(grid)}

    n_lanes = len(zones.lanes)
    total = np.zeros((grid.size, n_lanes), dtype=np.int32)
    slow = np.zeros((grid.size, n_lanes), dtype=np.int32)

    rows = np.flatnonzero(countable)
    for r in rows:
        k = frame_pos[int(frames[r])]
        li = int(idx.lane_idx[r])
        total[k, li] += 1
        if speed[r] <= th.crawl_speed_px_s:
            slow[k, li] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(total > 0, slow / np.maximum(total, 1), 0.0)
    lane_congested = (total >= th.min_vehicles_per_lane) & (frac >= th.slow_fraction)

    groups = group_lanes_by_direction(zones, th.direction_group_tolerance_deg)
    fps = veh.fps or 25.0
    step = max(1, veh.frame_stride)
    out: list[FrameSegment] = []

    for members in groups:
        if not members:
            continue
        # ALL lanes of the direction, as the class definition requires.
        flags = lane_congested[:, members].all(axis=1)
        if not flags.any():
            continue
        runs = runs_to_segments(grid, flags, fps,
                                min_duration_sec=th.min_duration_sec,
                                gap_sec=th.debounce_gap_sec,
                                frame_stride=step)
        for s, e in runs:
            m = (grid >= s) & (grid <= e)
            out.append(FrameSegment(
                s, e, score=1.0, track_ids=(),
                debug={"lanes": [zones.lanes[i].id for i in members],
                       "duration_sec": round((e - s + step) / fps, 2),
                       "peak_vehicles": int(total[m][:, members].sum(axis=1).max())},
            ))
    return out
