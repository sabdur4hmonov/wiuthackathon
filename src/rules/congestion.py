"""congestion -- standstill or crawling traffic across one DIRECTION of travel,
judged for the direction as a whole, not lane by lane.

TWO THINGS MAKE THIS HARD.

1. WHAT IS A DIRECTION. zones.json may name it: each lane can carry a
   direction_group ("southbound", "northbound"), and the real camera's file
   does. Lanes with no group (the one-way bottom-left leg) take no part. Only
   when no lane names a group are directions recovered by clustering lane
   arrows, which is what the synthetic four-way scene relies on.

2. A NORMAL RED-LIGHT QUEUE LOOKS EXACTLY LIKE CONGESTION. Same stopped
   vehicles, often the same duration -- a red phase can run 60-90 s, so no
   duration threshold separates them.

   What separates them is WHERE the stopped traffic is. Every vehicle inside a
   signal_queue_zone is excluded from the statistics entirely. A queue waiting
   on red sits inside that zone and is invisible to this rule; only traffic
   outside every queue zone is counted. On the real camera the whole SB
   approach and the NB approach wedge are queue zones, so what is measured is
   the traffic that has already cleared the intersection -- the SB exit and the
   NB departure lanes. A jam there is congestion no red phase explains.

   The honest limit: a jam confined to the approach (a queue that does not
   clear on green) looks like a long red and is not detected. Telling them
   apart needs the signal phase per frame, which Stage 1 does not record.

Over a sliding window_sec, a direction is congested when on average at least
min_vehicles of its vehicles are counted AND at least slow_fraction of those
samples crawl. That state must hold for min_duration_sec, bridged across
short dips.
"""
from __future__ import annotations

import numpy as np

from ..geometry import angle_between
from ..thresholds import TH, CongestionThresholds
from ..tracks import COL
from . import FrameSegment, rule
from .util import box_heights, runs_to_segments, sustained_speed
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


def direction_groups(zones, tolerance_deg: float) -> list[tuple[str, list[int]]]:
    """(name, lane indices) per direction: named groups if zones.json has any."""
    named: dict[str, list[int]] = {}
    for i, lane in enumerate(zones.lanes):
        grp = getattr(lane, "direction_group", None)
        if grp:
            named.setdefault(grp, []).append(i)
    if named:
        return list(named.items())
    return [(f"cluster{k}", g) for k, g in
            enumerate(group_lanes_by_direction(zones, tolerance_deg))]


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
    tracked = data[:, COL["track_id"]] >= 0
    # Crawling judged over seconds of movement, so box jitter on a queued car
    # does not read as motion. Untracked rows carry no speed at all and are
    # left out: counting them would add phantom stationary vehicles.
    slow_row = sustained_speed(veh) <= th.crawl_speed_L_s * box_heights(data)

    groups = direction_groups(zones, th.direction_group_tolerance_deg)
    group_of_lane = np.full(len(zones.lanes) + 1, -1, dtype=np.int64)  # [-1] -> no lane
    for gi, (_name, members) in enumerate(groups):
        group_of_lane[members] = gi
    row_group = group_of_lane[idx.lane_idx]          # lane_idx -1 hits the sentinel

    # Vehicles lawfully waiting at a signal are invisible to this rule.
    countable = (row_group >= 0) & (~idx.in_signal_queue) & tracked
    if not countable.any():
        return []

    grid, pos = np.unique(frames, return_inverse=True)
    total = np.zeros((grid.size, len(groups)), dtype=np.int64)
    slow = np.zeros((grid.size, len(groups)), dtype=np.int64)
    np.add.at(total, (pos[countable], row_group[countable]), 1)
    np.add.at(slow, (pos[countable & slow_row], row_group[countable & slow_row]), 1)

    # Judged over a sliding window, not frame by frame: with free-flowing cars
    # passing a crawling lane every couple of seconds, single frames between
    # them look 100 % slow, and gap-bridging would stitch those into a jam.
    fps = veh.fps or 25.0
    step = max(1, veh.frame_stride)
    half = th.window_sec * fps / 2.0
    lo = np.searchsorted(grid, grid - half, side="left")
    hi = np.searchsorted(grid, grid + half, side="right")
    csum_t = np.vstack([np.zeros((1, len(groups)), np.int64), np.cumsum(total, axis=0)])
    csum_s = np.vstack([np.zeros((1, len(groups)), np.int64), np.cumsum(slow, axis=0)])
    win_total = csum_t[hi] - csum_t[lo]
    win_slow = csum_s[hi] - csum_s[lo]
    per_frame = win_total / np.maximum(hi - lo, 1)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(win_total > 0, win_slow / np.maximum(win_total, 1), 0.0)
    congested = (per_frame >= th.min_vehicles) & (frac >= th.slow_fraction)

    out: list[FrameSegment] = []
    for gi, (name, members) in enumerate(groups):
        if not congested[:, gi].any():
            continue
        runs = runs_to_segments(grid, congested[:, gi], fps,
                                min_duration_sec=th.min_duration_sec,
                                gap_sec=th.debounce_gap_sec,
                                frame_stride=step)
        for s, e in runs:
            m = (grid >= s) & (grid <= e)
            out.append(FrameSegment(
                s, e, score=1.0, track_ids=(),
                debug={"direction": name,
                       "lanes": [zones.lanes[i].id for i in members],
                       "duration_sec": round((e - s + step) / fps, 2),
                       "peak_vehicles": int(total[m, gi].max())},
            ))
    return out
