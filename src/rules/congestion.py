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
samples crawl AND at least min_distinct_slow different tracks are crawling. A
vehicle only counts as crawling once its track has read slow for
slow_persist_sec in a row. That state must hold for min_duration_sec, bridged
across short dips.

3. SAMPLE RATE. At keyframe rate (0.5 s) a moving platoon can alias into a
   track that seems to stand still. Measured against a 0.1 s run of sample_004:
   8 % of the rows called slow were such phantoms, and slow_persist_sec removes
   85 % of those. That is why the rule runs at keyframe rate and wrong_way does
   not: wrong_way reads direction, which aliasing reverses outright.
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


def persistent_slow(data: np.ndarray, slow: np.ndarray, persist_sec: float,
                    sample_sec: float) -> np.ndarray:
    """(N,) slow AND the row's track has read slow on every sample for persist_sec.

    One missed sample is tolerated (gap up to 2.5 samples); a moving sample or
    a longer gap restarts the count. Untracked rows (id -1) never qualify.
    """
    out = np.zeros(slow.shape, dtype=bool)
    if persist_sec <= 0:
        return slow & (data[:, COL["track_id"]] >= 0)
    ids = data[:, COL["track_id"]].astype(np.int64)
    t = data[:, COL["t_sec"]].astype(float)
    max_gap = 2.5 * max(sample_sec, 1e-6)
    order = np.lexsort((t, ids))
    run_start, prev_id, prev_t = None, None, None
    for i in order:
        if ids[i] < 0 or not slow[i]:
            run_start = None
        elif run_start is None or ids[i] != prev_id or t[i] - prev_t > max_gap:
            run_start = t[i]
        prev_id, prev_t = ids[i], t[i]
        # A float tolerance: samples sit on frame_idx / fps, not exact tenths.
        out[i] = run_start is not None and t[i] - run_start >= persist_sec - 1e-3
    return out


@rule("congestion")
def detect(tracks, zones, th: CongestionThresholds | None = None
           ) -> list[FrameSegment]:
    th = th or TH.congestion

    # Speed needs samples no sparser than max_sample_sec (keyframes pass).
    if tracks.frame_stride / (tracks.fps or 25.0) > th.max_sample_sec:
        return []
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
    # left out: counting them would add phantom stationary vehicles. A row
    # only counts once its track has stayed slow for slow_persist_sec, which
    # removes most keyframe-rate aliasing (see the thresholds).
    slow_row = persistent_slow(data, sustained_speed(veh) <= th.crawl_speed_L_s * box_heights(data),
                               th.slow_persist_sec, veh.frame_stride / (veh.fps or 25.0))

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
    # Corroboration: enough DIFFERENT slow vehicles in the window, not one
    # track counted over and over.
    counted_slow = countable & slow_row
    if th.min_distinct_slow > 1:
        ids = data[:, COL["track_id"]].astype(np.int64)
        for gi in range(len(groups)):
            sel = counted_slow & (row_group == gi)
            if not congested[:, gi].any():
                continue
            order = np.argsort(pos[sel], kind="stable")
            p_sel, id_sel = pos[sel][order], ids[sel][order]
            for k in np.nonzero(congested[:, gi])[0]:
                a, b = np.searchsorted(p_sel, [lo[k], hi[k]], side="left")
                if np.unique(id_sel[a:b]).size < th.min_distinct_slow:
                    congested[k, gi] = False

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
