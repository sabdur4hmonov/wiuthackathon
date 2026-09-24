"""stopped_vehicle -- a vehicle stationary on the carriageway for >= 10 s,
not queued at a signal.

THE DESIGN PROBLEM: track_id is not a reliable identity.

ByteTrack has no appearance model, so a stationary vehicle can pick up a new id
at any moment -- a pedestrian walks in front of it, another vehicle's box
overlaps, the detector drops it for a few frames. If the 10-second clock were
keyed on track_id, every one of those would reset it, and a vehicle stopped for
30 s would be seen as three 10-second fragments, none of which reaches the bar.
Real stopped vehicles are exactly the ones most likely to be occluded, because
they are stationary while traffic moves around them.

So the clock is keyed on SPATIAL CONTINUITY of the ground point (gx, gy)
instead. Stationary samples are clustered by position and time, across all
track ids. A cluster is one physical stop: whatever id the tracker happened to
attach to it, the vehicle was in that place, not moving, for that long.

SPEED IS A PRE-FILTER, DISPLACEMENT IS THE DECISION. Speed is a differenced
position, so bbox jitter is amplified by 1/dt. The speed gate is loose; the
cluster's max drift from where it started does the real work, because jitter
does not accumulate and genuine creep does.

WHERE SITTING STILL IS LAWFUL. Every signal_queue_zone -- the SB approach,
the NB approach, the NB bus stop and the left-turn box on the real camera --
is excluded, plus a margin around each, so the back of a queue jittering over
a zone edge cannot add up to a stop. Parking bays are outside the carriageway.

BIASED QUIET. Three further gates exist only to keep the class from
over-firing while there are no labels: the stop must be backed by detections
for most of its span; some id in it must be seen elsewhere during the clip (it
arrived or it left); and it must not be part of a queue -- another stationary
vehicle right beside it for most of the stop means traffic is standing, which
is a queue (or congestion), not a stopped vehicle. A detection that never
moves -- street furniture read as a car, or a car parked all clip whose ground
point leaks over the kerb -- never fires. The price is a vehicle stopped for the whole clip with an id that never
moves: accepted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..thresholds import TH, StoppedVehicleThresholds
from ..tracks import COL
from . import FrameSegment, rule
from .util import box_heights, cut_by_frame, min_boundary_distance, sustained_speed
from .zoneindex import zone_index


@dataclass
class _Cluster:
    """One physical stop, accumulated across whatever track ids it wore."""

    first_frame: int
    last_frame: int
    sum_x: float
    sum_y: float
    sum_l: float
    n: int
    anchor_x: float
    anchor_y: float
    ids: set = field(default_factory=set)
    frames: set = field(default_factory=set)

    @property
    def cx(self) -> float:
        return self.sum_x / self.n

    @property
    def cy(self) -> float:
        return self.sum_y / self.n

    @property
    def size(self) -> float:
        return self.sum_l / self.n


def _queued(c: _Cluster, frames, gx, gy, ids, speed, size, th) -> bool:
    """Another stationary vehicle beside this stop for most of it?"""
    during = (frames >= c.first_frame) & (frames <= c.last_frame)
    near = (during & ~np.isin(ids, list(c.ids)) & (ids >= 0)
            & (speed <= th.speed_L_s * size)
            & (np.hypot(gx - c.cx, gy - c.cy) <= th.queue_neighbour_L * c.size))
    return len(np.unique(frames[near])) >= th.queue_share * len(c.frames)


@rule("stopped_vehicle")
def detect(tracks, zones, th: StoppedVehicleThresholds | None = None
           ) -> list[FrameSegment]:
    th = th or TH.stopped_vehicle
    if zones is None or len(tracks) == 0:
        return []

    veh = tracks.vehicles()
    if len(veh) == 0:
        return []

    data = veh.data
    idx = zone_index(veh, zones)

    frames = data[:, COL["frame_idx"]].astype(np.int64)
    gx = data[:, COL["gx"]]
    gy = data[:, COL["gy"]]
    conf = data[:, COL["conf"]]
    ids = data[:, COL["track_id"]].astype(np.int64)
    size = box_heights(data)
    # Movement over seconds, not the jittery per-frame speed. Untracked rows
    # have no history to judge, so they pass and the drift test decides.
    speed = np.where(ids >= 0, sustained_speed(veh), 0.0)

    # On the road, not where sitting still is lawful, confidently a vehicle,
    # and not obviously moving. Whole-table masks: a per-row polygon test in
    # the loop below would cost seconds per video.
    cut = cut_by_frame(data, veh.width or zones.image_width,
                       veh.height or zones.image_height)
    h_frame = veh.height or zones.image_height
    eligible = (idx.on_carriageway & (~idx.in_signal_queue) & ~cut
                & (size >= th.min_box_frac * h_frame)
                & (conf >= th.min_confidence) & (speed <= th.speed_L_s * size))
    candidates = np.flatnonzero(eligible)
    if candidates.size:
        pts = np.column_stack([gx[candidates], gy[candidates]])
        keep = (min_boundary_distance(pts, zones.carriageway)
                >= th.kerb_margin_L * size[candidates])
        if zones.signal_queue_zones:
            keep &= (min_boundary_distance(pts, zones.signal_queue_zones)
                     >= th.queue_margin_L * size[candidates])
        candidates = candidates[keep]
    # Chronological order as a PERMUTATION, not a copy: the shared zone index
    # is aligned with the table's own row order.
    candidates = candidates[np.argsort(frames[candidates], kind="stable")]

    fps = veh.fps or 25.0
    max_gap_frames = th.max_gap_sec * fps

    clusters: list[_Cluster] = []
    open_clusters: list[_Cluster] = []
    # Plain Python scalars: indexing numpy per iteration dominates this loop.
    for x, y, f, s, tid in zip(gx[candidates].tolist(), gy[candidates].tolist(),
                               frames[candidates].tolist(), size[candidates].tolist(),
                               ids[candidates].tolist()):
        open_clusters = [c for c in open_clusters if (f - c.last_frame) <= max_gap_frames]

        best, best_d = None, float("inf")
        for c in open_clusters:
            d = math.hypot(x - c.cx, y - c.cy)
            if d <= th.same_stop_radius_L * c.size and d < best_d:
                # A cluster that has wandered from where it started is a slow
                # creep, not a stop.
                if math.hypot(x - c.anchor_x, y - c.anchor_y) <= th.max_drift_L * c.size:
                    best, best_d = c, d

        if best is None:
            c = _Cluster(f, f, x, y, s, 1, x, y, {tid}, {f})
            clusters.append(c)
            open_clusters.append(c)
        else:
            best.last_frame = f
            best.sum_x += x
            best.sum_y += y
            best.sum_l += s
            best.n += 1
            best.ids.add(tid)
            best.frames.add(f)

    step = max(1, veh.frame_stride)
    rows_of: dict[int, np.ndarray] | None = None
    out: list[FrameSegment] = []
    for c in clusters:
        duration = (c.last_frame - c.first_frame + step) / fps
        if duration < th.min_duration_sec:
            continue
        expected = (c.last_frame - c.first_frame) // step + 1
        coverage = len(c.frames) / max(expected, 1)
        if coverage < th.min_coverage:
            continue

        # Arrived or left during the clip? Only real ids count: every
        # untracked detection shares id -1, so it proves nothing.
        if rows_of is None:
            order = np.argsort(ids, kind="stable")
            uniq, starts = np.unique(ids[order], return_index=True)
            rows_of = dict(zip(uniq.tolist(), np.split(order, starts[1:])))
        reach = 0.0
        for tid in c.ids:
            if tid < 0 or tid not in rows_of:
                continue
            r = rows_of[tid]
            reach = max(reach, float(np.hypot(gx[r] - c.cx, gy[r] - c.cy).max()))
        if reach < th.motion_evidence_L * c.size:
            continue
        if _queued(c, frames, gx, gy, ids, speed, size, th):
            continue

        out.append(FrameSegment(
            c.first_frame, c.last_frame,
            score=min(1.0, duration / max(th.min_duration_sec, 1e-6)),
            track_ids=tuple(sorted(i for i in c.ids if i >= 0)),
            debug={"duration_sec": round(duration, 2),
                   "samples": c.n,
                   "coverage": round(coverage, 2),
                   "n_ids": len(c.ids),
                   "at": (round(c.cx, 1), round(c.cy, 1)),
                   "size_px": round(c.size, 1)},
        ))
    return out
