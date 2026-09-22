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

So the clock is keyed on SPATIAL CONTINUITY instead. Stationary samples are
clustered by position and time, across all track ids. A cluster is one physical
stop: whatever id the tracker happened to attach to it, the vehicle was in that
place, not moving, for that long.

The cost is that two different vehicles stopping at the same spot within
max_gap_sec of each other merge into one stop. That trade is right: the spot is
occupied either way, and splitting it would cost a detection that IoU matching
would probably have recovered anyway.

SPEED IS A PRE-FILTER, DISPLACEMENT IS THE DECISION. Speed is a differenced
position, so bbox jitter is amplified by 1/dt -- measurement puts a perfectly
stationary vehicle at roughly 5.6 * jitter_px of apparent speed. A tight speed
gate therefore throws away real stopped vehicles, which are exactly the ones
whose boxes wobble most. So the gate is loose and the cluster's max_drift_px
does the real work: jitter is zero-mean and does not accumulate, while genuine
creep does, and breaks the cluster well before the 10 s bar.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..thresholds import TH, StoppedVehicleThresholds
from ..tracks import COL
from . import FrameSegment, rule
from .zoneindex import zone_index


@dataclass
class _Cluster:
    """One physical stop, accumulated across whatever track ids it wore."""

    first_frame: int
    last_frame: int
    sum_x: float
    sum_y: float
    n: int
    anchor_x: float
    anchor_y: float
    ids: set = field(default_factory=set)

    @property
    def cx(self) -> float:
        return self.sum_x / self.n

    @property
    def cy(self) -> float:
        return self.sum_y / self.n


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

    # A stationary sample only counts if it is on the road and NOT somewhere
    # sitting still is lawful. Computed for every row at once: a per-row
    # polygon test in the loop below costs seconds per video.
    eligible = idx.on_carriageway & (~idx.in_signal_queue)

    frames = data[:, COL["frame_idx"]].astype(np.int64)
    gx = data[:, COL["gx"]]
    gy = data[:, COL["gy"]]
    speed = data[:, COL["speed"]]
    ids = data[:, COL["track_id"]].astype(np.int64)

    # Chronological order as a PERMUTATION, not a copy: the shared zone index
    # is aligned with the table's own row order, and re-sorting the array would
    # desynchronise it (and force the index to be rebuilt).
    candidates = np.flatnonzero(eligible & (speed <= th.speed_px_s))
    candidates = candidates[np.argsort(frames[candidates], kind="stable")]

    fps = veh.fps or 25.0
    max_gap_frames = th.max_gap_sec * fps

    clusters: list[_Cluster] = []
    open_clusters: list[_Cluster] = []

    for k in candidates:
        pt = (float(gx[k]), float(gy[k]))
        f = int(frames[k])
        # Retire clusters that have gone quiet; they can no longer absorb.
        still_open = []
        for c in open_clusters:
            if (f - c.last_frame) <= max_gap_frames:
                still_open.append(c)
        open_clusters = still_open

        best = None
        best_d = float("inf")
        for c in open_clusters:
            d = np.hypot(pt[0] - c.cx, pt[1] - c.cy)
            if d <= th.same_stop_radius_px and d < best_d:
                # Reject a cluster that has wandered too far from where it
                # started: that is a slow creep, not a stop.
                drift = np.hypot(pt[0] - c.anchor_x, pt[1] - c.anchor_y)
                if drift <= th.max_drift_px:
                    best, best_d = c, d

        if best is None:
            c = _Cluster(f, f, pt[0], pt[1], 1, pt[0], pt[1], {int(ids[k])})
            clusters.append(c)
            open_clusters.append(c)
        else:
            best.last_frame = f
            best.sum_x += pt[0]
            best.sum_y += pt[1]
            best.n += 1
            best.ids.add(int(ids[k]))

    step = max(1, veh.frame_stride)
    out: list[FrameSegment] = []
    for c in clusters:
        duration = (c.last_frame - c.first_frame + step) / fps
        if duration < th.min_duration_sec:
            continue
        out.append(FrameSegment(
            c.first_frame, c.last_frame,
            score=min(1.0, duration / max(th.min_duration_sec, 1e-6)),
            track_ids=tuple(sorted(c.ids)),
            debug={"duration_sec": round(duration, 2),
                   "samples": c.n,
                   "n_ids": len(c.ids),
                   "at": (round(c.cx, 1), round(c.cy, 1))},
        ))
    return out
