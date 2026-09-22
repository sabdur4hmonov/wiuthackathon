"""wrong_way -- a vehicle travelling against the direction of its lane.

Three gates, each guarding a specific false positive:

  SPEED. src.geometry.angle_between returns 180 degrees for a zero-length
  vector, deliberately, so that a stationary object never reads as travelling
  WITH a lane. The consequence is that without a speed gate every parked car
  reads as travelling AGAINST it. This gate is load-bearing: remove it and the
  rule fires on every stopped vehicle in the scene.

  ANGLE. 120 degrees, not 90. A vehicle making a legal turn is inside its entry
  lane's polygon while already heading across it -- a right-angle divergence
  that is completely lawful. The bar has to sit well clear of 90 degrees or
  every turn at the intersection becomes a wrong-way event.

  PERSISTENCE. The divergence must last min_duration_sec AND the vehicle must
  cover min_distance_px against the lane. Duration alone is not enough: a
  vehicle nosing sideways in a queue holds a bad heading for seconds without
  going anywhere. Distance alone is not enough either: one noisy frame at speed
  covers ground. Both together are what a real wrong-way manoeuvre produces.

Note on lane membership: the rule only has an opinion where a lane contains the
ground point. Wherever the authored geometry leaves a gap -- typically the
intersection box -- there is no lane direction to disagree with, so turns
through it are silently immune. That is a property of the zones file, not of
this rule, and it will change when the real zones.json is drawn.
"""
from __future__ import annotations

import numpy as np

from ..thresholds import TH, WrongWayThresholds
from ..tracks import COL
from . import FrameSegment, rule
from .util import runs_to_segments
from .zoneindex import zone_index


@rule("wrong_way")
def detect(tracks, zones, th: WrongWayThresholds | None = None
           ) -> list[FrameSegment]:
    th = th or TH.wrong_way
    if zones is None or len(tracks) == 0 or not zones.lanes:
        return []

    veh = tracks.vehicles()
    if len(veh) == 0:
        return []

    data = veh.data
    idx = zone_index(veh, zones)
    ids = data[:, COL["track_id"]].astype(np.int64)
    frames = data[:, COL["frame_idx"]].astype(np.int64)
    gx, gy = data[:, COL["gx"]], data[:, COL["gy"]]
    vx, vy = data[:, COL["vx"]], data[:, COL["vy"]]
    speed = data[:, COL["speed"]]

    # Angle against the lane, for every row at once. Same definition as
    # geometry.angle_between (including its 180-degree answer for a degenerate
    # vector), but without a Python call per row: at 40k rows the loop costs
    # more than every other rule combined.
    lane_dirs = idx.lane_directions()
    vnorm = np.hypot(vx, vy)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = np.where(vnorm > 1e-9,
                          (vx * lane_dirs[:, 0] + vy * lane_dirs[:, 1])
                          / np.maximum(vnorm, 1e-9),
                          -1.0)
    offset_deg = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    # Gates: fast enough for the heading to mean anything, inside a lane, and
    # opposed by more than a legal turn ever is.
    opposed = ((speed >= th.min_speed_px_s)
               & (idx.lane_idx >= 0)
               & (offset_deg >= th.min_angle_deg))

    fps = veh.fps or 25.0
    step = max(1, veh.frame_stride)
    out: list[FrameSegment] = []

    for tid in np.unique(ids):
        if tid < 0:
            continue
        sel = np.flatnonzero(ids == tid)
        if sel.size < 2:
            continue
        sel = sel[np.argsort(frames[sel], kind="stable")]

        flags = opposed[sel]

        runs = runs_to_segments(frames[sel], flags, fps,
                                min_duration_sec=th.min_duration_sec,
                                gap_sec=th.debounce_gap_sec,
                                frame_stride=step)
        for s, e in runs:
            m = (frames[sel] >= s) & (frames[sel] <= e)
            if not m.any():
                continue
            rows = sel[m]
            # Straight-line displacement over the run, not path length: a
            # vehicle that wobbles in place covers path distance without
            # actually going anywhere against the traffic.
            px = float(np.hypot(gx[rows[-1]] - gx[rows[0]],
                                gy[rows[-1]] - gy[rows[0]]))
            if px < th.min_distance_px:
                continue
            out.append(FrameSegment(
                s, e,
                score=min(1.0, px / max(th.min_distance_px, 1e-6)),
                track_ids=(int(tid),),
                debug={"distance_px": round(px, 1),
                       "duration_sec": round((e - s + step) / fps, 2)},
            ))
    return out
