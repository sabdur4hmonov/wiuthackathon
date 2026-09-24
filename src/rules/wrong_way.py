"""wrong_way -- a vehicle travelling against the direction of its lane.

Lane directions come from zones.json, where each arrow runs along the painted
lane lines with its sign taken from optical flow measured over the real clips
-- so "against the lane" means against where traffic was actually seen to go.

Only PAINTED lanes are judged: lanes that some lane_marking separates. A
lane's direction is only as trustworthy as its paint. On the real camera the
unpainted areas right of the refuge (NB_approach, SB_exit) carry two-way,
queued and turning traffic -- the first real run flagged 11 lawful vehicles
there -- and the one-way leg has no lane lines either. They are treated like
the intersection box: no opinion. A scene with no markings at all falls back
to judging every lane.

Four gates, each guarding a specific false positive:

  SPEED. geometry.angle_between returns 180 degrees for a zero-length vector,
  deliberately, so that a stationary object never reads as travelling WITH a
  lane. Without a speed gate every stopped vehicle reads as travelling AGAINST
  it. The gate is in box heights per second, so it means the same thing at the
  far end of the avenue as it does under the camera.

  ANGLE. 135 degrees. A vehicle making a legal turn is inside its entry lane's
  polygon while already heading across it, roughly 90 degrees off; the bar
  sits well clear of that. Turns through the intersection box are immune
  anyway: the box deliberately has no lane.

  PERSISTENCE. The divergence must last min_duration_sec, bridged across short
  gaps so heading noise cannot split one manoeuvre into fragments.

  PROGRESS. Over the run the vehicle must make net progress AGAINST the lane
  axis of at least min_progress_L of its own box height. Straight-line
  distance alone would count a vehicle nosing sideways in a queue.
"""
from __future__ import annotations

import numpy as np

from ..thresholds import TH, WrongWayThresholds
from ..tracks import COL
from . import FrameSegment, rule
from .util import box_heights, runs_to_segments
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
    conf = data[:, COL["conf"]]
    size = box_heights(data)

    # Angle against the lane for every row at once; same definition as
    # geometry.angle_between, including its 180-degree answer for a zero vector.
    lane_dirs = idx.lane_directions()
    painted = {lid for m in zones.lane_markings for lid in m.separates}
    judged = np.array([(ln.id in painted) or not painted for ln in zones.lanes] + [False])
    vnorm = np.hypot(vx, vy)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = np.where(vnorm > 1e-9,
                          (vx * lane_dirs[:, 0] + vy * lane_dirs[:, 1])
                          / np.maximum(vnorm, 1e-9),
                          -1.0)
    offset_deg = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    opposed = ((speed >= th.min_speed_L_s * size)
               & judged[idx.lane_idx]                 # lane_idx -1 hits the False sentinel
               & (offset_deg >= th.min_angle_deg)
               & (conf >= th.min_confidence))

    fps = veh.fps or 25.0
    step = max(1, veh.frame_stride)
    out: list[FrameSegment] = []

    for tid in np.unique(ids[opposed]):
        if tid < 0:
            continue
        sel = np.flatnonzero(ids == tid)
        if sel.size < 2:
            continue
        sel = sel[np.argsort(frames[sel], kind="stable")]

        runs = runs_to_segments(frames[sel], opposed[sel], fps,
                                min_duration_sec=th.min_duration_sec,
                                gap_sec=th.debounce_gap_sec,
                                frame_stride=step)
        for s, e in runs:
            rows = sel[(frames[sel] >= s) & (frames[sel] <= e) & opposed[sel]]
            if rows.size < 2:
                continue
            axis = lane_dirs[rows].mean(axis=0)
            norm = float(np.hypot(*axis))
            if norm < 1e-6:
                continue
            axis /= norm
            # Net displacement projected on the lane axis; positive = against.
            against = -float((gx[rows[-1]] - gx[rows[0]]) * axis[0]
                             + (gy[rows[-1]] - gy[rows[0]]) * axis[1])
            unit = float(np.median(size[rows]))
            if against < th.min_progress_L * unit:
                continue
            out.append(FrameSegment(
                s, e,
                score=min(1.0, against / max(th.min_progress_L * unit, 1e-6)),
                track_ids=(int(tid),),
                debug={"against_L": round(against / unit, 2),
                       "duration_sec": round((e - s + step) / fps, 2),
                       "lane": zones.lanes[int(np.bincount(idx.lane_idx[rows]).argmax())].id},
            ))
    return out
