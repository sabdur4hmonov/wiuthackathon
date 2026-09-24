"""jaywalking -- a pedestrian on the carriageway outside a crossing.

The rule is a set difference -- ground point on the carriageway and in no
crossing polygon -- plus the gates that keep it from firing on things that
merely look like that:

  CROSSINGS. Every crossing polygon is excluded: on the real camera that is
  both halves of the avenue crossing (either side of the refuge, which is off
  the carriageway anyway) and the diagonal crossing over the bottom-left leg.
  A margin of half a body height around each is excluded too, so someone
  walking on the edge of the stripes does not flicker in and out.

  KERB. The ground point must be a full body height inside a real kerb edge
  (edges where the picture simply ends do not count). Someone waiting on the
  kerb, or boarding a bus, jitters across it constantly.

  RIDERS AND OCCUPANTS. A person box that sits mostly inside a vehicle box in
  the same frame is a motorcyclist, a cyclist or a passenger, not a
  pedestrian. Riders are the main source of "people" on the carriageway.

  FRAME EDGE. A box cut off by the edge of the picture has the frame edge
  for a ground point; it is ignored.

  PERSISTENCE. Runs shorter than min_duration_sec are dropped, and short gaps
  are bridged so one crossing is one event.

KNOWN FALSE POSITIVE, ACCEPTED: a person walking parallel to a zebra but
clearly outside its stripes is flagged. It happens on the diagonal crossing.

Unlike the vehicle rules, this one is keyed on track_id: pedestrians are not
stationary for long, so spatial clustering would merge two people passing each
other. An id switch mid-crossing splits the event; Stage 3's merge_gap_sec
repairs most of that.
"""
from __future__ import annotations

import numpy as np

from ..thresholds import TH, JaywalkingThresholds
from ..tracks import COL
from . import FrameSegment, rule
from .util import box_heights, cut_by_frame, min_boundary_distance, runs_to_segments
from .zoneindex import zone_index


def _inside_a_vehicle(persons: np.ndarray, rows: np.ndarray,
                      vehicles: np.ndarray, share: float) -> np.ndarray:
    """For each person row in `rows`: is >= `share` of its box inside a vehicle box
    in the same frame?"""
    out = np.zeros(rows.size, dtype=bool)
    if rows.size == 0 or vehicles.shape[0] == 0:
        return out
    vf = vehicles[:, COL["frame_idx"]].astype(np.int64)
    order = np.argsort(vf, kind="stable")
    pf = persons[rows, COL["frame_idx"]].astype(np.int64)
    lo = np.searchsorted(vf[order], pf, side="left")
    counts = np.searchsorted(vf[order], pf, side="right") - lo
    if counts.sum() == 0:
        return out
    # Every (person row, same-frame vehicle) pair at once.
    k = np.repeat(np.arange(rows.size), counts)
    first = np.repeat(np.cumsum(counts) - counts, counts)
    v = order[np.repeat(lo, counts) + np.arange(counts.sum()) - first]
    c = [COL["x1"], COL["y1"], COL["x2"], COL["y2"]]
    pb, vb = persons[rows][:, c][k], vehicles[v][:, c]
    iw = np.clip(np.minimum(vb[:, 2], pb[:, 2]) - np.maximum(vb[:, 0], pb[:, 0]), 0, None)
    ih = np.clip(np.minimum(vb[:, 3], pb[:, 3]) - np.maximum(vb[:, 1], pb[:, 1]), 0, None)
    area = np.maximum((pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1]), 1e-6)
    np.logical_or.at(out, k, (iw * ih) / area >= share)
    return out


@rule("jaywalking")
def detect(tracks, zones, th: JaywalkingThresholds | None = None
           ) -> list[FrameSegment]:
    th = th or TH.jaywalking
    if zones is None or len(tracks) == 0:
        return []

    peds = tracks.persons()
    if len(peds) == 0:
        return []

    data = peds.data
    idx = zone_index(peds, zones)
    ids = data[:, COL["track_id"]].astype(np.int64)
    frames = data[:, COL["frame_idx"]].astype(np.int64)
    conf = data[:, COL["conf"]]
    size = box_heights(data)

    # A box cut off by the frame edge has no trustworthy ground point.
    cut = cut_by_frame(data, peds.width or zones.image_width,
                       peds.height or zones.image_height)
    on_road = (idx.on_carriageway & (~idx.in_crossing) & (conf >= th.min_confidence)
               & ~cut)
    cand = np.flatnonzero(on_road)
    if cand.size:
        pts = np.column_stack([data[cand, COL["gx"]], data[cand, COL["gy"]]])
        ok = min_boundary_distance(pts, zones.carriageway) >= th.kerb_margin_L * size[cand]
        if zones.crossings:
            ok &= (min_boundary_distance(pts, zones.crossings)
                   >= th.crossing_margin_L * size[cand])
        ok &= ~_inside_a_vehicle(data, cand, tracks.vehicles().data, th.rider_overlap)
        on_road[cand[~ok]] = False

    fps = peds.fps or 25.0
    step = max(1, peds.frame_stride)
    out: list[FrameSegment] = []

    for tid in np.unique(ids[on_road]):
        if tid < 0:
            continue
        sel = np.flatnonzero(ids == tid)
        sel = sel[np.argsort(frames[sel], kind="stable")]
        runs = runs_to_segments(frames[sel], on_road[sel], fps,
                                min_duration_sec=th.min_duration_sec,
                                gap_sec=th.debounce_gap_sec,
                                frame_stride=step)
        for s, e in runs:
            duration = (e - s + step) / fps
            out.append(FrameSegment(
                s, e, score=1.0, track_ids=(int(tid),),
                debug={"duration_sec": round(duration, 2)},
            ))
    return out
