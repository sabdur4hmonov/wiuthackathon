"""jaywalking -- a pedestrian on the carriageway outside a crossing.

The rule itself is a set difference: ground point in carriageway, and not in any
crossing polygon. The work is all in the debouncing.

A pedestrian's ground point is the bottom-centre of a small, low-confidence box
at CCTV range. It jitters across the kerb line constantly while someone waits to
cross, and a naive implementation emits a dozen sub-second blips per pedestrian
-- each one a false positive, and each one competing for the same ground-truth
segment under greedy IoU matching. So runs are bridged across short gaps and
anything under min_duration_sec is dropped.

Unlike the vehicle rules, this one is keyed on track_id: pedestrians are not
stationary for long, so spatial clustering would merge two people passing each
other. The cost is that an id switch mid-crossing splits the event, which the
gap-bridging in Stage 3 (merge_gap_sec) then partly repairs.
"""
from __future__ import annotations

import numpy as np

from ..thresholds import TH, JaywalkingThresholds
from ..tracks import COL
from . import FrameSegment, rule
from .util import runs_to_segments
from .zoneindex import zone_index


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

    # On the road, not on a crossing, and confidently a person.
    on_road = idx.on_carriageway & (~idx.in_crossing) & (conf >= th.min_confidence)

    fps = peds.fps or 25.0
    step = max(1, peds.frame_stride)
    out: list[FrameSegment] = []

    for tid in np.unique(ids):
        if tid < 0:
            continue
        sel = np.flatnonzero(ids == tid)
        sel = sel[np.argsort(frames[sel], kind="stable")]
        flags = on_road[sel]

        runs = runs_to_segments(frames[sel], flags, fps,
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
