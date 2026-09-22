"""Vectorised zone membership for a whole track table at once.

Every rule asks the same questions of every row: is this ground point on the
carriageway, in a crossing, in a signal queue zone, and which lane is it in?
Asking them one row at a time costs a Python-level polygon test per row per
polygon, which is seconds per video -- enough to defeat the point of caching
Stage 1 and iterating on Stage 2.

So each question is answered once, for all rows, as a numpy array. Rules then
index by row. Lane assignment reproduces Zones.lane_of exactly: FIRST polygon
that contains the point wins, which is why zones.json lane order is meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import points_in_polygon
from ..tracks import COL
from ..zones import Lane, Zones


@dataclass
class ZoneIndex:
    """Per-row zone membership, aligned with the array it was built from."""

    on_carriageway: np.ndarray      # (N,) bool
    in_crossing: np.ndarray         # (N,) bool
    in_signal_queue: np.ndarray     # (N,) bool
    lane_idx: np.ndarray            # (N,) int, -1 where no lane contains it
    lanes: tuple[Lane, ...]

    @classmethod
    def build(cls, data: np.ndarray, zones: Zones) -> "ZoneIndex":
        """Index the ground points of a raw (N, len(COLUMNS)) array."""
        n = int(data.shape[0])
        if n == 0:
            empty_b = np.zeros(0, dtype=bool)
            return cls(empty_b, empty_b.copy(), empty_b.copy(),
                       np.zeros(0, dtype=np.int64), zones.lanes)

        pts = np.column_stack([data[:, COL["gx"]], data[:, COL["gy"]]]).astype(float)

        def any_of(areas) -> np.ndarray:
            acc = np.zeros(n, dtype=bool)
            for a in areas:
                acc |= points_in_polygon(pts, a.polygon)
            return acc

        lane_idx = np.full(n, -1, dtype=np.int64)
        for i, lane in enumerate(zones.lanes):
            # First match wins, so only fill rows still unassigned. This is
            # what makes Zones.lane_of and this index agree.
            unassigned = lane_idx < 0
            if not unassigned.any():
                break
            hit = np.zeros(n, dtype=bool)
            hit[unassigned] = points_in_polygon(pts[unassigned], lane.polygon)
            lane_idx[hit] = i

        return cls(
            on_carriageway=any_of(zones.carriageway),
            in_crossing=any_of(zones.crossings),
            in_signal_queue=any_of(zones.signal_queue_zones),
            lane_idx=lane_idx,
            lanes=zones.lanes,
        )

    def lane_at(self, row: int) -> Lane | None:
        i = int(self.lane_idx[row])
        return self.lanes[i] if i >= 0 else None

    def lane_directions(self) -> np.ndarray:
        """(N, 2) unit lane direction per row; (0, 0) where there is no lane."""
        dirs = np.zeros((self.lane_idx.size, 2), dtype=float)
        have = self.lane_idx >= 0
        if have.any():
            table = np.array([ln.direction for ln in self.lanes], dtype=float)
            dirs[have] = table[self.lane_idx[have]]
        return dirs


def zone_index(tracks, zones: Zones) -> ZoneIndex:
    """ZoneIndex for a TrackTable, built once and shared across rules.

    Four rules ask the same membership questions of the same rows. Building the
    index per rule costs roughly as much as every rule combined, so it is
    memoised on the table instance. Keyed by the Zones object's identity, so a
    different scene never serves a stale index.
    """
    key = ("zone_index", id(zones))
    hit = tracks._memo.get(key)
    if hit is not None:
        return hit
    idx = ZoneIndex.build(tracks.data, zones)
    tracks._memo[key] = idx
    return idx
