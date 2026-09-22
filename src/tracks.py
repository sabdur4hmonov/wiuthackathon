"""The tracks table: the ONLY thing Stage 1 hands to Stage 2.

One row per (processed frame, tracked object). Stored as a dense float32 matrix
plus a column-name list, because that is what both the cache and the rules want:

* rules do vectorised per-track slicing, which a table gives for free;
* a float32 matrix serialises to .npz with no extra dependency and no schema
  drift, unlike pickle;
* adding a derived column later does not invalidate anything structurally --
  only the perception config hash, which is exactly the right blast radius.

Everything derived (ground point, velocity, speed, heading) is computed ONCE
here, at cache-build time, so Stage 2 never recomputes kinematics while the
author iterates on rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from . import geometry as g

# Column order is part of the on-disk format. Append only; never reorder.
COLUMNS: tuple[str, ...] = (
    "frame_idx",   # index into the ORIGINAL video, not the strided sequence
    "t_sec",       # frame_idx / fps
    "track_id",    # tracker id; -1 for an untracked detection
    "cls",         # COCO class id
    "conf",
    "x1", "y1", "x2", "y2",
    "gx", "gy",    # ground-contact point: bottom-centre of the box
    "vx", "vy",    # px/sec, smoothed over PerceptionConfig.velocity_window
    "speed",       # hypot(vx, vy), px/sec
    "heading",     # atan2(vy, vx), radians, image space
)
COL: dict[str, int] = {name: i for i, name in enumerate(COLUMNS)}

# COCO ids we keep. Kept here rather than in config because the mapping from id
# to meaning is a property of the weights, not a tunable.
COCO_NAMES: dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
}
VEHICLE_CLASSES = frozenset({1, 2, 3, 5, 7})
PERSON_CLASSES = frozenset({0})


@dataclass
class TrackTable:
    """A whole video's tracks, plus the metadata a rule needs for time maths."""

    data: np.ndarray                 # (N, len(COLUMNS)) float32
    fps: float
    duration: float
    n_frames: int
    width: int
    height: int
    frame_stride: int
    complete: bool = True            # False when perception stopped on budget
    processed_until_sec: float = 0.0  # last timestamp actually looked at

    # -- construction -------------------------------------------------------
    @classmethod
    def empty(cls, fps: float = 25.0, duration: float = 0.0, n_frames: int = 0,
              width: int = 0, height: int = 0, frame_stride: int = 1) -> "TrackTable":
        return cls(np.zeros((0, len(COLUMNS)), dtype=np.float32), fps, duration,
                   n_frames, width, height, frame_stride, True, 0.0)

    # -- access -------------------------------------------------------------
    def __len__(self) -> int:
        return int(self.data.shape[0])

    def col(self, name: str) -> np.ndarray:
        return self.data[:, COL[name]]

    @property
    def track_ids(self) -> np.ndarray:
        if not len(self):
            return np.empty(0, dtype=np.int64)
        ids = np.unique(self.col("track_id").astype(np.int64))
        return ids[ids >= 0]

    def rows_for_track(self, track_id: int) -> np.ndarray:
        """All rows of one track, sorted by time."""
        if not len(self):
            return np.zeros((0, len(COLUMNS)), dtype=np.float32)
        m = self.col("track_id").astype(np.int64) == int(track_id)
        rows = self.data[m]
        return rows[np.argsort(rows[:, COL["t_sec"]], kind="stable")]

    def rows_at_frame(self, frame_idx: int) -> np.ndarray:
        m = self.col("frame_idx").astype(np.int64) == int(frame_idx)
        return self.data[m]

    def iter_tracks(self) -> Iterator[tuple[int, np.ndarray]]:
        """(track_id, rows) for every track, rows time-sorted."""
        for tid in self.track_ids:
            yield int(tid), self.rows_for_track(int(tid))

    def of_classes(self, classes: frozenset[int]) -> "TrackTable":
        """A view restricted to certain COCO classes, metadata preserved."""
        if not len(self):
            return self
        m = np.isin(self.col("cls").astype(np.int64), list(classes))
        return TrackTable(self.data[m], self.fps, self.duration, self.n_frames,
                          self.width, self.height, self.frame_stride,
                          self.complete, self.processed_until_sec)

    def vehicles(self) -> "TrackTable":
        return self.of_classes(VEHICLE_CLASSES)

    def persons(self) -> "TrackTable":
        return self.of_classes(PERSON_CLASSES)

    def summary(self) -> str:
        return (f"tracks: {len(self)} rows, {len(self.track_ids)} ids, "
                f"{self.duration:.1f}s @ {self.fps:.2f}fps, stride {self.frame_stride}"
                + ("" if self.complete
                   else f", TRUNCATED at {self.processed_until_sec:.1f}s"))


# ---------------------------------------------------------------------------
# derived kinematics
# ---------------------------------------------------------------------------
def compute_kinematics(data: np.ndarray, window: int) -> np.ndarray:
    """Fill gx, gy, vx, vy, speed, heading in place and return the array.

    Velocity is a centred finite difference over up to `window` processed
    samples of the SAME track. A single-frame difference on a jittery CCTV box
    is mostly detector noise, and every downstream rule thresholds on speed or
    heading, so the smoothing is not cosmetic.
    """
    if data.shape[0] == 0:
        return data

    data[:, COL["gx"]] = (data[:, COL["x1"]] + data[:, COL["x2"]]) / 2.0
    data[:, COL["gy"]] = data[:, COL["y2"]]

    ids = data[:, COL["track_id"]].astype(np.int64)
    for tid in np.unique(ids):
        if tid < 0:
            continue
        idx = np.where(ids == tid)[0]
        rows = data[idx]
        order = np.argsort(rows[:, COL["t_sec"]], kind="stable")
        idx = idx[order]
        t = data[idx, COL["t_sec"]]
        gx = data[idx, COL["gx"]]
        gy = data[idx, COL["gy"]]
        n = idx.size
        if n < 2:
            continue
        half = max(1, window // 2)
        vx = np.zeros(n, dtype=np.float32)
        vy = np.zeros(n, dtype=np.float32)
        for k in range(n):
            lo, hi = max(0, k - half), min(n - 1, k + half)
            dt = t[hi] - t[lo]
            if dt <= 1e-6:
                # Fall back to the widest span available rather than divide by ~0.
                lo, hi = 0, n - 1
                dt = t[hi] - t[lo]
                if dt <= 1e-6:
                    continue
            vx[k] = (gx[hi] - gx[lo]) / dt
            vy[k] = (gy[hi] - gy[lo]) / dt
        data[idx, COL["vx"]] = vx
        data[idx, COL["vy"]] = vy
        data[idx, COL["speed"]] = np.hypot(vx, vy)
        data[idx, COL["heading"]] = np.arctan2(vy, vx)
    return data


def make_row(frame_idx: int, t_sec: float, track_id: int, cls: int, conf: float,
             x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """One detection row with derived columns left at zero."""
    r = np.zeros(len(COLUMNS), dtype=np.float32)
    r[COL["frame_idx"]] = frame_idx
    r[COL["t_sec"]] = t_sec
    r[COL["track_id"]] = track_id
    r[COL["cls"]] = cls
    r[COL["conf"]] = conf
    r[COL["x1"]], r[COL["y1"]], r[COL["x2"]], r[COL["y2"]] = x1, y1, x2, y2
    gx, gy = g.bbox_ground_point(x1, y1, x2, y2)
    r[COL["gx"]], r[COL["gy"]] = gx, gy
    return r
