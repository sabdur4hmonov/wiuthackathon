"""Helpers shared by the rules. Pure functions over numpy, no zones knowledge.

The one that matters most is `runs_to_segments`: nearly every rule is "compute a
per-frame boolean, then find the sustained runs of it", and getting the
debouncing wrong is the difference between one clean event and forty blips.
"""
from __future__ import annotations

import numpy as np

from ..geometry import distances_to_boundary
from ..tracks import COL
from . import FrameSegment


def box_heights(data: np.ndarray) -> np.ndarray:
    """(N,) box height per row, the unit ("L") every distance threshold uses."""
    return np.maximum(data[:, COL["y2"]] - data[:, COL["y1"]], 1.0)


def cut_by_frame(data: np.ndarray, width: float, height: float,
                 margin: float = 2.0) -> np.ndarray:
    """(N,) True where a box touches the left, right or bottom frame edge.

    Such a box is truncated, so its bottom-centre is where the picture ends,
    not where the object meets the road. Every false stopped_vehicle and three
    false jaywalking events on the first real run had one.
    """
    return ((data[:, COL["x1"]] <= margin) | (data[:, COL["x2"]] >= width - margin)
            | (data[:, COL["y2"]] >= height - margin))


def sustained_speed(tracks, window_sec: float = 2.0) -> np.ndarray:
    """(N,) speed from net displacement over +-window/2 of each row, px/s.

    The table's own `speed` is a ~0.3 s centred difference, which box jitter
    dominates for a slow vehicle. Rules that ask "is this vehicle crawling"
    need movement over seconds, where jitter averages out and real creep
    accumulates. Untracked rows (id -1) have no history: they get +inf, so no
    rule can mistake them for a stationary vehicle.

    Memoised on the table: two rules ask for it.
    """
    key = ("sustained_speed", float(window_sec))
    hit = tracks._memo.get(key)
    if hit is not None:
        return hit
    data = tracks.data
    out = np.full(data.shape[0], np.inf)
    ids = data[:, COL["track_id"]].astype(np.int64)
    t = data[:, COL["t_sec"]].astype(float)
    order = np.lexsort((t, ids))
    uniq, starts = np.unique(ids[order], return_index=True)
    half = window_sec / 2.0
    for tid, rows in zip(uniq.tolist(), np.split(order, starts[1:])):
        if tid < 0:
            continue
        tt = t[rows]
        if rows.size < 2 or tt[-1] - tt[0] <= 1e-6:
            continue
        x, y = data[rows, COL["gx"]], data[rows, COL["gy"]]
        lo = np.clip(tt - half, tt[0], tt[-1])
        hi = np.clip(tt + half, tt[0], tt[-1])
        dt = np.maximum(hi - lo, 1e-6)
        out[rows] = np.hypot(np.interp(hi, tt, x) - np.interp(lo, tt, x),
                             np.interp(hi, tt, y) - np.interp(lo, tt, y)) / dt
    tracks._memo[key] = out
    return out


def min_boundary_distance(pts: np.ndarray, areas) -> np.ndarray:
    """(N,) distance from each point to the nearest edge of any of `areas`."""
    out = np.full(len(pts), np.inf)
    for a in areas:
        out = np.minimum(out, distances_to_boundary(
            pts, a.polygon, getattr(a, "frame_edges", None) or None))
    return out


def runs_to_segments(frames: np.ndarray, flags: np.ndarray, fps: float,
                     min_duration_sec: float = 0.0,
                     gap_sec: float = 0.0,
                     frame_stride: int = 1) -> list[tuple[int, int]]:
    """Frames where `flags` is True, grouped into sustained runs.

    Args:
        frames: frame indices, ascending, not necessarily contiguous.
        flags: per-frame boolean, same length.
        gap_sec: runs separated by at most this are bridged. This absorbs the
            single-frame dropouts that a real detector produces constantly;
            without it every occlusion splits one event into two, and two
            half-length segments each score worse against tIoU 0.7 than one
            whole one.
        min_duration_sec: runs shorter than this are dropped.

    Returns closed (start_frame, end_frame) intervals.
    """
    frames = np.asarray(frames).astype(np.int64).ravel()
    flags = np.asarray(flags).astype(bool).ravel()
    if frames.size == 0 or frames.size != flags.size:
        return []

    order = np.argsort(frames, kind="stable")
    frames, flags = frames[order], flags[order]

    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return []

    fps = fps or 25.0
    gap_frames = gap_sec * fps
    runs: list[list[int]] = []
    for i in idx:
        f = int(frames[i])
        if runs and (f - runs[-1][1]) <= gap_frames:
            runs[-1][1] = f
        else:
            runs.append([f, f])

    step = max(1, int(frame_stride))
    out = []
    for s, e in runs:
        # A run of one sample still covers a stride's worth of time.
        if (e - s + step) / fps >= min_duration_sec:
            out.append((s, e))
    return out


def to_segments(runs: list[tuple[int, int]], score: float = 1.0,
                track_ids: tuple[int, ...] = (),
                debug: dict | None = None) -> list[FrameSegment]:
    return [FrameSegment(s, e, score, track_ids, dict(debug or {}))
            for s, e in runs]


def frame_grid(tracks) -> np.ndarray:
    """Every frame index perception actually sampled, ascending and unique.

    Rules that aggregate across objects (congestion) need a common time axis;
    they must not invent frames perception never looked at.
    """
    if len(tracks) == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(tracks.col("frame_idx").astype(np.int64))


def span_seconds(start_frame: int, end_frame: int, fps: float,
                 frame_stride: int = 1) -> float:
    fps = fps or 25.0
    return (end_frame - start_frame + max(1, int(frame_stride))) / fps
