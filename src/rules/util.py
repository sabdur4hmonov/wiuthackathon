"""Helpers shared by the rules. Pure functions over numpy, no zones knowledge.

The one that matters most is `runs_to_segments`: nearly every rule is "compute a
per-frame boolean, then find the sustained runs of it", and getting the
debouncing wrong is the difference between one clean event and forty blips.
"""
from __future__ import annotations

import numpy as np

from . import FrameSegment


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
