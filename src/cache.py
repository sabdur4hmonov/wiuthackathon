"""Perception cache -- a DEV TOOL, never a correctness dependency.

Stage 1 costs seconds per video-second; Stage 2 costs milliseconds. Iterating on
rules fifty times must not re-run the detector fifty times. That is the entire
purpose of this module.

The hard rule: a cache failure must never change predictions. The judges' box
may have a read-only checkout, a full disk, or no writable temp dir. So every
read and write is wrapped, every failure degrades to "recompute", and nothing
in here raises into the pipeline. Entries are keyed on
(video content hash, perception config hash), so editing PerceptionConfig or
swapping the video invalidates automatically -- there is no manual "clear the
cache" step to forget.

Location: $WIUT_CACHE_DIR, else the OS temp dir. Never inside the repo, so a
stale entry can never be committed and shipped.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .tracks import COLUMNS, TrackTable

CACHE_ENV_VAR = "WIUT_CACHE_DIR"
DISABLE_ENV_VAR = "WIUT_NO_CACHE"
_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# location and policy
# ---------------------------------------------------------------------------
def cache_dir() -> Path | None:
    """Where entries live, or None when caching is off or unusable."""
    if os.environ.get(DISABLE_ENV_VAR, "").strip() not in ("", "0", "false", "False"):
        return None
    raw = os.environ.get(CACHE_ENV_VAR, "").strip()
    base = Path(raw) if raw else Path(tempfile.gettempdir()) / "wiut_cv_cache"
    try:
        base.mkdir(parents=True, exist_ok=True)
        # Prove it is writable now rather than discovering it mid-run.
        probe = base / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return base
    except Exception:
        return None


def is_enabled() -> bool:
    return cache_dir() is not None


# ---------------------------------------------------------------------------
# keys
# ---------------------------------------------------------------------------
def video_hash(video_path: str | Path, chunk: int = 4 << 20) -> str:
    """sha256 of the file's bytes.

    Full-content, not size+mtime: a re-encode that keeps the size would
    otherwise serve stale tracks, and a copy that changes mtime would
    needlessly miss. Streaming at 4 MB keeps memory flat; on a ~100 MB clip this
    costs a fraction of a second against a budget of minutes, and it is skipped
    entirely when the cache is disabled.
    """
    h = hashlib.sha256()
    with open(video_path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:24]


def entry_path(vhash: str, chash: str) -> Path | None:
    base = cache_dir()
    if base is None:
        return None
    return base / f"tracks_{vhash}_{chash}_v{_FORMAT_VERSION}.npz"


# ---------------------------------------------------------------------------
# read / write -- both fail soft
# ---------------------------------------------------------------------------
def load(vhash: str, chash: str) -> TrackTable | None:
    """Return the cached table, or None on any miss, corruption or error."""
    p = entry_path(vhash, chash)
    if p is None or not p.exists():
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            cols = [str(c) for c in z["columns"]]
            if tuple(cols) != COLUMNS:
                return None                      # schema drifted; recompute
            meta = json.loads(str(z["meta"].item()))
            data = np.asarray(z["data"], dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != len(COLUMNS):
            return None
        return TrackTable(
            data=data,
            fps=float(meta["fps"]),
            duration=float(meta["duration"]),
            n_frames=int(meta["n_frames"]),
            width=int(meta["width"]),
            height=int(meta["height"]),
            frame_stride=int(meta["frame_stride"]),
            complete=bool(meta.get("complete", True)),
            processed_until_sec=float(meta.get("processed_until_sec", 0.0)),
        )
    except Exception:
        return None


def store(vhash: str, chash: str, table: TrackTable) -> bool:
    """Persist a table. Returns whether it was written; never raises.

    A truncated table (perception stopped on budget) is NOT cached: it is an
    artefact of one machine being slow on one run, and serving it back later
    would silently cap every future run at the same point.
    """
    if not table.complete:
        return False
    p = entry_path(vhash, chash)
    if p is None:
        return False
    tmp = p.with_suffix(".npz.tmp")
    try:
        meta = {
            "fps": table.fps, "duration": table.duration,
            "n_frames": table.n_frames, "width": table.width,
            "height": table.height, "frame_stride": table.frame_stride,
            "complete": table.complete,
            "processed_until_sec": table.processed_until_sec,
        }
        # Write through a handle: np.savez_compressed appends ".npz" to a PATH
        # whose name does not already end in it, which would leave the temp file
        # somewhere the rename below cannot find.
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh,
                data=table.data.astype(np.float32),
                columns=np.array(COLUMNS, dtype=object).astype(str),
                meta=np.array(json.dumps(meta)),
            )
        # Atomic replace so a crash mid-write cannot leave a half file that a
        # later run would read as valid.
        os.replace(tmp, p)
        return True
    except Exception:
        try:
            Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass
        return False


def clear() -> int:
    """Delete every entry. Returns how many were removed. Dev convenience."""
    base = cache_dir()
    if base is None:
        return 0
    n = 0
    for p in base.glob("tracks_*.npz"):
        try:
            p.unlink()
            n += 1
        except Exception:
            pass
    return n


def describe() -> str:
    base = cache_dir()
    if base is None:
        return "cache: DISABLED (recomputing every run)"
    entries = list(base.glob("tracks_*.npz"))
    size = sum(p.stat().st_size for p in entries) / (1 << 20) if entries else 0.0
    return f"cache: {base} ({len(entries)} entries, {size:.1f} MB)"
