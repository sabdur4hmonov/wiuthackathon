"""Part A orchestration: the one place the three stages are wired together.

Contract with the harness: this function must NEVER raise and must NEVER
overrun. Either failure mode costs both Part A and Part B for the video, since
run_submission.py replaces the whole entry with {"events": [], "risk": []}.
So every step is guarded and the budget is consulted between stages.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

from . import cache as cache_mod
from .budget import Budget
from .config import CFG, enforce_offline, perception_config_hash, seed_everything
from .postprocess import finalise
from .rules import run_rules
from .tracks import TrackTable
from .zones import ZonesError, Zones, load_zones


def detect_events(video_path: str, verbose: bool = True,
                  use_cache: bool = True) -> list[list]:
    """Full Part A pass. Returns harness-ready events; [] on any problem."""
    t0 = time.perf_counter()
    enforce_offline()
    seed_everything()

    try:
        budget = Budget.for_video(video_path, t0=t0)
    except Exception as e:
        print(f"[pipeline] cannot probe {video_path}: {e}", file=sys.stderr)
        return []

    try:
        tracks = _get_tracks(video_path, budget, use_cache=use_cache, verbose=verbose)
        zones = _get_zones(tracks, verbose=verbose)

        with budget.stage("rules"):
            raw = run_rules(tracks, zones)
        with budget.stage("postprocess"):
            events = finalise(raw, budget.duration)
    except Exception:
        print("[pipeline] Part A failed, returning []:\n" + traceback.format_exc(),
              file=sys.stderr)
        events = []

    if verbose:
        print(budget.format_report(), file=sys.stderr)
    return events


def _get_tracks(video_path: str, budget: Budget, use_cache: bool,
                verbose: bool) -> TrackTable:
    """Cache lookup, else run perception. A cache failure only costs time."""
    from .perception import run_perception

    vhash = chash = None
    if use_cache and cache_mod.is_enabled():
        try:
            with budget.stage("cache_hash"):
                vhash = cache_mod.video_hash(video_path)
            chash = perception_config_hash()
            hit = cache_mod.load(vhash, chash)
            if hit is not None:
                budget.note("perception", 0.0, f"cache HIT {vhash}/{chash}")
                if verbose:
                    print(f"[cache] hit: {hit.summary()}", file=sys.stderr)
                return hit
        except Exception as e:
            # Never let the dev tool affect the answer.
            if verbose:
                print(f"[cache] lookup failed, recomputing: {e!r}", file=sys.stderr)
            vhash = None

    tracks = run_perception(video_path, budget, verbose=verbose)

    if use_cache and vhash and chash:
        try:
            if cache_mod.store(vhash, chash, tracks):
                if verbose:
                    print(f"[cache] stored {vhash}/{chash}", file=sys.stderr)
        except Exception:
            pass
    return tracks


def _get_zones(tracks: TrackTable, verbose: bool) -> Zones | None:
    """Load the scene geometry, rescaled to this video. None if un-authored.

    Returning None rather than raising keeps CP0 runnable before the geometry
    exists. Rules must treat None as "no scene knowledge" and emit nothing,
    which is the correct behaviour: a guessed lane is worse than no lane.
    """
    size = (tracks.width, tracks.height) if tracks.width and tracks.height else None
    try:
        z = load_zones(frame_size=size)
        if verbose:
            print(f"[zones] {z.summary()}", file=sys.stderr)
        return z
    except ZonesError as e:
        if verbose:
            # One line, not the full list: this runs once per video and the
            # detail is a `python tools/validate_zones.py` away.
            first = str(e).splitlines()[0]
            print(f"[zones] {first}\n"
                  f"[zones] rules will emit nothing. Run "
                  f"tools/validate_zones.py for the full list.", file=sys.stderr)
        return None
