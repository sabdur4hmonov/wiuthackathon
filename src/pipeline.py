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
from .budget import Budget, measure_part_b_floor
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

    _calibrate_part_b_reserve(video_path, budget, verbose=verbose)

    try:
        tracks = _get_tracks(video_path, budget, use_cache=use_cache, verbose=verbose)
        zones = _get_zones(tracks, verbose=verbose)
        zones = _align_zones(video_path, tracks, zones, budget, verbose=verbose)

        with budget.stage("rules"):
            raw = run_rules(tracks, zones)
        with budget.stage("postprocess"):
            events = finalise(raw, budget.duration)
    except Exception:
        print("[pipeline] Part A failed, returning []:\n" + traceback.format_exc(),
              file=sys.stderr)
        events = []

    _release_part_a()
    if verbose:
        print(budget.format_report(), file=sys.stderr)
    return events


def _release_part_a() -> None:
    """Leave nothing of ours competing with the harness's Part B decode.

    Every capture and container is closed where it is opened (perception's
    finally, measure_part_b_floor, align) and the ffmpeg pipe is stopped with
    its reader; what survives a detect_events call is garbage and the CUDA
    cache. The detector itself stays loaded for the next video.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - cleanup must never cost the video
        pass


def _calibrate_part_b_reserve(video_path: str, budget: Budget,
                              verbose: bool) -> None:
    """Replace the fixed part_b_reserve guess with a measurement from THIS file.

    Must run before anything else touches part_a_hard (which derives from
    part_b_reserve): a fixed 1.2x multiplier was tuned against synthetic 1080p8
    clips, and the real camera footage turned out to be XAVC H.264 High 4:2:2,
    10-bit 4K -- far more expensive per frame to decode. Guessing wrong in
    either direction is bad: too low starves Part A's hard stop of the margin
    Part B actually needs (risking an overrun that zeroes BOTH parts), too high
    starves Part A of time it could safely have used.

    Wrapped in try/except and bounded in cost by measure_part_b_floor itself
    (see BudgetConfig.part_b_probe_frames/part_b_probe_max_sec): a probe
    failure must never be fatal, and this must never be allowed to eat a
    meaningful slice of the budget it exists to protect.
    """
    cfg = CFG.budget
    # Bounded the same way safety_margin is: a flat cap protects the probe
    # from running long on a huge clip, a fractional cap protects a tiny
    # clip's already-thin budget from the probe's own fixed overhead.
    max_probe_sec = min(cfg.part_b_probe_max_sec,
                        cfg.part_b_probe_max_frac * budget.total_budget)
    try:
        rates: list[float] = []
        with budget.stage("part_b_probe"):
            estimate, probe_wall = measure_part_b_floor(
                video_path, budget.n_frames,
                n_probe=cfg.part_b_probe_frames,
                max_probe_sec=max_probe_sec,
                rates_out=rates,
            )
        if estimate is not None:
            budget.set_measured_part_b(estimate)
            budget.note("part_b_probe", 0.0,
                       f"measured {estimate:.1f}s for Part B "
                       f"({probe_wall * 1000:.0f}ms, median of "
                       f"{'/'.join(f'{r * 1000:.0f}' for r in rates)} ms/frame -> "
                       f"{estimate / max(budget.duration, 1e-9):.3f}x realtime)")
            if verbose:
                print(f"[budget] Part B reserve calibrated from a real probe: "
                      f"{estimate:.1f}s (was {cfg.part_b_reserve}x fixed "
                      f"guess = {cfg.part_b_reserve * budget.duration:.1f}s)",
                      file=sys.stderr)
        elif verbose:
            print("[budget] Part B probe read 0 frames; keeping the fixed "
                  f"{cfg.part_b_reserve}x reserve", file=sys.stderr)
    except Exception as e:
        # Never let this calibration step cost the video its prediction.
        if verbose:
            print(f"[budget] Part B probe failed, keeping the fixed "
                  f"{cfg.part_b_reserve}x reserve: {e!r}", file=sys.stderr)


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


def _align_zones(video_path: str, tracks: TrackTable, zones: Zones | None,
                 budget: Budget, verbose: bool) -> Zones | None:
    """Warp zones.json onto this video's camera framing (see src/align.py).

    The pose normally rides on the tracks, estimated from frames Stage 1
    decoded anyway. Tracks from an older cache entry may carry none, or a pose
    skipped for budget before CP4; only then are a few frames decoded here,
    regardless of budget. Any failure leaves the zones unwarped.
    """
    from . import align

    if zones is None:
        return None
    try:
        pose = align.Pose.from_dict(tracks.pose)
        # No pose stored, or Stage 1 could not afford one: estimate it here if
        # there is room now.
        # Never skipped for budget (see perception._keep_for_alignment).
        if pose is None or (pose.is_identity and pose.reason.startswith("skipped")):
            with budget.stage("align"):
                pose = align.estimate_from_video(video_path, budget.duration)
        if verbose:
            align.log(pose)
        return align.warp_zones(zones, pose, tracks.width or zones.image_width)
    except Exception as e:  # noqa: BLE001 - a failed alignment must not cost the video
        if verbose:
            print(f"[align] failed, zones left unwarped: {e!r}", file=sys.stderr)
        return zones


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
