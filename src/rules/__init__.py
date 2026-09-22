"""Stage 2 -- tracks + zones in, raw event segments out.

Pure and fast: no video I/O, no model, no disk. Runs off the cached track table
in well under a second, which is what makes fifty rule iterations a day
possible. Keep it that way -- anything that needs a pixel belongs in Stage 1.

A rule is a function (tracks, zones) -> list[FrameSegment], registered with
@rule("<class_id>"). It works in FRAME INDICES, which are exact; the registry
converts to seconds once, here, and hands the result to Stage 3. No rule
invents its own merge, blip or overlap logic -- postprocess.finalise owns all
of that, so every class is sanitised the same way.

Rules must emit NOTHING when zones is None. A guessed lane is worse than no
lane: a class we emit that never occurs scores 0 for that class AND enlarges
Part A's denominator.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

from ..tracks import TrackTable
from ..zones import Zones


# ---------------------------------------------------------------------------
# what a rule returns
# ---------------------------------------------------------------------------
@dataclass
class FrameSegment:
    """A candidate event as a closed frame interval [start_frame, end_frame].

    Frames rather than seconds because that is what rules actually reason in,
    and because integer indices do not accumulate float error when segments are
    merged and split. The conversion to seconds happens once, in run_rules.
    """

    start_frame: int
    end_frame: int
    score: float = 1.0
    track_ids: tuple[int, ...] = ()
    debug: dict = field(default_factory=dict)


@dataclass
class RawSegment:
    """One candidate event in seconds, before Stage 3. score arbitrates merges."""

    start: float
    end: float
    label: str
    score: float = 1.0
    track_ids: tuple[int, ...] = ()
    debug: dict = field(default_factory=dict)

    def as_event(self) -> list:
        return [float(self.start), float(self.end), str(self.label)]


def frames_to_seconds(seg: FrameSegment, label: str, fps: float,
                      frame_stride: int = 1) -> RawSegment:
    """Convert a closed frame interval to a half-open second interval.

    The segment covers frames start..end INCLUSIVE, so it ends one sample after
    the last frame that carried evidence. `frame_stride` is that sample step:
    perception only looked at every Nth frame, so the true event boundary lies
    somewhere in the stride-sized window after the last flagged frame. Using the
    stride rather than 1 keeps the segment from being systematically short,
    which matters because tIoU 0.7 punishes clipped boundaries.
    """
    fps = fps or 25.0
    step = max(1, int(frame_stride))
    start = max(0.0, seg.start_frame / fps)
    end = (seg.end_frame + step) / fps
    return RawSegment(start, end, label, seg.score, seg.track_ids, dict(seg.debug))


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
#: label -> (tracks, zones) -> list[FrameSegment]. Registering a rule is the
#: ONLY thing needed to start emitting a class: solution.CLASSES is derived
#: from this dict, and Stage 3 needs no change.
RULES: dict[str, callable] = {}


def rule(label: str):
    """Decorator registering a rule function for one class id."""

    def deco(fn):
        RULES[label] = fn
        return fn

    return deco


def run_rules(tracks: TrackTable, zones: Zones | None,
              enabled: set[str] | None = None) -> list[RawSegment]:
    """Run every registered rule and return their segments in seconds.

    A rule that raises is skipped, not fatal: one broken rule must not cost the
    whole video, which is what an exception escaping detect_events would do.
    A rule may also return RawSegments directly, which pass through unchanged.
    """
    out: list[RawSegment] = []
    for label, fn in RULES.items():
        if enabled is not None and label not in enabled:
            continue
        try:
            produced = fn(tracks, zones) or []
        except Exception as e:  # noqa: BLE001 - deliberate: isolate rule failures
            print(f"[rules] {label} raised, skipped: {e!r}", file=sys.stderr)
            continue
        for seg in produced:
            if isinstance(seg, RawSegment):
                out.append(seg)
            elif isinstance(seg, FrameSegment):
                out.append(frames_to_seconds(seg, label, tracks.fps,
                                             tracks.frame_stride))
            elif isinstance(seg, (tuple, list)) and len(seg) == 2:
                # Bare (start_frame, end_frame), as the rule spec allows.
                out.append(frames_to_seconds(
                    FrameSegment(int(seg[0]), int(seg[1])), label,
                    tracks.fps, tracks.frame_stride))
            else:
                print(f"[rules] {label} returned an unusable segment {seg!r}, "
                      f"skipped", file=sys.stderr)
    return out


def emitted_classes() -> set[str]:
    """Classes this build can actually produce. Used to trim solution.CLASSES."""
    return set(RULES)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------
# Importing these modules is what populates RULES. Keep the import at the
# bottom: the rule modules import FrameSegment and `rule` from this one.
from . import congestion, jaywalking, stopped_vehicle, wrong_way  # noqa: E402,F401

__all__ = [
    "FrameSegment", "RawSegment", "RULES", "rule", "run_rules",
    "emitted_classes", "frames_to_seconds",
]
