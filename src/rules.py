"""Stage 2 -- tracks + zones in, raw event segments out.

Pure and fast: no video I/O, no model, no disk. Runs off the cached track table
in well under a second, which is what makes fifty rule iterations a day
possible. Keep it that way -- anything that needs a pixel belongs in Stage 1.

CP0 ships this stage EMPTY on purpose. Every rule here depends on scene
geometry, and config/zones.json is still un-authored: there is no camera.md in
the kit, so no lane direction, stop line or crossing is known yet. Guessing them
would produce events that look plausible and score zero while inflating the
Part A denominator (a class we emit that never occurs is a guaranteed 0 that
also grows |C|).

Each rule returns RawSegments; Stage 3 merges and sanitises them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .tracks import TrackTable
from .zones import Zones


@dataclass
class RawSegment:
    """One candidate event before merging. score drives fragment arbitration."""

    start: float
    end: float
    label: str
    score: float = 1.0
    track_ids: tuple[int, ...] = ()
    debug: dict = field(default_factory=dict)

    def as_event(self) -> list:
        return [float(self.start), float(self.end), str(self.label)]


# ---------------------------------------------------------------------------
# rule registry
# ---------------------------------------------------------------------------
# Signature: (tracks, zones) -> list[RawSegment]. Registering a rule is the
# ONLY thing needed to start emitting a class; Stage 3 and solution.py need no
# change. Nothing is registered at CP0.
RULES: dict[str, callable] = {}


def rule(label: str):
    """Decorator registering a rule function for one class id."""

    def deco(fn):
        RULES[label] = fn
        return fn

    return deco


def run_rules(tracks: TrackTable, zones: Zones | None,
              enabled: set[str] | None = None) -> list[RawSegment]:
    """Run every registered rule and concatenate the raw segments.

    A rule that raises is skipped, not fatal: one broken rule must not cost the
    whole video, which is what an exception escaping detect_events would do.
    """
    out: list[RawSegment] = []
    for label, fn in RULES.items():
        if enabled is not None and label not in enabled:
            continue
        try:
            out.extend(fn(tracks, zones) or [])
        except Exception as e:  # noqa: BLE001 - deliberate: isolate rule failures
            import sys

            print(f"[rules] {label} raised, skipped: {e!r}", file=sys.stderr)
    return out


def emitted_classes() -> set[str]:
    """Classes this build can actually produce. Used to trim solution.CLASSES."""
    return set(RULES)
