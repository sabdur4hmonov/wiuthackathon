"""solution.py -- the only file the organizers' harness imports.

Keep the three names exactly as they are: CLASSES, detect_events, RiskEstimator.
Everything real lives in src/; this file is the seam.

Note on CLASSES. Score A's class set is built by evaluate.py from the labels
present in the ground truth UNION the labels present in our predictions
(evaluate_part_a), so shortening this list does not shrink the denominator by
itself -- only not EMITTING a class does. What shortening it does do is act as
a safety net: run_submission.py filters our events against this list before
writing them, so a class listed here that a buggy rule emits on the wrong video
becomes a guaranteed zero that also enlarges |C|. The list is therefore kept in
lockstep with what the rules can actually produce.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# The harness imports this file by path, so the repo root is only on sys.path
# when it happens to be the working directory. Make that explicit.
_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import enforce_offline, seed_everything
from src.pipeline import detect_events as _detect_events
from src.risk import RiskEstimator as _RiskEstimator
from src.rules import emitted_classes

enforce_offline()
seed_everything()

# The 14 official ids, in the official order. Never add to this.
OFFICIAL_CLASSES: list[str] = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]

# Only the classes some registered rule can actually emit. At CP0 no rule is
# registered, so this is empty and we emit nothing -- which is exactly right:
# an emitted class that never occurs scores 0 AND grows the denominator.
_EMITTED = emitted_classes()
CLASSES: list[str] = [c for c in OFFICIAL_CLASSES if c in _EMITTED]

RISK_HORIZON_SEC = 5.0


def detect_events(video_path: str) -> list[list]:
    """Part A. [[start_sec, end_sec, label], ...], same-class segments disjoint."""
    return _detect_events(video_path)


class RiskEstimator(_RiskEstimator):
    """Part B. Causal: only the frames step() has already been handed."""

    pass
