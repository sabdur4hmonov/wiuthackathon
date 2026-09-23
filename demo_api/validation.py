"""Strict, path-safe validation at the untrusted model-adapter boundary.

The numeric/event rules mirror the organizer evaluator and the frontend
parser. Only documented prediction fields are returned to HTTP clients.
"""

from __future__ import annotations

import math
from pathlib import Path

OFFICIAL_CLASSES = frozenset(
    (
        "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
        "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
        "solid_line_crossing", "stop_line", "congestion", "road_obstacle",
        "fire_smoke",
    )
)
UNSAFE_KEYS = frozenset(("__proto__", "constructor", "prototype"))


class PredictionError(ValueError):
    """The adapter returned data that cannot be exposed as predictions."""


def safe_filename(filename: str) -> bool:
    return (
        isinstance(filename, str)
        and 1 <= len(filename) <= 200
        and filename not in UNSAFE_KEYS
        and filename.lower().endswith(".mp4")
        and Path(filename).name == filename
        and not any(char in filename for char in "/\\:")
        and not any(ord(char) < 32 or ord(char) == 127 for char in filename)
    )


def _number(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def sanitized_prediction(raw: object, uploaded_filename: str) -> dict:
    """Reject non-contract fields and return only safe official fields.

    Each demo request handles one uploaded file, so the result must contain
    exactly that file. Extra keys (including harness logs) are rejected.
    """
    if not safe_filename(uploaded_filename):
        raise PredictionError("Invalid uploaded filename.")
    if not isinstance(raw, dict) or set(raw) - {"team", "videos"}:
        raise PredictionError("Prediction has unexpected top-level fields.")
    videos = raw.get("videos")
    if not isinstance(videos, dict) or set(videos) != {uploaded_filename}:
        raise PredictionError("Prediction must contain only the uploaded video.")

    clean: dict = {}
    if "team" in raw:
        team = raw["team"]
        if (
            not isinstance(team, str)
            or len(team) > 100
            or any(char in team for char in "/\\:")
            or any(ord(char) < 32 or ord(char) == 127 for char in team)
        ):
            raise PredictionError("Invalid team name.")
        clean["team"] = team

    entry = videos[uploaded_filename]
    if not isinstance(entry, dict) or set(entry) - {"events", "risk"}:
        raise PredictionError("Video prediction has unexpected fields.")
    raw_events = entry.get("events")
    if not isinstance(raw_events, list):
        raise PredictionError("Events must be an array.")

    events: list[list] = []
    by_class: dict[str, list[tuple[float, float]]] = {}
    for item in raw_events:
        if not isinstance(item, list) or len(item) != 3:
            raise PredictionError("Invalid event shape.")
        start, end, label = item
        if not (_number(start) and _number(end) and 0 <= start < end):
            raise PredictionError("Invalid event interval.")
        if not isinstance(label, str) or label not in OFFICIAL_CLASSES:
            raise PredictionError("Invalid event label.")
        start, end = float(start), float(end)
        events.append([start, end, label])
        by_class.setdefault(label, []).append((start, end))
    for intervals in by_class.values():
        intervals.sort()
        if any(b[0] < a[1] for a, b in zip(intervals, intervals[1:])):
            raise PredictionError("Same-class events overlap.")

    raw_risk = entry.get("risk", [])
    if not isinstance(raw_risk, list):
        raise PredictionError("Risk must be an array when present.")
    risk: list[list[float]] = []
    previous_time = -1.0
    for item in raw_risk:
        if not isinstance(item, list) or len(item) != 2:
            raise PredictionError("Invalid risk point shape.")
        time_sec, score = item
        if not (_number(time_sec) and _number(score)):
            raise PredictionError("Invalid risk value.")
        if time_sec < 0 or time_sec < previous_time or not 0 <= score <= 1:
            raise PredictionError("Risk timestamp or score is out of range.")
        previous_time = float(time_sec)
        risk.append([previous_time, float(score)])

    clean["videos"] = {uploaded_filename: {"events": events, "risk": risk}}
    return clean
