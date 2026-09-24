"""Loading, validating and querying the hand-authored scene geometry.

config/zones.json describes the real camera's scene, traced against its own
footage (there is no camera.md in the kit). A half-authored file must never
silently produce plausible-looking garbage, so `load_zones` validates by
default and raises on any null or TODO it finds.

Rules code should never touch the raw dict. It asks this module questions --
"is this point on the carriageway", "which lane is it in", "did it cross a stop
line" -- so the on-disk shape can change without touching Stage 2.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import geometry as g
from .config import ZONES_PATH

MANOEUVRES = {"through", "left", "right", "u_turn"}
MARKING_STYLES = {"solid", "dashed", "double_solid"}


class ZonesError(ValueError):
    """Raised when zones.json is missing, malformed or still un-authored."""


# ---------------------------------------------------------------------------
# dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Lane:
    id: str
    polygon: np.ndarray
    direction: tuple[float, float]           # unit vector, image space
    permitted_manoeuvres: frozenset[str]
    governed_by_stop_line: str | None
    # Optional name of the direction of travel this lane belongs to (e.g.
    # "southbound"). Only congestion reads it; when no lane sets one, rules
    # fall back to clustering lanes by their arrows.
    direction_group: str | None = None

    def contains(self, pt: g.Point) -> bool:
        return g.point_in_polygon(pt, self.polygon)

    def heading_offset_deg(self, track_dir: tuple[float, float]) -> float:
        """0 = with the lane, 180 = against it."""
        return g.heading_vs_lane(track_dir, self.direction)


@dataclass(frozen=True)
class StopLine:
    id: str
    segment: g.Segment
    approach_side: int                        # +1 or -1
    governs_lanes: tuple[str, ...]

    def crossed_inbound(self, prev: g.Point, curr: g.Point) -> bool:
        """True when a step crosses the line AWAY from the approach side.

        Direction matters: a car rolling backwards over the line has not run
        the light. crossing_direction returns the side it ended on, so an
        inbound crossing is the one that ends on -approach_side.
        """
        d = g.crossing_direction(prev, curr, self.segment)
        return d != 0 and d == -self.approach_side

    def signed_distance(self, pt: g.Point) -> float:
        """Signed perpendicular distance; positive on the approach side."""
        return g.signed_distance_to_line(pt, self.segment) * self.approach_side


@dataclass(frozen=True)
class Area:
    """A named polygon: carriageway piece, crossing, or signal queue zone."""

    id: str
    polygon: np.ndarray
    lanes: tuple[str, ...] = ()
    # Per edge (vertex i -> i+1): True where the edge runs along the border of
    # the authored frame. Such an edge is where the picture ends, not a kerb.
    # Computed once in authored coordinates, so it survives rescaling and
    # camera-pose warping, which only move vertices.
    frame_edges: tuple[bool, ...] = ()

    def contains(self, pt: g.Point) -> bool:
        return g.point_in_polygon(pt, self.polygon)


@dataclass(frozen=True)
class TrafficLight:
    id: str
    roi: tuple[float, float, float, float]   # x1, y1, x2, y2
    controls_lanes: tuple[str, ...]


@dataclass(frozen=True)
class LaneMarking:
    id: str
    segment: g.Segment
    style: str
    separates: tuple[str, ...]

    @property
    def is_crossable(self) -> bool:
        return self.style == "dashed"


@dataclass
class Zones:
    """The whole authored scene, in the coordinate frame of the video."""

    image_width: int
    image_height: int
    source_video: str | None
    carriageway: tuple[Area, ...]
    lanes: tuple[Lane, ...]
    stop_lines: tuple[StopLine, ...]
    crossings: tuple[Area, ...]
    signal_queue_zones: tuple[Area, ...]
    traffic_lights: tuple[TrafficLight, ...]
    lane_markings: tuple[LaneMarking, ...]
    notes: str = ""
    scale: tuple[float, float] = (1.0, 1.0)
    _lane_by_id: dict[str, Lane] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._lane_by_id = {ln.id: ln for ln in self.lanes}

    # -- queries used by Stage 2 -------------------------------------------
    def on_carriageway(self, pt: g.Point) -> bool:
        return any(a.contains(pt) for a in self.carriageway)

    def lane_of(self, pt: g.Point) -> Lane | None:
        """The lane containing pt, or None. First match wins; lanes must not overlap."""
        for ln in self.lanes:
            if ln.contains(pt):
                return ln
        return None

    def lane(self, lane_id: str) -> Lane | None:
        return self._lane_by_id.get(lane_id)

    def in_crossing(self, pt: g.Point) -> Area | None:
        for a in self.crossings:
            if a.contains(pt):
                return a
        return None

    def in_signal_queue(self, pt: g.Point) -> bool:
        """True where a vehicle may legitimately sit still on a red."""
        return any(a.contains(pt) for a in self.signal_queue_zones)

    def stop_line(self, stop_line_id: str) -> StopLine | None:
        for sl in self.stop_lines:
            if sl.id == stop_line_id:
                return sl
        return None

    def solid_markings(self) -> tuple[LaneMarking, ...]:
        return tuple(m for m in self.lane_markings if not m.is_crossable)

    @property
    def has_traffic_light(self) -> bool:
        return bool(self.traffic_lights)

    def summary(self) -> str:
        return (f"zones: {len(self.carriageway)} carriageway, {len(self.lanes)} lanes, "
                f"{len(self.stop_lines)} stop lines, {len(self.crossings)} crossings, "
                f"{len(self.signal_queue_zones)} queue zones, "
                f"{len(self.traffic_lights)} signals, "
                f"{len(self.lane_markings)} markings "
                f"@ {self.image_width}x{self.image_height}")


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def _is_todo(v: Any) -> bool:
    """A value the author has not supplied yet."""
    if v is None:
        return True
    if isinstance(v, str) and (v.strip() == "" or v.strip().upper().startswith("TODO")):
        return True
    return False


def _strip_comments(node: Any) -> Any:
    """Drop the _draw / _README annotation keys before structural checks."""
    if isinstance(node, dict):
        return {k: _strip_comments(v) for k, v in node.items() if not k.startswith("_")}
    if isinstance(node, list):
        return [_strip_comments(v) for v in node]
    return node


def _check_polygon(raw: Any, where: str, errors: list[str]) -> np.ndarray | None:
    if _is_todo(raw):
        errors.append(f"{where}.polygon is still un-authored (null/TODO)")
        return None
    try:
        p = np.asarray(raw, dtype=float).reshape(-1, 2)
    except Exception:
        errors.append(f"{where}.polygon must be [[x, y], ...], got {raw!r}")
        return None
    if p.shape[0] < 3:
        errors.append(f"{where}.polygon needs at least 3 vertices, got {p.shape[0]}")
        return None
    if g.polygon_area(p) < 1.0:
        errors.append(f"{where}.polygon is degenerate (area {g.polygon_area(p):.3f} px^2)")
        return None
    return p


def _check_segment(raw: Any, where: str, errors: list[str]) -> g.Segment | None:
    if _is_todo(raw):
        errors.append(f"{where}.segment is still un-authored (null/TODO)")
        return None
    try:
        s = np.asarray(raw, dtype=float).reshape(2, 2)
    except Exception:
        errors.append(f"{where}.segment must be [[x1, y1], [x2, y2]], got {raw!r}")
        return None
    if math.hypot(s[1, 0] - s[0, 0], s[1, 1] - s[0, 1]) < 1.0:
        errors.append(f"{where}.segment has zero length")
        return None
    return ((float(s[0, 0]), float(s[0, 1])), (float(s[1, 0]), float(s[1, 1])))


def validate_raw(raw: dict) -> list[str]:
    """Structural + completeness check. Returns a list of human-readable errors.

    An empty list means the file is fully authored and internally consistent.
    A non-empty list must abort the pipeline: rules built on half-drawn geometry
    produce confident nonsense, which is worse than producing nothing.
    """
    errors: list[str] = []
    d = _strip_comments(raw)

    if d.get("schema_version") != 1:
        errors.append(f"schema_version must be 1, got {d.get('schema_version')!r}")

    # -- authored_against --------------------------------------------------
    aa = d.get("authored_against") or {}
    for k in ("image_width", "image_height"):
        if _is_todo(aa.get(k)) or not isinstance(aa.get(k), int) or aa.get(k, 0) <= 0:
            errors.append(f"authored_against.{k} must be a positive int, got {aa.get(k)!r}")

    # -- carriageway -------------------------------------------------------
    cw = d.get("carriageway") or []
    if not cw:
        errors.append("carriageway must have at least one polygon: nothing can be "
                      "judged on-road without it")
    for i, a in enumerate(cw):
        _check_polygon(a.get("polygon"), f"carriageway[{i}]", errors)

    # -- lanes -------------------------------------------------------------
    lanes = d.get("lanes") or []
    if not lanes:
        errors.append("lanes must have at least one entry")
    lane_ids: set[str] = set()
    for i, ln in enumerate(lanes):
        where = f"lanes[{i}]"
        lid = ln.get("id")
        if _is_todo(lid):
            errors.append(f"{where}.id is still the TODO placeholder")
        elif lid in lane_ids:
            errors.append(f"{where}.id {lid!r} is duplicated")
        else:
            lane_ids.add(lid)
        _check_polygon(ln.get("polygon"), where, errors)
        f_, t_ = ln.get("direction_from"), ln.get("direction_to")
        if _is_todo(f_) or _is_todo(t_):
            errors.append(f"{where} direction arrow un-authored "
                          f"(need direction_from and direction_to)")
        else:
            try:
                dx = float(t_[0]) - float(f_[0])
                dy = float(t_[1]) - float(f_[1])
                if math.hypot(dx, dy) < 1.0:
                    errors.append(f"{where} direction arrow has zero length")
            except Exception:
                errors.append(f"{where} direction points must be [x, y]")
        bad = set(ln.get("permitted_manoeuvres") or []) - MANOEUVRES
        if bad:
            errors.append(f"{where}.permitted_manoeuvres has unknown values {sorted(bad)}; "
                          f"allowed: {sorted(MANOEUVRES)}")
        grp = ln.get("direction_group")
        if grp is not None and (not isinstance(grp, str) or _is_todo(grp)):
            errors.append(f"{where}.direction_group must be a name or absent, got {grp!r}")

    # -- stop lines --------------------------------------------------------
    sl_ids: set[str] = set()
    for i, sl in enumerate(d.get("stop_lines") or []):
        where = f"stop_lines[{i}]"
        if _is_todo(sl.get("id")):
            errors.append(f"{where}.id is still the TODO placeholder")
        else:
            sl_ids.add(sl["id"])
        _check_segment(sl.get("segment"), where, errors)
        if sl.get("approach_side") not in (-1, 1):
            errors.append(f"{where}.approach_side must be -1 or +1, got "
                          f"{sl.get('approach_side')!r}")
        for lid in sl.get("governs_lanes") or []:
            if lid not in lane_ids:
                errors.append(f"{where}.governs_lanes references unknown lane {lid!r}")

    for i, ln in enumerate(lanes):
        ref = ln.get("governed_by_stop_line")
        if ref is not None and ref not in sl_ids:
            errors.append(f"lanes[{i}].governed_by_stop_line references unknown "
                          f"stop line {ref!r}")

    # -- crossings / queue zones ------------------------------------------
    for key in ("crossings", "signal_queue_zones"):
        for i, a in enumerate(d.get(key) or []):
            where = f"{key}[{i}]"
            if _is_todo(a.get("id")):
                errors.append(f"{where}.id is still the TODO placeholder")
            _check_polygon(a.get("polygon"), where, errors)
            for lid in a.get("lanes") or []:
                if lid not in lane_ids:
                    errors.append(f"{where}.lanes references unknown lane {lid!r}")

    # -- traffic lights ----------------------------------------------------
    # An EMPTY list is legal and meaningful: no signal visible in frame.
    for i, tl in enumerate(d.get("traffic_lights") or []):
        where = f"traffic_lights[{i}]"
        if _is_todo(tl.get("id")):
            errors.append(f"{where}.id is still the TODO placeholder")
        roi = tl.get("roi")
        if _is_todo(roi):
            errors.append(f"{where}.roi is un-authored; if no signal is visible, "
                          f"delete this entry and leave traffic_lights empty")
        else:
            try:
                x1, y1, x2, y2 = (float(v) for v in roi)
                if not (x2 > x1 and y2 > y1):
                    errors.append(f"{where}.roi must be [x1, y1, x2, y2] with x2>x1, y2>y1")
            except Exception:
                errors.append(f"{where}.roi must be four numbers, got {roi!r}")
        for lid in tl.get("controls_lanes") or []:
            if lid not in lane_ids:
                errors.append(f"{where}.controls_lanes references unknown lane {lid!r}")

    # -- lane markings -----------------------------------------------------
    for i, m in enumerate(d.get("lane_markings") or []):
        where = f"lane_markings[{i}]"
        if _is_todo(m.get("id")):
            errors.append(f"{where}.id is still the TODO placeholder")
        _check_segment(m.get("segment"), where, errors)
        if m.get("style") not in MARKING_STYLES:
            errors.append(f"{where}.style must be one of {sorted(MARKING_STYLES)}, "
                          f"got {m.get('style')!r}")
        for lid in m.get("separates") or []:
            if lid not in lane_ids:
                errors.append(f"{where}.separates references unknown lane {lid!r}")

    return errors


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_zones(path: str | Path | None = None,
               frame_size: tuple[int, int] | None = None,
               validate: bool = True) -> Zones:
    """Read, validate and build the Zones object.

    Args:
        path: zones.json; defaults to config/zones.json.
        frame_size: (width, height) of the video being processed. If it differs
            from authored_against, every coordinate is rescaled. Authoring
            against a 1080p frame and running on a 720p video is a normal
            mistake and silently shifting every polygon would be a nightmare to
            debug, so we handle it explicitly.
        validate: set False only in tools that want to inspect a partial file.

    Raises:
        ZonesError: file missing, malformed, or still holding TODOs.
    """
    p = Path(path) if path is not None else ZONES_PATH
    if not p.exists():
        raise ZonesError(f"zones file not found: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ZonesError(f"{p} is not valid JSON: {e}") from e

    if validate:
        errors = validate_raw(raw)
        if errors:
            listed = "\n  - ".join(errors)
            raise ZonesError(
                f"{p} is not fully authored ({len(errors)} problem(s)):\n  - {listed}\n"
                f"Draw the missing geometry with tools/annotate.html, check it with "
                f"tools/draw_zones.py, then re-run tools/validate_zones.py."
            )

    d = _strip_comments(raw)
    aa = d.get("authored_against") or {}
    aw, ah = int(aa.get("image_width") or 0), int(aa.get("image_height") or 0)
    sx = sy = 1.0
    if frame_size and aw and ah:
        sx, sy = frame_size[0] / aw, frame_size[1] / ah

    def poly(raw_pts: Any) -> np.ndarray:
        return np.asarray(raw_pts, dtype=float).reshape(-1, 2) * np.array([sx, sy])

    def seg(raw_pts: Any) -> g.Segment:
        s = np.asarray(raw_pts, dtype=float).reshape(2, 2) * np.array([sx, sy])
        return ((float(s[0, 0]), float(s[0, 1])), (float(s[1, 0]), float(s[1, 1])))

    lanes = []
    for ln in d.get("lanes") or []:
        f_, t_ = ln["direction_from"], ln["direction_to"]
        # The direction is a unit vector, so scaling is only needed when the
        # aspect ratio changes; applying it keeps the arrow true to the drawing.
        dvec = g.unit(((float(t_[0]) - float(f_[0])) * sx,
                       (float(t_[1]) - float(f_[1])) * sy))
        lanes.append(Lane(
            id=ln["id"],
            polygon=poly(ln["polygon"]),
            direction=dvec,
            permitted_manoeuvres=frozenset(ln.get("permitted_manoeuvres") or []),
            governed_by_stop_line=ln.get("governed_by_stop_line"),
            direction_group=ln.get("direction_group"),
        ))

    z = Zones(
        image_width=int(round(aw * sx)) or aw,
        image_height=int(round(ah * sy)) or ah,
        source_video=aa.get("source_video"),
        carriageway=tuple(Area(a.get("id", f"cw{i}"), poly(a["polygon"]),
                               frame_edges=_frame_edges(a["polygon"], aw, ah))
                          for i, a in enumerate(d.get("carriageway") or [])),
        lanes=tuple(lanes),
        stop_lines=tuple(StopLine(sl["id"], seg(sl["segment"]),
                                  int(sl["approach_side"]),
                                  tuple(sl.get("governs_lanes") or []))
                         for sl in d.get("stop_lines") or []),
        crossings=tuple(Area(a["id"], poly(a["polygon"]))
                        for a in d.get("crossings") or []),
        signal_queue_zones=tuple(Area(a["id"], poly(a["polygon"]),
                                      tuple(a.get("lanes") or []))
                                 for a in d.get("signal_queue_zones") or []),
        traffic_lights=tuple(TrafficLight(
            t["id"],
            tuple(float(v) * s for v, s in zip(t["roi"], (sx, sy, sx, sy))),
            tuple(t.get("controls_lanes") or []))
            for t in d.get("traffic_lights") or []),
        lane_markings=tuple(LaneMarking(m["id"], seg(m["segment"]), m["style"],
                                        tuple(m.get("separates") or []))
                            for m in d.get("lane_markings") or []),
        notes=d.get("notes", "") or "",
        scale=(sx, sy),
    )
    return z


def _frame_edges(raw_pts: Any, width: float, height: float,
                 tol: float = 2.0) -> tuple[bool, ...]:
    """Which edges of an authored polygon lie along the authored frame border."""
    p = np.asarray(raw_pts, dtype=float).reshape(-1, 2)
    if not width or not height or p.shape[0] < 2:
        return ()
    q = np.roll(p, -1, axis=0)
    out = []
    for (x1, y1), (x2, y2) in zip(p, q):
        out.append(bool((abs(x1) <= tol and abs(x2) <= tol)
                        or (abs(x1 - width) <= tol and abs(x2 - width) <= tol)
                        or (abs(y1) <= tol and abs(y2) <= tol)
                        or (abs(y1 - height) <= tol and abs(y2 - height) <= tol)))
    return tuple(out)


def is_authored(path: str | Path | None = None) -> bool:
    """True when zones.json is complete. Used to decide whether rules can run."""
    p = Path(path) if path is not None else ZONES_PATH
    if not p.exists():
        return False
    try:
        return not validate_raw(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return False
