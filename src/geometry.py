"""Pure 2-D geometry on image coordinates.

Conventions used everywhere in this repo:

* Image coordinates: x right, y DOWN (OpenCV). Consequences that bite:
  - the cross product sign is mirrored relative to textbook maths, so what
    reads as "left of the line" is "right of the line" on screen;
  - an angle measured with atan2(dy, dx) grows CLOCKWISE on screen.
  We do not flip anything -- we stay in image space consistently and only care
  about sign consistency between a track heading and a lane direction, both of
  which are measured the same way.
* A polygon is a list of (x, y) vertices, implicitly closed, any winding.
* A segment is ((x1, y1), (x2, y2)).

No module here may import cv2, torch or anything from the pipeline: these
functions are used by both the batch rules and the causal risk estimator.
"""
from __future__ import annotations

import math

import numpy as np

Point = tuple[float, float]
Segment = tuple[Point, Point]
Polygon = list[Point] | np.ndarray

EPS = 1e-9


# ---------------------------------------------------------------------------
# point in polygon
# ---------------------------------------------------------------------------
def point_in_polygon(pt: Point, poly: Polygon) -> bool:
    """Ray-casting test. Points exactly on an edge count as INSIDE.

    The on-edge case is made explicit because a vehicle's ground-contact point
    sitting on a lane boundary is common, and letting it flicker between lanes
    would generate phantom lane-change events.
    """
    p = np.asarray(poly, dtype=float)
    if p.shape[0] < 3:
        return False
    x, y = float(pt[0]), float(pt[1])

    # on-edge check first
    for i in range(p.shape[0]):
        a, b = p[i], p[(i + 1) % p.shape[0]]
        if _point_on_segment((x, y), (tuple(a), tuple(b))):
            return True

    inside = False
    j = p.shape[0] - 1
    for i in range(p.shape[0]):
        xi, yi = p[i]
        xj, yj = p[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi + EPS) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def points_in_polygon(pts: np.ndarray, poly: Polygon) -> np.ndarray:
    """Vectorised point_in_polygon for an (N, 2) array. Returns (N,) bool.

    Edge cases are resolved by the scalar version so the two always agree.
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    return np.array([point_in_polygon((float(x), float(y)), poly) for x, y in pts],
                    dtype=bool)


def _point_on_segment(pt: Point, seg: Segment, tol: float = 1e-6) -> bool:
    (x1, y1), (x2, y2) = seg
    x, y = pt
    cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    if abs(cross) > tol * max(1.0, abs(x2 - x1) + abs(y2 - y1)):
        return False
    dot = (x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)
    if dot < -tol:
        return False
    sq_len = (x2 - x1) ** 2 + (y2 - y1) ** 2
    return dot <= sq_len + tol


# ---------------------------------------------------------------------------
# signed distance to a line
# ---------------------------------------------------------------------------
def signed_distance_to_line(pt: Point, seg: Segment) -> float:
    """Perpendicular signed distance from pt to the INFINITE line through seg.

    Sign is by the 2-D cross product (b-a) x (p-a): consistent, and the side
    that is "positive" flips if you reverse the segment's endpoints. Callers
    must therefore fix an orientation once (we do it in zones.json, where every
    stop line is authored in a defined direction) and never reverse it.
    """
    (x1, y1), (x2, y2) = seg
    dx, dy = x2 - x1, y2 - y1
    norm = math.hypot(dx, dy)
    if norm < EPS:
        return math.hypot(pt[0] - x1, pt[1] - y1)
    return (dx * (pt[1] - y1) - dy * (pt[0] - x1)) / norm


def side_of_line(pt: Point, seg: Segment) -> int:
    """-1, 0 or +1 -- which side of the oriented line pt falls on."""
    d = signed_distance_to_line(pt, seg)
    if abs(d) < 1e-6:
        return 0
    return 1 if d > 0 else -1


# ---------------------------------------------------------------------------
# segment crossing
# ---------------------------------------------------------------------------
def segments_intersect(a: Segment, b: Segment) -> bool:
    """Proper or improper intersection of two finite segments."""
    (p1, p2), (p3, p4) = a, b
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    for d, p, seg in ((d1, p1, b), (d2, p2, b), (d3, p3, a), (d4, p4, a)):
        if abs(d) < EPS and _point_on_segment(p, seg):
            return True
    return False


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def crossing_direction(prev: Point, curr: Point, line: Segment) -> int:
    """Did the step prev -> curr cross `line`, and in which direction?

    Returns +1 or -1 for a crossing (the sign is the side it ended up on), 0
    for no crossing. This is what separates a red-light runner from a vehicle
    that merely reversed over the stop line, so the direction matters as much
    as the fact of the crossing.
    """
    if not segments_intersect((prev, curr), line):
        return 0
    s_prev = side_of_line(prev, line)
    s_curr = side_of_line(curr, line)
    if s_prev == s_curr:
        return 0
    if s_curr != 0:
        return s_curr
    return -s_prev


def segment_intersection(a: Segment, b: Segment) -> Point | None:
    """Intersection point of two segments, or None if they do not meet."""
    (x1, y1), (x2, y2) = a
    (x3, y3), (x4, y4) = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < EPS:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / den
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None


# ---------------------------------------------------------------------------
# headings
# ---------------------------------------------------------------------------
def heading(dx: float, dy: float) -> float:
    """Heading of a motion vector in radians, atan2(dy, dx) in image space."""
    return math.atan2(dy, dx)


def unit(v: tuple[float, float]) -> tuple[float, float]:
    n = math.hypot(v[0], v[1])
    if n < EPS:
        return (0.0, 0.0)
    return (v[0] / n, v[1] / n)


def angle_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Unsigned angle between two vectors, radians in [0, pi].

    Returns pi (fully opposed) when either vector is degenerate, so that a
    stationary track is never mistaken for one travelling WITH the lane.
    Wrong-way rules must gate on speed anyway, but the default should not be
    the permissive one.
    """
    ua, ub = unit(a), unit(b)
    if ua == (0.0, 0.0) or ub == (0.0, 0.0):
        return math.pi
    dot = max(-1.0, min(1.0, ua[0] * ub[0] + ua[1] * ub[1]))
    return math.acos(dot)


def heading_vs_lane(track_dir: tuple[float, float],
                    lane_dir: tuple[float, float]) -> float:
    """Angle in DEGREES between a track's heading and its lane's direction.

    0 = travelling with the lane, 180 = straight against it. The wrong_way rule
    thresholds on this; keeping it in degrees keeps zones.json readable.
    """
    return math.degrees(angle_between(track_dir, lane_dir))


# ---------------------------------------------------------------------------
# track resampling
# ---------------------------------------------------------------------------
def resample_track(times: np.ndarray, points: np.ndarray,
                   step_sec: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample a trajectory onto a uniform time grid by linear interpolation.

    Perception runs on a frame stride and drops frames when the detector misses,
    so raw tracks are unevenly sampled. Rules that measure "10 seconds
    stationary" or a speed need a uniform grid or they silently weight the
    dense stretches more. Returns (grid_times, interpolated_points).
    """
    times = np.asarray(times, dtype=float).ravel()
    points = np.asarray(points, dtype=float)
    if times.size == 0:
        # reshape(0, -1) is ambiguous for numpy, so bail before reshaping.
        width = points.shape[1] if points.ndim == 2 and points.shape[1] else 2
        return np.empty(0), np.empty((0, width))
    points = points.reshape(len(times), -1)
    if times.size == 1:
        return times.copy(), points.copy()

    order = np.argsort(times, kind="stable")
    times, points = times[order], points[order]
    grid = np.arange(times[0], times[-1] + step_sec * 0.5, step_sec)
    if grid.size == 0:
        grid = times[:1].copy()
    out = np.empty((grid.size, points.shape[1]), dtype=float)
    for c in range(points.shape[1]):
        out[:, c] = np.interp(grid, times, points[:, c])
    return grid, out


def polyline_length(points: np.ndarray) -> float:
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    if p.shape[0] < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))))


def polygon_area(poly: Polygon) -> float:
    """Unsigned shoelace area. Used by the zones validator to reject degenerates."""
    p = np.asarray(poly, dtype=float).reshape(-1, 2)
    if p.shape[0] < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def bbox_ground_point(x1: float, y1: float, x2: float, y2: float) -> Point:
    """Ground-contact point of a detection box: BOTTOM-CENTRE, not the centroid.

    On a CCTV view the centroid floats up the object as it grows nearer the
    camera, so a lane test on the centroid drifts across lane boundaries for a
    vehicle driving perfectly straight. The bottom edge midpoint is where the
    object meets the road plane, which is the only point a ground-plane zone
    test is valid for.
    """
    return ((x1 + x2) / 2.0, y2)
