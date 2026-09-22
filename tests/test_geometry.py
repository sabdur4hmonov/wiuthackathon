"""Unit tests for the geometry helpers.

These are the foundation every Stage 2 rule stands on, and a sign error in
signed_distance_to_line or crossing_direction would not crash -- it would
silently invert a rule. So the sign conventions are pinned explicitly here,
in IMAGE coordinates (y down), not textbook coordinates.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src import geometry as g


# ---------------------------------------------------------------------------
# point in polygon
# ---------------------------------------------------------------------------
SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


@pytest.mark.parametrize("pt,expected", [
    ((5.0, 5.0), True),        # centre
    ((0.1, 0.1), True),        # just inside a corner
    ((-0.1, 5.0), False),      # just outside
    ((15.0, 5.0), False),
    ((5.0, -1.0), False),
])
def test_point_in_polygon_basic(pt, expected):
    assert g.point_in_polygon(pt, SQUARE) is expected


@pytest.mark.parametrize("pt", [(0.0, 5.0), (10.0, 5.0), (5.0, 0.0), (5.0, 10.0),
                                (0.0, 0.0), (10.0, 10.0)])
def test_points_on_edge_count_as_inside(pt):
    """A ground point sitting exactly on a lane boundary must not flicker."""
    assert g.point_in_polygon(pt, SQUARE) is True


def test_point_in_polygon_concave():
    # An L shape; the notch must read as outside.
    L = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]
    assert g.point_in_polygon((2, 2), L) is True
    assert g.point_in_polygon((2, 8), L) is True
    assert g.point_in_polygon((8, 2), L) is True
    assert g.point_in_polygon((8, 8), L) is False       # the notch


def test_point_in_polygon_winding_agnostic():
    reversed_square = list(reversed(SQUARE))
    assert g.point_in_polygon((5, 5), reversed_square) is True
    assert g.point_in_polygon((-5, 5), reversed_square) is False


def test_degenerate_polygon_is_never_inside():
    assert g.point_in_polygon((0, 0), [(0, 0), (1, 1)]) is False
    assert g.point_in_polygon((0, 0), []) is False


def test_points_in_polygon_matches_scalar():
    pts = np.array([[5, 5], [-1, -1], [0, 5], [10, 10], [11, 3]], dtype=float)
    vec = g.points_in_polygon(pts, SQUARE)
    scal = [g.point_in_polygon(tuple(p), SQUARE) for p in pts]
    assert list(vec) == scal


# ---------------------------------------------------------------------------
# signed distance / side
# ---------------------------------------------------------------------------
def test_signed_distance_sign_convention_image_coords():
    """Pin the convention: cross = (b-a) x (p-a), y DOWN.

    For a left-to-right segment, a point BELOW it on screen (larger y) is
    positive. Everything downstream -- stop-line approach sides especially --
    depends on this exact sign, including the annotator's sideOfLine().
    """
    seg = ((0.0, 0.0), (10.0, 0.0))
    assert g.signed_distance_to_line((5.0, 4.0), seg) == pytest.approx(4.0)
    assert g.signed_distance_to_line((5.0, -4.0), seg) == pytest.approx(-4.0)
    assert g.side_of_line((5.0, 4.0), seg) == 1
    assert g.side_of_line((5.0, -4.0), seg) == -1
    assert g.side_of_line((5.0, 0.0), seg) == 0


def test_signed_distance_flips_when_segment_reversed():
    fwd = ((0.0, 0.0), (10.0, 0.0))
    rev = ((10.0, 0.0), (0.0, 0.0))
    p = (5.0, 4.0)
    assert g.signed_distance_to_line(p, fwd) == pytest.approx(
        -g.signed_distance_to_line(p, rev))


def test_signed_distance_is_perpendicular_not_endpoint():
    seg = ((0.0, 0.0), (10.0, 0.0))
    # Well beyond the segment, but the line is infinite.
    assert abs(g.signed_distance_to_line((100.0, 3.0), seg)) == pytest.approx(3.0)


def test_signed_distance_diagonal():
    seg = ((0.0, 0.0), (10.0, 10.0))
    assert abs(g.signed_distance_to_line((10.0, 0.0), seg)) == pytest.approx(
        10.0 / math.sqrt(2))


def test_degenerate_segment_falls_back_to_point_distance():
    assert g.signed_distance_to_line((3.0, 4.0), ((0.0, 0.0), (0.0, 0.0))) == \
        pytest.approx(5.0)


# ---------------------------------------------------------------------------
# intersection and crossing
# ---------------------------------------------------------------------------
def test_segments_intersect():
    assert g.segments_intersect(((0, 0), (10, 10)), ((0, 10), (10, 0)))
    assert not g.segments_intersect(((0, 0), (1, 1)), ((5, 5), (6, 6)))
    # touching at an endpoint counts
    assert g.segments_intersect(((0, 0), (5, 5)), ((5, 5), (10, 0)))
    # parallel
    assert not g.segments_intersect(((0, 0), (10, 0)), ((0, 5), (10, 5)))


def test_crossing_direction_both_ways():
    line = ((0.0, 0.0), (10.0, 0.0))
    # from above the line (negative side) to below (positive side)
    assert g.crossing_direction((5.0, -2.0), (5.0, 2.0), line) == 1
    # and back
    assert g.crossing_direction((5.0, 2.0), (5.0, -2.0), line) == -1
    # no crossing at all
    assert g.crossing_direction((5.0, -2.0), (5.0, -1.0), line) == 0
    # crosses the infinite line but misses the finite segment
    assert g.crossing_direction((50.0, -2.0), (50.0, 2.0), line) == 0


def test_crossing_direction_requires_a_side_change():
    """Sliding along the line is not a crossing."""
    line = ((0.0, 0.0), (10.0, 0.0))
    assert g.crossing_direction((2.0, 0.0), (8.0, 0.0), line) == 0


def test_segment_intersection_point():
    p = g.segment_intersection(((0, 0), (10, 10)), ((0, 10), (10, 0)))
    assert p is not None
    assert p[0] == pytest.approx(5.0)
    assert p[1] == pytest.approx(5.0)
    assert g.segment_intersection(((0, 0), (1, 0)), ((0, 5), (1, 5))) is None


# ---------------------------------------------------------------------------
# headings
# ---------------------------------------------------------------------------
def test_angle_between():
    assert g.angle_between((1, 0), (1, 0)) == pytest.approx(0.0)
    assert g.angle_between((1, 0), (-1, 0)) == pytest.approx(math.pi)
    assert g.angle_between((1, 0), (0, 1)) == pytest.approx(math.pi / 2)


def test_degenerate_vector_reads_as_opposed_not_aligned():
    """A stationary track must never look like it is going WITH the lane.

    Returning 0 here would make every parked car satisfy a 'with traffic' test.
    """
    assert g.angle_between((0, 0), (1, 0)) == pytest.approx(math.pi)
    assert g.heading_vs_lane((0.0, 0.0), (1.0, 0.0)) == pytest.approx(180.0)


def test_heading_vs_lane_degrees():
    assert g.heading_vs_lane((1, 0), (1, 0)) == pytest.approx(0.0)
    assert g.heading_vs_lane((-1, 0), (1, 0)) == pytest.approx(180.0)
    assert g.heading_vs_lane((0, 1), (1, 0)) == pytest.approx(90.0)


def test_unit():
    assert g.unit((3.0, 4.0)) == pytest.approx((0.6, 0.8))
    assert g.unit((0.0, 0.0)) == (0.0, 0.0)


def test_heading_is_atan2_in_image_space():
    assert g.heading(0.0, 1.0) == pytest.approx(math.pi / 2)    # downward on screen
    assert g.heading(1.0, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------------
def test_resample_uniform_grid():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    p = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
    grid, out = g.resample_track(t, p, 0.5)
    assert grid[0] == pytest.approx(0.0)
    assert grid[-1] == pytest.approx(3.0)
    assert np.allclose(np.diff(grid), 0.5)
    assert out[1, 0] == pytest.approx(5.0)          # midpoint interpolation


def test_resample_handles_uneven_input():
    """Perception drops frames; the grid must still be uniform."""
    t = np.array([0.0, 0.1, 0.2, 2.0])
    p = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [20.0, 0.0]])
    grid, out = g.resample_track(t, p, 0.5)
    assert np.allclose(np.diff(grid), 0.5)
    assert out.shape[0] == grid.shape[0]


def test_resample_sorts_unordered_input():
    t = np.array([2.0, 0.0, 1.0])
    p = np.array([[20.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
    grid, out = g.resample_track(t, p, 1.0)
    assert np.allclose(out[:, 0], [0.0, 10.0, 20.0])


def test_resample_edge_cases():
    grid, out = g.resample_track(np.array([]), np.empty((0, 2)), 0.5)
    assert grid.size == 0 and out.shape[0] == 0
    grid, out = g.resample_track(np.array([1.0]), np.array([[3.0, 4.0]]), 0.5)
    assert grid.size == 1 and out.shape == (1, 2)


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------
def test_ground_point_is_bottom_centre_not_centroid():
    """The ground-contact point is where the object meets the road plane."""
    assert g.bbox_ground_point(10.0, 20.0, 30.0, 60.0) == (20.0, 60.0)


def test_polygon_area():
    assert g.polygon_area(SQUARE) == pytest.approx(100.0)
    assert g.polygon_area(list(reversed(SQUARE))) == pytest.approx(100.0)
    assert g.polygon_area([(0, 0), (1, 1)]) == 0.0


def test_polyline_length():
    assert g.polyline_length(np.array([[0, 0], [3, 4], [3, 8]])) == pytest.approx(9.0)
    assert g.polyline_length(np.array([[0, 0]])) == 0.0


# ---------------------------------------------------------------------------
# vectorised point-in-polygon must agree with the scalar one everywhere
# ---------------------------------------------------------------------------
def test_vectorised_agrees_with_scalar_on_a_random_sweep():
    """Stage 2 uses the vectorised path; a divergence would silently change
    which lane every track row is assigned to."""
    rng = np.random.default_rng(4)
    poly = [(100, 100), (900, 140), (860, 700), (300, 820), (120, 500)]
    pts = rng.uniform(0, 1000, size=(4000, 2))
    vec = g.points_in_polygon(pts, poly)
    scal = np.array([g.point_in_polygon((float(a), float(b)), poly) for a, b in pts])
    assert np.array_equal(vec, scal)


def test_vectorised_agrees_on_vertices_and_edges():
    poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10],
                    [5, 0], [10, 5], [5, 10], [0, 5],
                    [5, 5], [-1, 5], [11, 5]], dtype=float)
    vec = g.points_in_polygon(pts, poly)
    scal = np.array([g.point_in_polygon((float(a), float(b)), poly) for a, b in pts])
    assert np.array_equal(vec, scal)


def test_vectorised_agrees_on_a_concave_polygon():
    L = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]
    rng = np.random.default_rng(9)
    pts = rng.uniform(-2, 12, size=(2000, 2))
    vec = g.points_in_polygon(pts, L)
    scal = np.array([g.point_in_polygon((float(a), float(b)), L) for a, b in pts])
    assert np.array_equal(vec, scal)


def test_vectorised_edge_cases():
    assert g.points_in_polygon(np.empty((0, 2)), SQUARE).shape == (0,)
    assert not g.points_in_polygon(np.array([[0.0, 0.0]]), [(0, 0), (1, 1)]).any()


def test_vectorised_is_actually_fast():
    """The whole reason it exists. A Python loop here costs seconds per video."""
    import time

    rng = np.random.default_rng(1)
    poly = [(100, 100), (900, 140), (860, 700), (300, 820), (120, 500)]
    pts = rng.uniform(0, 1000, size=(40000, 2))
    t0 = time.perf_counter()
    g.points_in_polygon(pts, poly)
    assert time.perf_counter() - t0 < 0.5
