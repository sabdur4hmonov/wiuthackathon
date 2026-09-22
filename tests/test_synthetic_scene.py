"""The synthetic scene fixture must be trustworthy before any rule leans on it.

A bug here would produce rule tests that pass against nonsense, which is worse
than no tests at all: it would create confidence that does not transfer to real
footage.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fixtures.synthetic_scene import (CROSSING_POINT, JAYWALK_POINT,
                                      OFF_ROAD_POINT, QUEUE_POINT,
                                      STOPPED_POINT, build_zones_dict,
                                      point_in_lane, scene_zones,
                                      write_zones_json)
from src.zones import validate_raw

ROOT = Path(__file__).resolve().parent.parent


def test_scene_validates_clean():
    assert validate_raw(build_zones_dict()) == []


def test_scene_passes_the_real_validator_cli(tmp_path):
    """Through tools/validate_zones.py, not just the library function."""
    p = write_zones_json(tmp_path / "zones.json")
    res = subprocess.run(
        [sys.executable, "tools/validate_zones.py", "--zones", str(p)],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "VALID" in res.stdout


def test_scene_shape(tmp_path):
    z = scene_zones(tmp_path)
    assert len(z.carriageway) == 2
    assert len(z.lanes) == 8
    assert len(z.stop_lines) == 4
    assert len(z.crossings) == 2
    assert len(z.signal_queue_zones) == 2
    assert z.has_traffic_light
    assert len(z.lane_markings) == 3


def test_road_narrows_with_distance():
    """Oblique CCTV view, not a top-down grid."""
    from fixtures.synthetic_scene import ns_centre, ns_halfwidth

    assert ns_halfwidth(320.0) < ns_halfwidth(700.0) < ns_halfwidth(1080.0)
    # ...and the centreline drifts, so the road is not axis-aligned either.
    assert ns_centre(320.0) != ns_centre(1080.0)


@pytest.mark.parametrize("lane_id", [
    "ns_nb_inner", "ns_nb_outer", "ns_sb_inner", "ns_sb_outer",
    "ew_eb_inner", "ew_eb_outer", "ew_wb_inner", "ew_wb_outer",
])
def test_every_lane_contains_its_own_sample_points(lane_id, tmp_path):
    """point_in_lane must actually land inside the lane it names."""
    z = scene_zones(tmp_path)
    lane = z.lane(lane_id)
    assert lane is not None
    for along in (0.0, 0.25, 0.5, 0.75, 1.0):
        pt = point_in_lane(lane_id, along)
        assert lane.contains(pt), f"{lane_id} does not contain along={along} {pt}"


@pytest.mark.parametrize("lane_id", [
    "ns_nb_inner", "ns_nb_outer", "ns_sb_inner", "ns_sb_outer",
    "ew_eb_inner", "ew_eb_outer", "ew_wb_inner", "ew_wb_outer",
])
def test_travelling_along_a_lane_agrees_with_its_direction_vector(lane_id, tmp_path):
    """The single most important consistency check for wrong_way.

    If a lane's arrow disagreed with the direction point_in_lane advances in,
    every wrong_way test would be inverted and still pass.
    """
    z = scene_zones(tmp_path)
    lane = z.lane(lane_id)
    a = point_in_lane(lane_id, 0.1)
    b = point_in_lane(lane_id, 0.9)
    travel = (b[0] - a[0], b[1] - a[1])
    assert lane.heading_offset_deg(travel) < 20.0
    # ...and reversing it reads as against the lane.
    assert lane.heading_offset_deg((-travel[0], -travel[1])) > 160.0


def test_named_points_mean_what_they_say(tmp_path):
    z = scene_zones(tmp_path)

    assert z.on_carriageway(JAYWALK_POINT)
    assert z.in_crossing(JAYWALK_POINT) is None

    assert z.on_carriageway(CROSSING_POINT)
    assert z.in_crossing(CROSSING_POINT) is not None

    assert z.on_carriageway(QUEUE_POINT)
    assert z.in_signal_queue(QUEUE_POINT)

    assert z.on_carriageway(STOPPED_POINT)
    assert not z.in_signal_queue(STOPPED_POINT)

    assert not z.on_carriageway(OFF_ROAD_POINT)
    assert z.lane_of(OFF_ROAD_POINT) is None


def test_stop_line_approach_sides_are_derived_not_guessed(tmp_path):
    """A waiting vehicle must read as being on the approach side."""
    from fixtures.synthetic_scene import ew_y, ns_x

    z = scene_zones(tmp_path)
    waiting = {
        "sl_ns_south": (ns_x(880.0, 0.5), 880.0),
        "sl_ns_north": (ns_x(560.0, -0.5), 560.0),
        "sl_ew_west": (480.0, ew_y(480.0, 0.5)),
        "sl_ew_east": (1450.0, ew_y(1450.0, -0.5)),
    }
    for sl_id, pt in waiting.items():
        sl = z.stop_line(sl_id)
        assert sl is not None
        assert sl.signed_distance(pt) > 0, f"{sl_id}: waiting point is past the line"


def test_crossing_a_stop_line_inbound_is_directional(tmp_path):
    from fixtures.synthetic_scene import SL_NS_SOUTH_Y, ns_x

    z = scene_zones(tmp_path)
    sl = z.stop_line("sl_ns_south")
    below = (ns_x(SL_NS_SOUTH_Y + 30, 0.5), SL_NS_SOUTH_Y + 30)
    above = (ns_x(SL_NS_SOUTH_Y - 30, 0.5), SL_NS_SOUTH_Y - 30)
    assert sl.crossed_inbound(below, above) is True
    assert sl.crossed_inbound(above, below) is False


def test_intersection_box_resolves_to_the_major_road(tmp_path):
    """Documented behaviour: NS lanes are listed first, so lane_of picks them."""
    from fixtures.synthetic_scene import (INTERSECTION_Y_BOTTOM,
                                          INTERSECTION_Y_TOP, ns_x)

    z = scene_zones(tmp_path)
    mid_y = (INTERSECTION_Y_TOP + INTERSECTION_Y_BOTTOM) / 2.0
    lane = z.lane_of((ns_x(mid_y, 0.5), mid_y))
    assert lane is not None
    assert lane.id.startswith("ns_")


def test_scene_rescales_to_another_resolution(tmp_path):
    z = scene_zones(tmp_path, frame_size=(960, 540))
    assert z.scale == pytest.approx((0.5, 0.5))
    pt = point_in_lane("ns_nb_outer", 0.5)
    assert z.lane_of((pt[0] * 0.5, pt[1] * 0.5)) is not None


def test_scene_is_deterministic():
    assert json.dumps(build_zones_dict(), sort_keys=True) == \
        json.dumps(build_zones_dict(), sort_keys=True)


def test_scene_docstring_states_its_limits():
    """The fixture must say what it does not model, so nobody over-trusts it."""
    import fixtures.synthetic_scene as mod

    doc = mod.__doc__ or ""
    assert "DOES *NOT* MODEL" in doc
    for needle in ("DISTORTION", "CALIBRATION", "OCCLUSION", "SIGNAL PHASE"):
        assert needle in doc
