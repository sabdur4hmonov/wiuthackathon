"""zones.json schema, validator and loader.

The validator's job is to make a half-authored config impossible to run with.
The shipped config/zones.json must therefore FAIL validation until a human has
drawn the scene -- that is the single most important assertion here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config import ZONES_PATH
from src.zones import ZonesError, is_authored, load_zones, validate_raw

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# the shipped template
# ---------------------------------------------------------------------------
def test_shipped_zones_json_is_valid_json():
    json.loads(ZONES_PATH.read_text(encoding="utf-8"))


def test_shipped_zones_json_fails_validation_until_authored():
    """No fabricated coordinates. It must refuse to load, loudly."""
    raw = json.loads(ZONES_PATH.read_text(encoding="utf-8"))
    errors = validate_raw(raw)
    assert errors, "the shipped template must NOT pass validation"
    assert any("un-authored" in e or "positive int" in e for e in errors)


def test_shipped_zones_json_has_no_fabricated_geometry():
    """Every geometric field ships null."""
    raw = json.loads(ZONES_PATH.read_text(encoding="utf-8"))
    for section in ("carriageway", "lanes", "crossings", "signal_queue_zones"):
        for item in raw.get(section, []):
            assert item.get("polygon") is None, f"{section} has a polygon filled in"
    for item in raw.get("stop_lines", []) + raw.get("lane_markings", []):
        assert item.get("segment") is None
    for item in raw.get("traffic_lights", []):
        assert item.get("roi") is None
    aa = raw["authored_against"]
    assert aa["image_width"] is None and aa["image_height"] is None


def test_shipped_template_documents_every_section():
    """Each section carries a _draw note saying what to trace."""
    raw = json.loads(ZONES_PATH.read_text(encoding="utf-8"))
    for section in ("carriageway", "lanes", "stop_lines", "crossings",
                    "signal_queue_zones", "traffic_lights", "lane_markings"):
        assert raw[section], f"{section} has no template entry"
        assert any(k.startswith("_draw") for k in raw[section][0]), \
            f"{section} template has no _draw instruction"


def test_is_authored_is_false_for_the_template():
    assert is_authored(ZONES_PATH) is False


def test_load_zones_raises_on_the_template():
    with pytest.raises(ZonesError, match="not fully authored"):
        load_zones(ZONES_PATH)


# ---------------------------------------------------------------------------
# a complete fixture
# ---------------------------------------------------------------------------
def complete_zones() -> dict:
    """A minimal but fully authored scene: 2 lanes, a stop line, a crossing."""
    return {
        "schema_version": 1,
        "authored_against": {"image_width": 1920, "image_height": 1080,
                             "source_video": "clip1.mp4", "source_frame_idx": 0},
        "carriageway": [{"id": "main",
                         "polygon": [[0, 500], [1920, 500], [1920, 1080], [0, 1080]]}],
        "lanes": [
            {"id": "nb_1", "polygon": [[0, 500], [960, 500], [960, 1080], [0, 1080]],
             "direction_from": [100, 1000], "direction_to": [100, 600],
             "permitted_manoeuvres": ["through", "right"],
             "governed_by_stop_line": "sl_nb"},
            {"id": "sb_1", "polygon": [[960, 500], [1920, 500], [1920, 1080], [960, 1080]],
             "direction_from": [1800, 600], "direction_to": [1800, 1000],
             "permitted_manoeuvres": ["through"],
             "governed_by_stop_line": None},
        ],
        "stop_lines": [{"id": "sl_nb", "segment": [[0, 700], [960, 700]],
                        "approach_side": 1, "governs_lanes": ["nb_1"]}],
        "crossings": [{"id": "x_1",
                       "polygon": [[0, 620], [1920, 620], [1920, 700], [0, 700]]}],
        "signal_queue_zones": [{"id": "q_nb",
                                "polygon": [[0, 700], [960, 700], [960, 950], [0, 950]],
                                "lanes": ["nb_1"]}],
        "traffic_lights": [{"id": "sig_1", "roi": [1500, 100, 1560, 220],
                            "controls_lanes": ["nb_1"]}],
        "lane_markings": [{"id": "centre", "segment": [[960, 500], [960, 1080]],
                           "style": "solid", "separates": ["nb_1", "sb_1"]}],
        "notes": "test fixture",
    }


def write(tmp_path: Path, d: dict) -> Path:
    p = tmp_path / "zones.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def test_complete_fixture_validates(tmp_path):
    assert validate_raw(complete_zones()) == []
    z = load_zones(write(tmp_path, complete_zones()))
    assert len(z.lanes) == 2
    assert len(z.stop_lines) == 1
    assert z.has_traffic_light


def test_lane_direction_is_a_unit_vector(tmp_path):
    z = load_zones(write(tmp_path, complete_zones()))
    lane = z.lane("nb_1")
    assert lane is not None
    assert np.hypot(*lane.direction) == pytest.approx(1.0)
    # Drawn from y=1000 up to y=600, i.e. upward on screen: dy negative.
    assert lane.direction[1] == pytest.approx(-1.0)


def test_heading_offset_against_the_lane(tmp_path):
    z = load_zones(write(tmp_path, complete_zones()))
    lane = z.lane("nb_1")
    assert lane.heading_offset_deg((0.0, -1.0)) == pytest.approx(0.0)    # with
    assert lane.heading_offset_deg((0.0, 1.0)) == pytest.approx(180.0)   # against


def test_queries(tmp_path):
    z = load_zones(write(tmp_path, complete_zones()))
    assert z.on_carriageway((500, 900)) is True
    assert z.on_carriageway((500, 100)) is False
    assert z.lane_of((100, 900)).id == "nb_1"
    assert z.lane_of((1800, 900)).id == "sb_1"
    assert z.lane_of((500, 100)) is None
    assert z.in_crossing((500, 660)).id == "x_1"
    assert z.in_crossing((500, 900)) is None
    assert z.in_signal_queue((500, 800)) is True
    assert z.in_signal_queue((500, 1050)) is False


def test_stop_line_crossing_direction(tmp_path):
    z = load_zones(write(tmp_path, complete_zones()))
    sl = z.stop_line("sl_nb")
    # approach_side=+1, so approaching cars sit BELOW the line (larger y).
    assert sl.signed_distance((500, 900)) > 0
    # Driving up across the line = inbound.
    assert sl.crossed_inbound((500, 750), (500, 650)) is True
    # Rolling backwards over it is not running the light.
    assert sl.crossed_inbound((500, 650), (500, 750)) is False
    # Not crossing at all.
    assert sl.crossed_inbound((500, 900), (500, 800)) is False


def test_rescaling_to_a_different_video_resolution(tmp_path):
    """Authoring on 1080p and running 720p must not silently shift everything."""
    p = write(tmp_path, complete_zones())
    z = load_zones(p, frame_size=(960, 540))
    assert z.scale == pytest.approx((0.5, 0.5))
    # A point at (1000, 1800) in authored space maps to (500, 900).
    assert z.on_carriageway((250, 450)) is True
    assert np.hypot(*z.lane("nb_1").direction) == pytest.approx(1.0)


def test_solid_markings_filter(tmp_path):
    d = complete_zones()
    d["lane_markings"].append({"id": "dash", "segment": [[10, 10], [10, 200]],
                               "style": "dashed", "separates": []})
    z = load_zones(write(tmp_path, d))
    assert [m.id for m in z.solid_markings()] == ["centre"]


def test_empty_traffic_lights_is_legal(tmp_path):
    """No signal in frame is a real scene fact, not an omission."""
    d = complete_zones()
    d["traffic_lights"] = []
    assert validate_raw(d) == []
    z = load_zones(write(tmp_path, d))
    assert z.has_traffic_light is False


# ---------------------------------------------------------------------------
# what the validator must catch
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d.update(schema_version=2), "schema_version"),
    (lambda d: d["authored_against"].update(image_width=None), "image_width"),
    (lambda d: d.update(carriageway=[]), "carriageway must have"),
    (lambda d: d.update(lanes=[]), "lanes must have"),
    (lambda d: d["lanes"][0].update(polygon=None), "un-authored"),
    (lambda d: d["lanes"][0].update(polygon=[[0, 0], [1, 1]]), "at least 3 vertices"),
    (lambda d: d["lanes"][0].update(polygon=[[0, 0], [1, 0], [2, 0]]), "degenerate"),
    (lambda d: d["lanes"][0].update(direction_to=None), "direction arrow"),
    (lambda d: d["lanes"][0].update(direction_to=[100, 1000]), "zero length"),
    (lambda d: d["lanes"][0].update(permitted_manoeuvres=["fly"]), "unknown values"),
    (lambda d: d["lanes"][0].update(id="TODO_lane_1"), "TODO placeholder"),
    (lambda d: d["lanes"][1].update(id="nb_1"), "duplicated"),
    (lambda d: d["lanes"][0].update(governed_by_stop_line="nope"), "unknown stop line"),
    (lambda d: d["stop_lines"][0].update(segment=None), "un-authored"),
    (lambda d: d["stop_lines"][0].update(approach_side=0), "approach_side"),
    (lambda d: d["stop_lines"][0].update(governs_lanes=["ghost"]), "unknown lane"),
    (lambda d: d["crossings"][0].update(polygon=None), "un-authored"),
    (lambda d: d["traffic_lights"][0].update(roi=None), "un-authored"),
    (lambda d: d["traffic_lights"][0].update(roi=[10, 10, 5, 5]), "x2>x1"),
    (lambda d: d["lane_markings"][0].update(style=None), "style must be one of"),
    (lambda d: d["lane_markings"][0].update(style="wiggly"), "style must be one of"),
    (lambda d: d["lane_markings"][0].update(separates=["ghost"]), "unknown lane"),
    (lambda d: d["signal_queue_zones"][0].update(lanes=["ghost"]), "unknown lane"),
])
def test_validator_catches(mutate, needle):
    d = complete_zones()
    mutate(d)
    errors = validate_raw(d)
    assert any(needle in e for e in errors), \
        f"expected an error containing {needle!r}, got {errors}"


def test_missing_file_raises():
    with pytest.raises(ZonesError, match="not found"):
        load_zones(Path("does/not/exist.json"))


def test_malformed_json_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ZonesError, match="not valid JSON"):
        load_zones(p)


def test_validate_skipped_when_asked(tmp_path):
    """Tools that inspect a half-drawn file need this escape hatch."""
    d = complete_zones()
    d["notes"] = "partial"
    z = load_zones(write(tmp_path, d), validate=False)
    assert z.notes == "partial"


def test_comment_keys_are_ignored_by_the_validator():
    d = complete_zones()
    d["_README"] = ["chatter"]
    d["lanes"][0]["_draw"] = "trace this"
    assert validate_raw(d) == []
