"""tools/scan_windows.py is a search aid for manual review -- never a rule.

These tests pin both halves of that: it flags what the rules are too strict to
report (so a person looks), and it cannot leak into what the pipeline emits.
"""
import importlib
import sys
from pathlib import Path

import pytest

from fixtures.synthetic_scene import STOPPED_POINT, point_in_lane, scene_zones
from fixtures.synthetic_tracks import TrackBuilder
from src.rules import RULES, emitted_classes
from src.rules import stopped_vehicle as stopped_mod

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
scan_windows = importlib.import_module("scan_windows")


@pytest.fixture(scope="module")
def zones(tmp_path_factory):
    return scene_zones(tmp_path_factory.mktemp("zones"))


def _stop(seconds_stopped: float):
    b = TrackBuilder(duration=40.0)
    (b.vehicle()
      .start_at(point_in_lane("ns_nb_outer", 0.02), t=0.0)
      .drive_to(STOPPED_POINT, speed=220.0)
      .stop_for(seconds_stopped)
      .drive_to(point_in_lane("ns_nb_outer", 0.9), speed=220.0))
    return b.build()


def test_a_stop_too_short_for_the_rule_is_still_flagged_for_a_person(zones):
    tracks = _stop(7.0)                         # the rule wants 10 s
    assert stopped_mod.detect(tracks, zones) == []
    ws = scan_windows.stationary_windows(tracks, zones)
    assert len(ws) == 1
    assert ws[0].label == "stopped_vehicle"
    assert ws[0].end - ws[0].start == pytest.approx(7.0, abs=1.5)


def test_moving_traffic_is_not_flagged_as_stationary(zones):
    b = TrackBuilder(duration=30.0)
    (b.vehicle()
      .start_at(point_in_lane("ns_nb_outer", 0.02), t=0.0)
      .drive_to(point_in_lane("ns_nb_outer", 0.98), speed=120.0))
    assert scan_windows.stationary_windows(b.build(), zones) == []


def test_caps_hold(zones):
    tracks = _stop(7.0)
    ws = scan_windows.scan(tracks, zones)
    for kind, cap in scan_windows.CAPS.items():
        assert sum(w.kind == kind for w in ws) <= cap


def test_scanning_does_not_change_what_the_pipeline_emits(zones):
    before_rules, before_classes = set(RULES), emitted_classes()
    scan_windows.scan(_stop(7.0), zones)
    assert set(RULES) == before_rules == {"congestion", "stopped_vehicle", "jaywalking", "wrong_way"}
    assert emitted_classes() == before_classes


def test_nothing_in_src_or_solution_imports_the_scan_aid():
    offenders = [p for p in list((ROOT / "src").rglob("*.py")) + [ROOT / "solution.py"]
                 if "scan_windows" in p.read_text(encoding="utf-8")]
    assert offenders == []
