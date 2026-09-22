"""End-to-end contract with the organizers' unmodified harness.

These tests import the REAL run_submission.py and evaluate.py and run them over
a throwaway video. They are the ones that would catch "the repo is in a state
where the harness crashes", which is the failure that costs everything.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
cv2 = pytest.importorskip("cv2")


@pytest.fixture(scope="module")
def tiny_videos(tmp_path_factory) -> Path:
    """A folder with two short clips the harness can walk."""
    d = tmp_path_factory.mktemp("videos")
    for name, n in (("test_001.mp4", 40), ("test_002.mp4", 25)):
        w = cv2.VideoWriter(str(d / name), cv2.VideoWriter_fourcc(*"mp4v"),
                            25.0, (96, 64))
        rng = np.random.default_rng(0)
        for _ in range(n):
            w.write(rng.integers(0, 255, (64, 96, 3), dtype=np.uint8))
        w.release()
    return d


# ---------------------------------------------------------------------------
# the interface the harness requires
# ---------------------------------------------------------------------------
def test_solution_exposes_the_required_names():
    """run_submission.load_solution checks exactly these three."""
    sys.path.insert(0, str(ROOT))
    import solution

    for name in ("CLASSES", "detect_events", "RiskEstimator"):
        assert hasattr(solution, name), f"solution.py must define {name}"


def test_classes_is_a_subset_of_the_official_list():
    """'You may REMOVE classes you never predict; do not add new ids.'"""
    sys.path.insert(0, str(ROOT))
    import evaluate
    import solution

    assert set(solution.CLASSES) <= set(evaluate.OFFICIAL_CLASSES)


def test_classes_matches_what_the_rules_can_emit():
    """A class we list but never produce is a guaranteed 0 that grows |C|."""
    sys.path.insert(0, str(ROOT))
    import solution
    from src.rules import emitted_classes

    assert set(solution.CLASSES) == emitted_classes()


def test_detect_events_returns_a_valid_shape(tiny_videos):
    sys.path.insert(0, str(ROOT))
    import solution

    events = solution.detect_events(str(tiny_videos / "test_001.mp4"))
    assert isinstance(events, list)
    for ev in events:
        assert isinstance(ev, list) and len(ev) == 3
        s, e, label = ev
        assert isinstance(s, float) and isinstance(e, float)
        assert 0.0 <= s < e
        assert label in solution.CLASSES


def test_detect_events_never_raises_on_a_broken_file(tmp_path):
    """A crash costs the video BOTH parts; returning [] costs only Part A."""
    sys.path.insert(0, str(ROOT))
    import solution

    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"this is not a video")
    assert solution.detect_events(str(bad)) == []
    assert solution.detect_events(str(tmp_path / "does_not_exist.mp4")) == []


def test_detect_events_is_deterministic(tiny_videos):
    sys.path.insert(0, str(ROOT))
    import solution

    v = str(tiny_videos / "test_001.mp4")
    assert solution.detect_events(v) == solution.detect_events(v)


# ---------------------------------------------------------------------------
# the real harness, end to end
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def predictions(tiny_videos, tmp_path_factory) -> Path:
    """Run the organizers' run_submission.py exactly as they would."""
    out = tmp_path_factory.mktemp("out") / "predictions.json"
    res = subprocess.run(
        [sys.executable, "run_submission.py",
         "--videos", str(tiny_videos), "--out", str(out), "--team", "test-team"],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert out.exists()
    return out


def test_harness_runs_clean(predictions):
    data = json.loads(predictions.read_text())
    assert set(data["videos"]) == {"test_001.mp4", "test_002.mp4"}
    for vid, log in data["log"].items():
        assert log["errors"] == [], f"{vid} produced harness errors: {log['errors']}"


def test_harness_stays_inside_the_time_budget(predictions):
    """Over budget means the entry is discarded -- both parts, not just one."""
    data = json.loads(predictions.read_text())
    for vid, log in data["log"].items():
        assert log["total_sec"] <= log["budget_sec"], (
            f"{vid}: {log['total_sec']}s used of a {log['budget_sec']}s budget"
        )


def test_harness_writes_a_risk_curve_per_frame(predictions):
    """The harness records one sample per frame; an empty curve scores 0."""
    data = json.loads(predictions.read_text())
    assert len(data["videos"]["test_001.mp4"]["risk"]) == 40
    assert len(data["videos"]["test_002.mp4"]["risk"]) == 25


def test_risk_curve_is_quiet_by_default(predictions):
    """No alarm without a signal: alarm precision pools over every video."""
    data = json.loads(predictions.read_text())
    for vid, entry in data["videos"].items():
        scores = [s for _t, s in entry["risk"]]
        assert scores, f"{vid} has no risk curve"
        assert max(scores) < 0.5, f"{vid} would raise a spurious alarm"


def test_validate_only_passes(predictions):
    """The CP0 acceptance gate: a submittable package."""
    res = subprocess.run(
        [sys.executable, "evaluate.py", "--pred", str(predictions), "--validate-only"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "VALID" in res.stdout
    assert "INVALID" not in res.stdout


def test_predictions_pass_the_validator_in_process(predictions):
    sys.path.insert(0, str(ROOT))
    import evaluate

    errors, _warnings = evaluate.validate(json.loads(predictions.read_text()))
    assert errors == []


def test_shipped_predictions_samples_json_is_valid():
    """The file the submission package must contain."""
    sys.path.insert(0, str(ROOT))
    import evaluate

    p = ROOT / "predictions_samples.json"
    if not p.exists():
        pytest.skip("predictions_samples.json not generated yet")
    errors, _ = evaluate.validate(json.loads(p.read_text()))
    assert errors == []


# ---------------------------------------------------------------------------
# how the harness would punish us
# ---------------------------------------------------------------------------
def test_our_postprocessing_beats_the_harness_overlap_rule():
    """The harness DROPS the later of two overlapping same-class segments.

    We merge instead, so no detection is silently deleted. This pins the
    difference: feed both the same overlapping pair and compare.
    """
    sys.path.insert(0, str(ROOT))
    import run_submission

    from src.postprocess import finalise
    from src.rules import RawSegment

    pair = [RawSegment(10.0, 20.0, "accident"), RawSegment(15.0, 25.0, "accident")]

    ours = finalise(pair, duration=60.0)
    assert ours == [[10.0, 25.0, "accident"]]        # merged, both kept

    theirs, problems = run_submission.clean_events(
        [s.as_event() for s in pair], ["accident"], 60.0)
    assert theirs == [[10.0, 20.0, "accident"]]      # the second is deleted
    assert any("overlaps" in p for p in problems)


def test_harness_accepts_our_finalised_output_unchanged():
    """clean_events must not have to drop or alter anything we emit."""
    sys.path.insert(0, str(ROOT))
    import run_submission

    from src.postprocess import finalise
    from src.rules import RawSegment

    events = finalise(
        [RawSegment(0.0, 30.0, "accident"), RawSegment(7.0, 40.0, "accident"),
         RawSegment(41.0, 50.0, "near_miss"), RawSegment(-2.0, 3.0, "jaywalking")],
        duration=60.0)
    kept, problems = run_submission.clean_events(
        events, ["accident", "near_miss", "jaywalking"], 60.0)
    assert problems == []
    assert kept == sorted(events)
