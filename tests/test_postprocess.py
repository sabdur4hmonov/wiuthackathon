"""Stage 3 tests -- especially the no-overlap guarantee.

The harness resolves a same-class overlap by keeping the earlier-starting
segment and DROPPING the other. That silently deletes real detections, so
Stage 3 must merge them first and the result must satisfy evaluate.py's own
overlap rule (strict `s2 < e1` on sorted same-class segments).
"""
from __future__ import annotations

import pytest

from src.config import PostConfig
from src.postprocess import (assert_no_overlap, clamp, drop_blips,
                             enforce_no_overlap, finalise, merge_same_class)
from src.rules import RawSegment


def S(start, end, label="accident", score=1.0):
    return RawSegment(start, end, label, score)


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------
def test_clamp_to_duration():
    out = clamp([S(-5.0, 10.0), S(50.0, 120.0)], duration=60.0)
    assert [(s.start, s.end) for s in out] == [(0.0, 10.0), (50.0, 60.0)]


def test_clamp_drops_zero_length_after_clipping():
    assert clamp([S(60.0, 70.0)], duration=60.0) == []
    assert clamp([S(5.0, 5.0)], duration=60.0) == []


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------
def test_merge_joins_fragments_within_gap():
    """A tracker ID switch splits one event in two; merging recovers the IoU."""
    out = merge_same_class([S(10.0, 12.0), S(12.5, 15.0)], gap=1.0)
    assert len(out) == 1
    assert (out[0].start, out[0].end) == (10.0, 15.0)


def test_merge_leaves_distant_segments_alone():
    out = merge_same_class([S(10.0, 12.0), S(20.0, 22.0)], gap=1.0)
    assert len(out) == 2


def test_merge_is_per_class():
    """Different classes may overlap; the task spec requires it."""
    out = merge_same_class([S(10.0, 20.0, "accident"), S(11.0, 19.0, "wrong_way")],
                           gap=1.0)
    assert len(out) == 2


def test_merge_absorbs_contained_segment():
    out = merge_same_class([S(10.0, 30.0), S(15.0, 20.0)], gap=0.0)
    assert len(out) == 1
    assert (out[0].start, out[0].end) == (10.0, 30.0)


def test_merge_chains_three_fragments():
    out = merge_same_class([S(0.0, 2.0), S(2.5, 4.0), S(4.5, 6.0)], gap=1.0)
    assert len(out) == 1
    assert (out[0].start, out[0].end) == (0.0, 6.0)


def test_merge_keeps_best_score():
    out = merge_same_class([S(0.0, 2.0, score=0.2), S(2.2, 4.0, score=0.9)], gap=1.0)
    assert out[0].score == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# blips
# ---------------------------------------------------------------------------
def test_drop_blips():
    out = drop_blips([S(0.0, 0.2), S(1.0, 3.0)], min_duration=0.5)
    assert len(out) == 1
    assert out[0].start == 1.0


def test_blip_exactly_at_threshold_is_kept():
    assert len(drop_blips([S(0.0, 0.5)], min_duration=0.5)) == 1


# ---------------------------------------------------------------------------
# the overlap guarantee
# ---------------------------------------------------------------------------
def test_enforce_no_overlap_merges_rather_than_drops():
    """The point of the whole exercise: keep both detections, as one segment."""
    out = enforce_no_overlap([S(10.0, 20.0), S(15.0, 25.0)])
    assert len(out) == 1
    assert (out[0].start, out[0].end) == (10.0, 25.0)


def test_touching_segments_are_legal():
    """evaluate.py uses strict `s2 < e1`, so s2 == e1 must survive as two."""
    out = enforce_no_overlap([S(10.0, 20.0), S(20.0, 30.0)])
    assert len(out) == 2
    assert_no_overlap([s.as_event() for s in out])


def test_enforce_no_overlap_is_idempotent():
    once = enforce_no_overlap([S(0, 5), S(3, 9), S(8, 12)])
    twice = enforce_no_overlap(once)
    assert [(s.start, s.end) for s in once] == [(s.start, s.end) for s in twice]


def test_assert_no_overlap_raises_on_a_real_overlap():
    with pytest.raises(AssertionError, match="overlapping"):
        assert_no_overlap([[0.0, 10.0, "accident"], [5.0, 15.0, "accident"]])


def test_assert_no_overlap_allows_different_classes():
    assert_no_overlap([[0.0, 10.0, "accident"], [5.0, 15.0, "near_miss"]])


# ---------------------------------------------------------------------------
# finalise: the whole stage
# ---------------------------------------------------------------------------
def test_finalise_produces_harness_ready_events():
    events = finalise([S(10.0, 12.0), S(12.5, 15.0), S(30.0, 30.1)], duration=60.0)
    assert events == [[10.0, 15.0, "accident"]]      # merged, blip dropped


def test_finalise_never_emits_an_overlap_however_messy_the_input():
    messy = [S(0, 30), S(5, 8), S(7, 40), S(39, 50), S(60, 61), S(-3, 2)]
    events = finalise(messy, duration=55.0)
    assert_no_overlap(events)                        # would raise inside finalise too
    assert events == [[0.0, 50.0, "accident"]]


def test_finalise_sorts_by_start():
    events = finalise([S(30.0, 40.0, "near_miss"), S(5.0, 10.0, "accident")],
                      duration=60.0)
    assert [e[0] for e in events] == sorted(e[0] for e in events)


def test_finalise_respects_config():
    cfg = PostConfig(min_duration_sec=3.0, merge_gap_sec=0.0)
    events = finalise([S(0.0, 2.0), S(2.5, 4.0)], duration=60.0, cfg=cfg)
    assert events == []                              # neither fragment reaches 3 s


def test_finalise_empty_input():
    assert finalise([], duration=60.0) == []


def test_finalise_output_passes_the_official_validator():
    """Run evaluate.py's real validator over a predictions file we built."""
    import evaluate

    events = finalise([S(0, 30), S(7, 40), S(41, 50, "near_miss")], duration=60.0)
    pred = {"team": "t", "videos": {"v.mp4": {"events": events, "risk": []}}}
    errors, _ = evaluate.validate(pred)
    assert errors == []
