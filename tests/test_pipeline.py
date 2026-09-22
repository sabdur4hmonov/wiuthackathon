"""Pipeline orchestration: the guarantees that protect the whole video.

detect_events must never raise and never overrun. Either failure makes the
harness discard the entry entirely -- events AND risk curve -- so these are the
most expensive bugs in the repo.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import cache as C
from src.budget import Budget
from src.pipeline import _get_tracks, _get_zones, detect_events
from src.tracks import TrackTable

cv2 = pytest.importorskip("cv2")


@pytest.fixture
def clip(tmp_path):
    p = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (96, 64))
    rng = np.random.default_rng(0)
    for _ in range(30):
        w.write(rng.integers(0, 255, (64, 96, 3), dtype=np.uint8))
    w.release()
    return p


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv(C.CACHE_ENV_VAR, str(tmp_path / "cache"))
    monkeypatch.delenv(C.DISABLE_ENV_VAR, raising=False)


# ---------------------------------------------------------------------------
# never raise
# ---------------------------------------------------------------------------
def test_returns_a_list_on_a_good_clip(clip, isolated_cache):
    assert isinstance(detect_events(str(clip), verbose=False), list)


@pytest.mark.parametrize("make", [
    lambda d: d / "missing.mp4",
    lambda d: d,                                 # a directory, not a file
])
def test_never_raises_on_bad_input(tmp_path, make, isolated_cache):
    assert detect_events(str(make(tmp_path)), verbose=False) == []


def test_never_raises_on_a_corrupt_file(tmp_path, isolated_cache):
    p = tmp_path / "corrupt.mp4"
    p.write_bytes(b"\x00\x01\x02 not a video")
    assert detect_events(str(p), verbose=False) == []


def test_a_broken_rule_does_not_cost_the_video(clip, isolated_cache, monkeypatch):
    """One bad rule must not become an empty prediction for everything."""
    from src import rules

    def exploding(tracks, zones):
        raise RuntimeError("rule bug")

    monkeypatch.setitem(rules.RULES, "accident", exploding)
    assert detect_events(str(clip), verbose=False) == []   # survived, no raise


def test_a_rules_failure_does_not_stop_other_rules(monkeypatch):
    from src import rules
    from src.rules import RawSegment

    monkeypatch.setattr(rules, "RULES", {
        "accident": lambda t, z: (_ for _ in ()).throw(RuntimeError("boom")),
        "near_miss": lambda t, z: [RawSegment(1.0, 3.0, "near_miss")],
    })
    out = rules.run_rules(TrackTable.empty(), None)
    assert [s.label for s in out] == ["near_miss"]


# ---------------------------------------------------------------------------
# zones
# ---------------------------------------------------------------------------
def test_unauthored_zones_yield_none_not_an_exception():
    """CP0 must run before the geometry exists."""
    t = TrackTable.empty(width=1920, height=1080)
    assert _get_zones(t, verbose=False) is None


def test_rules_receiving_none_zones_emit_nothing():
    """A guessed lane is worse than no lane."""
    from src.rules import run_rules

    assert run_rules(TrackTable.empty(), None) == []


# ---------------------------------------------------------------------------
# cache integration
# ---------------------------------------------------------------------------
@pytest.fixture
def fast_perception(monkeypatch):
    """Stub Stage 1 with a cheap, COMPLETE table.

    These tests are about cache behaviour, not detector speed. Running the real
    detector here would make them depend on whether this machine happens to have
    a GPU: on CPU a 1 s clip legitimately exceeds its own budget, perception is
    truncated, and a truncated table is deliberately not cached.
    """
    from src import perception
    from src.tracks import COLUMNS

    def stub(video_path, budget, cfg=None, verbose=True):
        data = np.arange(3 * len(COLUMNS), dtype=np.float32).reshape(3, -1)
        return TrackTable(data=data, fps=budget.fps, duration=budget.duration,
                          n_frames=budget.n_frames, width=96, height=64,
                          frame_stride=2, complete=True,
                          processed_until_sec=budget.duration)

    monkeypatch.setattr(perception, "run_perception", stub)
    return stub


def test_second_run_hits_the_cache(clip, isolated_cache, fast_perception):
    b1 = Budget.for_video(clip)
    t1 = _get_tracks(str(clip), b1, use_cache=True, verbose=False)
    assert t1.complete

    b2 = Budget.for_video(clip)
    t2 = _get_tracks(str(clip), b2, use_cache=True, verbose=False)
    assert np.array_equal(t1.data, t2.data)
    assert any("cache HIT" in (s.note or "") for s in b2.stages)


def test_no_cache_flag_forces_recompute(clip, isolated_cache, fast_perception):
    b1 = Budget.for_video(clip)
    _get_tracks(str(clip), b1, use_cache=True, verbose=False)
    b2 = Budget.for_video(clip)
    _get_tracks(str(clip), b2, use_cache=False, verbose=False)
    assert not any("cache HIT" in (s.note or "") for s in b2.stages)


def test_a_broken_cache_only_costs_time(clip, isolated_cache, monkeypatch):
    """The judges' box may not have a writable cache dir. It must not matter."""
    monkeypatch.setattr(C, "video_hash",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    assert isinstance(detect_events(str(clip), verbose=False), list)


def test_cache_and_no_cache_agree(clip, isolated_cache):
    cached = detect_events(str(clip), verbose=False, use_cache=True)
    fresh = detect_events(str(clip), verbose=False, use_cache=False)
    assert cached == fresh


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------
def test_timing_breakdown_is_recorded(clip, isolated_cache, capsys):
    detect_events(str(clip), verbose=True)
    err = capsys.readouterr().err
    assert "[budget]" in err
    for stage in ("perception", "rules", "postprocess"):
        assert stage in err


def test_budget_stop_still_returns_events(clip, isolated_cache, monkeypatch):
    """Degrade, never fail: a stopped run emits what it has."""
    from src import perception

    real = perception.run_perception

    def stop_immediately(video_path, budget, *a, **k):
        budget.force_stop("forced for test")
        return real(video_path, budget, *a, **k)

    monkeypatch.setattr(perception, "run_perception", stop_immediately)
    out = detect_events(str(clip), verbose=False)
    assert isinstance(out, list)


def test_truncated_perception_is_not_cached(clip, isolated_cache, monkeypatch):
    """A budget stop is one slow run, not a property of the video."""
    from src import perception
    from src.tracks import COLUMNS

    truncated = {"value": True}

    def stub(video_path, budget, cfg=None, verbose=True):
        data = np.arange(3 * len(COLUMNS), dtype=np.float32).reshape(3, -1)
        return TrackTable(data=data, fps=budget.fps, duration=budget.duration,
                          n_frames=budget.n_frames, width=96, height=64,
                          frame_stride=2, complete=not truncated["value"],
                          processed_until_sec=0.5)

    monkeypatch.setattr(perception, "run_perception", stub)

    b = Budget.for_video(clip)
    t = _get_tracks(str(clip), b, use_cache=True, verbose=False)
    assert t.complete is False

    # Nothing was stored, so the next run recomputes rather than serving a
    # table that stops where one slow machine happened to run out of time.
    truncated["value"] = False
    b2 = Budget.for_video(clip)
    t2 = _get_tracks(str(clip), b2, use_cache=True, verbose=False)
    assert not any("cache HIT" in (s.note or "") for s in b2.stages)
    assert t2.complete is True
