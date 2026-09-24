"""Budget manager tests.

The failure this guards against is total: an overrun makes the harness discard
BOTH parts for that video. So the limits are pinned numerically, and the
degradation path is checked to return rather than raise.
"""
from __future__ import annotations

import time

import pytest

from src.budget import Budget
from src.config import BudgetConfig


def mk(duration=100.0, fps=25.0, elapsed=0.0, cfg=None) -> Budget:
    """A budget whose clock is already `elapsed` seconds in."""
    return Budget(duration=duration, fps=fps, n_frames=int(duration * fps),
                  t0=time.perf_counter() - elapsed, cfg=cfg or BudgetConfig())


def test_limits_match_the_harness_factor():
    b = mk(duration=100.0)
    assert b.total_budget == pytest.approx(300.0)     # run_submission: 3.0x
    assert b.part_a_target == pytest.approx(120.0)
    assert b.part_b_reserve == pytest.approx(120.0)


def test_part_a_hard_leaves_room_for_part_b():
    """Part B re-decodes the whole video; its reserve is not negotiable."""
    b = mk(duration=100.0)
    assert b.part_a_hard == pytest.approx(150.0)      # the 1.5x factor binds
    assert b.part_a_hard + b.part_b_reserve <= b.total_budget


def test_reserve_binds_on_short_clips():
    """On a 5 s clip the safety margin is what limits Part A, not 1.5x."""
    b = mk(duration=5.0)
    assert b.safety_margin == pytest.approx(2.0)       # flat 2 s still applies
    by_factor = 1.5 * 5.0                              # 7.5
    by_reserve = 3.0 * 5.0 - 1.2 * 5.0 - 2.0           # 7.0
    assert b.part_a_hard == pytest.approx(min(by_factor, by_reserve))
    assert b.part_a_hard == pytest.approx(7.0)


def test_safety_margin_is_capped_as_a_fraction_of_the_budget():
    """A flat 2 s margin exceeds the whole Part A allowance on a ~1 s clip."""
    b = mk(duration=1.2)
    assert b.total_budget == pytest.approx(3.6)
    assert b.safety_margin == pytest.approx(0.15 * 3.6)     # 0.54, not 2.0
    assert b.part_a_hard > 0.5, "a short clip must still get usable Part A time"
    assert b.part_a_hard == pytest.approx(min(1.5 * 1.2,
                                              3.6 - 1.2 * 1.2 - 0.54))


def test_long_clips_keep_the_flat_margin():
    b = mk(duration=600.0)
    assert b.safety_margin == pytest.approx(2.0)


def test_part_a_hard_never_goes_negative():
    b = mk(duration=0.5)
    assert b.part_a_hard >= 0.0


def test_should_stop_flips_past_the_hard_limit():
    b = mk(duration=10.0, elapsed=0.0)
    assert b.should_stop() is False
    b = mk(duration=10.0, elapsed=99.0)
    assert b.should_stop() is True
    assert "hard stop" in b.stop_reason


def test_stop_reason_is_sticky():
    b = mk(duration=10.0, elapsed=99.0)
    assert b.should_stop() is True
    reason = b.stop_reason
    assert b.should_stop() is True
    assert b.stop_reason == reason                     # not recomputed


def test_over_target_is_softer_than_should_stop():
    b = mk(duration=100.0, elapsed=130.0)              # past 1.2x, under 1.5x
    assert b.over_target() is True
    assert b.should_stop() is False


def test_project_overrun_catches_a_slow_run_early():
    """Bail at frame 200 of 4000, not at frame 3900 with no time left."""
    b = mk(duration=100.0, elapsed=30.0)
    # 200 of 4000 frames in 30 s projects to ~600 s, way past the 150 s limit.
    assert b.project_overrun(done=200, total=4000) is True
    # 2000 of 4000 in 30 s projects to 60 s: fine.
    assert b.project_overrun(done=2000, total=4000) is False


def test_project_overrun_excludes_the_one_off_startup_cost():
    """The detector load must not be amortised across the first few frames.

    Without loop_t0 the model-load seconds get divided by `done` and multiplied
    back up by `total`, which aborts a run that would have finished inside a
    fraction of its budget. This is the exact bug the loop_t0 argument fixes.
    """
    b = mk(duration=60.0, elapsed=10.0)          # 9 s loading, 1 s of frames
    loop_t0 = time.perf_counter() - 1.0          # the frame loop began 1 s ago
    # 10 of 750 frames in 1 s -> 0.1 s/frame -> 10 + 740*0.1 = 84 s.
    # part_a_hard is 60*1.5 = 90 s, so this must NOT trip.
    assert b.project_overrun(done=10, total=750, loop_t0=loop_t0) is False
    # Counting the 9 s load as per-frame cost would project 10 s/frame: a bail.
    assert b.project_overrun(done=10, total=750) is True


def test_project_overrun_still_trips_when_the_frames_really_are_slow():
    b = mk(duration=60.0, elapsed=20.0)
    loop_t0 = time.perf_counter() - 15.0         # 15 s of frame loop
    # 10 of 750 frames in 15 s -> 1.5 s/frame -> 20 + 740*1.5 = 1130 s.
    assert b.project_overrun(done=10, total=750, loop_t0=loop_t0) is True


def test_project_overrun_is_safe_on_degenerate_counts():
    b = mk(duration=100.0, elapsed=10.0)
    assert b.project_overrun(0, 100) is False
    assert b.project_overrun(10, 0) is False
    assert b.project_overrun(-1, 100) is False
    assert b.project_overrun(100, 100) is False      # already finished


def test_force_stop():
    b = mk(duration=100.0)
    assert b.should_stop() is False
    b.force_stop("manual")
    assert b.should_stop() is True
    assert b.stop_reason == "manual"


def test_stage_timing_is_recorded():
    b = mk(duration=10.0)
    with b.stage("perception"):
        time.sleep(0.01)
    assert [s.name for s in b.stages] == ["perception"]
    assert b.stages[0].seconds >= 0.005


def test_stage_records_even_when_the_body_raises():
    """A failing stage must still appear in the breakdown."""
    b = mk(duration=10.0)
    with pytest.raises(ValueError):
        with b.stage("rules"):
            raise ValueError("boom")
    assert [s.name for s in b.stages] == ["rules"]


def test_report_shape():
    b = mk(duration=100.0, elapsed=12.0)
    b.note("perception", 10.0, "cache HIT")
    r = b.report()
    assert r["duration_sec"] == 100.0
    assert r["total_budget_sec"] == 300.0
    assert r["part_a_x_realtime"] == pytest.approx(0.12, abs=0.02)
    assert r["stopped_early"] is False
    assert r["stages"][0]["name"] == "perception"
    assert "cache HIT" in b.format_report()


def test_report_handles_zero_duration():
    b = mk(duration=0.0)
    r = b.report()
    assert r["part_a_x_realtime"] is None
    b.format_report()                                  # must not raise


def test_probe_duration_uses_the_harness_formula(tmp_path):
    """n_frames / fps, NOT the container duration -- they can disagree."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from src.budget import probe_duration

    p = tmp_path / "t.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
    for _ in range(50):
        w.write(np.zeros((48, 64, 3), dtype=np.uint8))
    w.release()

    duration, fps, n_frames = probe_duration(p)
    assert fps == pytest.approx(25.0, abs=0.5)
    assert n_frames == 50
    assert duration == pytest.approx(n_frames / fps)


def test_probe_duration_raises_on_an_unopenable_file(tmp_path):
    from src.budget import probe_duration

    bad = tmp_path / "not_a_video.mp4"
    bad.write_bytes(b"nonsense")
    with pytest.raises(RuntimeError, match="cannot open"):
        probe_duration(bad)



# ---------------------------------------------------------------------------
# measured Part B reserve
# ---------------------------------------------------------------------------
def test_part_b_reserve_defaults_to_the_fixed_multiplier():
    """No measurement taken yet -> the old CP0 behaviour, unchanged."""
    b = mk(duration=100.0)
    assert b.part_b_reserve == pytest.approx(1.2 * 100.0)


def test_set_measured_part_b_overrides_the_fixed_multiplier():
    b = mk(duration=100.0)
    b.set_measured_part_b(40.0)
    # 40.0 * the configured safety headroom (default 1.30), not the 1.2x guess.
    assert b.part_b_reserve == pytest.approx(40.0 * 1.30)
    assert b.part_b_reserve != pytest.approx(1.2 * 100.0)


def test_set_measured_part_b_feeds_part_a_hard():
    """The whole point: a real measurement must actually move the hard stop.

    Uses a 5 s clip, same regime as test_reserve_binds_on_short_clips: at
    duration=100 the 1.5x factor cap already binds regardless of the reserve
    (see test_part_a_hard_leaves_room_for_part_b), so a change to
    part_b_reserve would not be visible through part_a_hard there -- the test
    would pass for the wrong reason. At 5 s the RESERVE is what actually
    constrains part_a_hard, so a real measurement moving it is observable.
    """
    cfg = BudgetConfig()
    guessed = mk(duration=5.0, cfg=cfg)
    assert guessed.part_a_hard == pytest.approx(7.0)     # reserve-bound, see above

    measured_low = mk(duration=5.0, cfg=cfg)
    measured_low.set_measured_part_b(1.0)      # far cheaper than the 1.2x guess
    measured_high = mk(duration=5.0, cfg=cfg)
    measured_high.set_measured_part_b(10.0)    # far more expensive

    # A cheaper real Part B frees up more time for Part A, up to the 1.5x cap...
    assert measured_low.part_a_hard > guessed.part_a_hard
    assert measured_low.part_a_hard == pytest.approx(7.5)
    # ...and a more expensive one claws it back, never going negative.
    assert measured_high.part_a_hard < guessed.part_a_hard
    assert measured_high.part_a_hard >= 0.0


def test_set_measured_part_b_rejects_negative_input():
    b = mk(duration=100.0)
    b.set_measured_part_b(-5.0)
    assert b.part_b_reserve >= 0.0


def test_report_flags_whether_the_reserve_was_measured():
    b = mk(duration=100.0)
    assert b.report()["part_b_reserve_measured"] is False
    b.set_measured_part_b(30.0)
    assert b.report()["part_b_reserve_measured"] is True


def test_measure_part_b_floor_on_a_real_small_clip(tmp_path):
    """The probe must time cap.read() -- run_risk own call -- not grab()."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from src.budget import measure_part_b_floor

    p = tmp_path / "probe.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
    for _ in range(60):
        w.write(np.zeros((48, 64, 3), dtype=np.uint8))
    w.release()

    rates = []
    estimate, probe_wall = measure_part_b_floor(p, n_frames=60, n_probe=21,
                                                rates_out=rates)
    assert estimate is not None
    assert estimate > 0.0
    assert probe_wall > 0.0
    # Three probes at different positions; the estimate is their MEDIAN rate
    # over the whole clip, so one slow stretch cannot decide the budget.
    assert len(rates) == 3
    assert estimate == pytest.approx(float(np.median(rates)) * 60)


def test_measure_part_b_floor_bounds_its_own_wall_time(tmp_path):
    """The probe protects the budget it feeds; it must not itself run long."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from src.budget import measure_part_b_floor

    p = tmp_path / "probe2.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (32, 32))
    for _ in range(500):
        w.write(np.zeros((32, 32, 3), dtype=np.uint8))
    w.release()

    _estimate, probe_wall = measure_part_b_floor(
        p, n_frames=500, n_probe=500, max_probe_sec=0.05)
    assert probe_wall < 2.0, f"probe ran {probe_wall:.2f}s against a 0.05s cap"


def test_measure_part_b_floor_on_an_unopenable_file_returns_none(tmp_path):
    from src.budget import measure_part_b_floor

    bad = tmp_path / "not_a_video.mp4"
    bad.write_bytes(b"nonsense")
    estimate, probe_wall = measure_part_b_floor(bad, n_frames=100)
    assert estimate is None
    assert probe_wall >= 0.0


def test_measure_part_b_floor_on_a_zero_frame_probe(tmp_path):
    """n_probe=0 must not divide by zero or read forever."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from src.budget import measure_part_b_floor

    p = tmp_path / "probe3.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (32, 32))
    for _ in range(10):
        w.write(np.zeros((32, 32, 3), dtype=np.uint8))
    w.release()

    estimate, _wall = measure_part_b_floor(p, n_frames=10, n_probe=0)
    # n_probe clamped to at least 1, so a real frame is still read.
    assert estimate is not None
