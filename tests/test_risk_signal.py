"""The risk signal (time-to-collision) and its self-timing guard."""
import time

import numpy as np
import pytest

from src.config import RiskConfig
from src.risk.estimator import RiskEstimator, pair_scores

CFG = RiskConfig()
H = 100.0                                     # box height, px


def _score(pos, vel, person=(False, False)):
    return pair_scores(np.asarray(pos, float), np.asarray(vel, float),
                       np.full(len(pos), H), np.asarray(person, bool), CFG)


def test_head_on_one_second_apart_scores_one_half():
    # 400 px apart, closing at 400 px/s (4 heights/s): contact in ~1 s.
    s = _score([(0, 0), (400, 0)], [(200, 0), (-200, 0)])
    assert s == pytest.approx(0.5 ** (1.0 / CFG.ttc_half_sec), rel=0.05)


def test_further_away_scores_lower():
    near = _score([(0, 0), (400, 0)], [(200, 0), (-200, 0)])
    far = _score([(0, 0), (1600, 0)], [(200, 0), (-200, 0)])
    assert 0 < far < near


@pytest.mark.parametrize("pos,vel", [
    ([(0, 0), (0, 200)], [(300, 0), (300, 0)]),      # side by side, same speed
    ([(0, 0), (400, 0)], [(-200, 0), (200, 0)]),     # moving apart
    ([(0, 0), (400, 0)], [(0, 0), (0, 0)]),          # both stopped
    ([(0, 0), (400, 300)], [(200, 0), (-200, 0)]),   # pass each other 3 heights apart
])
def test_no_collision_course_scores_zero(pos, vel):
    assert _score(pos, vel) == 0.0


def test_two_pedestrians_never_count():
    assert _score([(0, 0), (400, 0)], [(200, 0), (-200, 0)], person=(True, True)) == 0.0
    assert _score([(0, 0), (400, 0)], [(200, 0), (-200, 0)], person=(True, False)) > 0.0


def test_guard_stops_work_past_its_own_budget():
    est = RiskEstimator(RiskConfig(work_budget_frac=0.01))
    est.reset({"fps": 25.0, "n_frames": 250})       # 10 s video: 0.1 s of work allowed
    est.work_sec = 0.2
    assert est._guard() is False
    assert "own work" in est.stopped
    est.last_score = 0.7
    # No work, and back to quiet: a held 0.7 would be an alarm to the end.
    assert est._score(np.zeros((64, 64, 3), np.uint8), 1.0) == 0.0


def test_guard_stops_when_part_b_projects_past_its_limit():
    est = RiskEstimator(RiskConfig(part_b_wall_limit_x=2.0))
    est.reset({"fps": 25.0, "n_frames": 250})       # 10 s video
    est.frame_count = 25                            # 10% through ...
    est.t_reset = time.perf_counter() - 3.0         # ... after 3 s: 30 s = 3.0x projected
    assert est._guard() is False
    assert "projected" in est.stopped


def test_guard_waits_for_enough_progress_before_extrapolating():
    est = RiskEstimator()
    est.reset({"fps": 25.0, "n_frames": 10_000})
    est.frame_count = 10                            # 0.1%: too early to judge
    est.t_reset = time.perf_counter() - 30.0
    assert est._guard() is True


def test_stopped_stays_stopped_and_reset_clears_it():
    est = RiskEstimator()
    est.reset({"fps": 25.0, "n_frames": 250})
    est.stopped = "x"
    assert est._guard() is False
    est.reset({"fps": 25.0, "n_frames": 250})
    assert est.stopped is None and est.work_sec == 0.0
