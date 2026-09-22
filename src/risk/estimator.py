"""The causal risk estimator.

Constraints this class is built around, all of them from evaluate.py:

* It is called for EVERY frame (official --risk-stride 1). Its per-frame cost is
  added to a budget that Part B's mandatory decode pass has already half spent,
  and an overrun wipes Part A's events too. So step() is cheap BY
  CONSTRUCTION: real work happens once every work_every_n_frames, and every
  other call returns the cached score off a branch and an integer compare.

* DEFAULT-QUIET. Alarm precision pools over all videos:
  precision = matched_alarms / all_alarms, summed across the whole test set. An
  alarm on a video with no accident cannot be matched by anything, so it is a
  pure false positive that drags down F1_alarm for every other video too. And
  alarms are runs of score >= 0.5, merged within 2 s -- so a score that idles
  near 0.5 and flickers across it manufactures alarms out of nothing. The score
  therefore sits at 0.0 and must be pushed up by a real signal.

* Causal. Only the frames already seen. No video file, no tracks cache.

* It must never raise: an exception is caught by the harness and costs the
  video its risk curve.

CP0 ships the pass-through: reset/step wired, cadence in place, score pinned at
the quiet default. That scores Score_B = 0 (a constant score is
chance-normalised to AP 0, and raises no alarms), which is the correct floor to
improve from -- not a negative.
"""
from __future__ import annotations

import numpy as np

from ..config import CFG, RiskConfig


class RiskEstimator:
    """P(an accident starts within the next 5 s), from past frames only."""

    def __init__(self, cfg: RiskConfig | None = None) -> None:
        self.cfg = cfg or CFG.risk
        self.meta: dict = {}
        self.fps: float = 25.0
        self.frame_count: int = 0
        self.last_score: float = self.cfg.default_score
        self._prev_gray: np.ndarray | None = None

    # -- harness interface --------------------------------------------------
    def reset(self, meta: dict) -> None:
        """Called once per video, before the first frame.

        meta = {"video_id", "fps", "width", "height", "n_frames"}.
        Every piece of per-video state must be cleared here: the harness reuses
        one estimator object per video, and leaking state across videos would
        make the score depend on folder order.
        """
        self.meta = dict(meta or {})
        try:
            self.fps = float(self.meta.get("fps") or 25.0) or 25.0
        except Exception:
            self.fps = 25.0
        self.frame_count = 0
        self.last_score = self.cfg.default_score
        self._prev_gray = None

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        """Return a score in [0, 1] for this frame. Must never raise."""
        try:
            n = self.frame_count
            self.frame_count = n + 1
            if n % max(1, self.cfg.work_every_n_frames) != 0:
                return self.last_score          # the cheap path: the common one
            score = self._score(frame, float(t_sec))
            if score != score:                  # NaN
                score = self.cfg.default_score
            self.last_score = float(
                min(self.cfg.max_score, max(0.0, score))
            )
            return self.last_score
        except Exception:
            # A crash here costs the whole risk curve. Hold the last value.
            return self.last_score

    # -- the actual signal --------------------------------------------------
    def _score(self, frame: np.ndarray, t_sec: float) -> float:
        """CP0: no signal yet, stay quiet.

        The planned signal is min time-to-collision between tracked vehicles,
        mapped so that 0.5 means "probably within 5 s". That needs a causal
        online tracker running inside this class -- it cannot borrow Stage 1's,
        which has seen the whole video. Until that exists, emitting anything
        above 0 would be noise that costs alarm precision across every video.
        """
        return self.cfg.default_score
