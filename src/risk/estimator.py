"""The causal risk estimator: time-to-collision from the frames step() is handed.

Constraints this class is built around, all of them from evaluate.py and the
harness:

* COST. step() is called for EVERY frame, inside the same 3x budget as Part A,
  after the harness's own full decode of the 4K file -- the biggest cost we
  have (1.2-3.0x realtime measured on the 4-core dev laptop). An overrun wipes
  the video's Part A events too. So:
    - real work runs once every work_every_n_frames (5 Hz); every other call
      returns the cached score off a branch and an integer compare;
    - a SELF-TIMING GUARD stops the work for good, for this video, as soon as
      (a) our own work passes work_budget_frac of the video's duration, or
      (b) Part B's projected wall time -- the harness's decode plus our work,
      extrapolated from progress so far -- passes part_b_wall_limit_x of the
      duration. After that step() returns the quiet default, 0. On a GPU box
      the work is ~20-30 ms per sample (~0.1-0.15x); on a CPU-only box
      detection alone is ~0.15 s per sample, so the guard trips early and
      Part B degrades to the quiet default rather than risking the video.

* CAUSAL. Only frames already handed to step(). Its own detector instance and
  its own tracker, reset per video. Nothing from Part A (tracks, events,
  zones, cache) and no video file: tests/test_risk_isolation.py enforces it
  structurally.

* DEFAULT-QUIET. Alarm precision pools over all videos, so an alarm on an
  accident-free video is a pure false positive everywhere. The score is 0
  unless two tracked road users are on a collision course: at closest
  approach within collision_radius_L box heights, closing at least
  min_closing_L_s. Score = 0.5 ** (ttc / ttc_half_sec) -- 0.5 at a 0.5 s
  time-to-collision -- and it must hold for persist_samples consecutive
  samples. Vehicle-vehicle and vehicle-pedestrian pairs only. Settings and
  the false-alarm measurement behind them: src/config.py RiskConfig.

  Score_B = 0.4 AP + 0.4 F1_alarm + 0.2 mTTA/W; every term is >= 0, so a
  signal can never score below the constant 0 it replaces -- the only real
  risk is time, which the guard owns.

* It must never raise: an exception is caught by the harness and costs the
  video its risk curve.

UNCALIBRATED: the sample clips contain no accident, so the thresholds are
set from first principles and checked only for their false-alarm rate on
those clips (see README, "Part B").
"""
from __future__ import annotations

import time

import numpy as np

from ..config import CFG, WEIGHTS_DIR, RiskConfig, enforce_offline

PERSON = 0
_MODEL = None
_MODEL_FAILED = False


def _load_model():
    """Our own detector instance, from the same local weights as Stage 1.
    Lazy: importing src.risk must not import ultralytics."""
    global _MODEL, _MODEL_FAILED
    if _MODEL is not None or _MODEL_FAILED:
        return _MODEL
    try:
        from ultralytics import YOLO

        from ..yolo import _disable_amp_check

        enforce_offline()
        path = WEIGHTS_DIR / CFG.perception.weights
        if not path.exists():
            raise FileNotFoundError(path)
        _MODEL = YOLO(str(path))
        _disable_amp_check(_MODEL)
    except Exception:  # noqa: BLE001 - no detector means a quiet risk curve, never a crash
        _MODEL_FAILED = True
        _MODEL = None
    return _MODEL


def pair_scores(pos: np.ndarray, vel: np.ndarray, size: np.ndarray, is_person: np.ndarray,
                cfg: RiskConfig) -> float:
    """Highest collision score over vehicle-vehicle and vehicle-person pairs.

    pos, vel: (n, 2) ground point (px) and velocity (px/s); size: (n,) box
    height (px). Straight-line extrapolation: time of closest approach
    t* = -r.w / |w|^2, distance there d* = |r + w t*|.
    """
    n = len(pos)
    if n < 2:
        return 0.0
    r = pos[None, :, :] - pos[:, None, :]              # j - i
    w = vel[None, :, :] - vel[:, None, :]
    ww = (w ** 2).sum(-1)
    rw = (r * w).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_star = np.where(ww > 1e-9, -rw / ww, np.inf)
        dist = np.hypot(r[..., 0], r[..., 1])
        closing = np.where(dist > 1e-9, -rw / dist, 0.0)
        d_star = np.hypot(r[..., 0] + w[..., 0] * t_star, r[..., 1] + w[..., 1] * t_star)
    L = (size[:, None] + size[None, :]) / 2.0
    both_people = is_person[:, None] & is_person[None, :]
    ok = ((t_star > 0) & np.isfinite(t_star)
          & (d_star <= cfg.collision_radius_L * L)
          & (closing >= cfg.min_closing_L_s * L)
          & ~both_people)
    np.fill_diagonal(ok, False)
    if not ok.any():
        return 0.0
    ttc = float(t_star[ok].min())
    return float(0.5 ** (ttc / cfg.ttc_half_sec))


class RiskEstimator:
    """P(an accident starts within the next 5 s), from past frames only."""

    def __init__(self, cfg: RiskConfig | None = None) -> None:
        self.cfg = cfg or CFG.risk
        self.reset({})

    # -- harness interface --------------------------------------------------
    def reset(self, meta: dict) -> None:
        """Called once per video, before the first frame.

        meta = {"video_id", "fps", "width", "height", "n_frames"}.
        Every piece of per-video state is cleared here: the harness may reuse
        one estimator object, and leaking state across videos would make the
        score depend on folder order.
        """
        self.meta = dict(meta or {})
        try:
            self.fps = float(self.meta.get("fps") or 25.0) or 25.0
        except Exception:
            self.fps = 25.0
        try:
            self.n_frames = int(self.meta.get("n_frames") or 0)
        except Exception:
            self.n_frames = 0
        self.duration = self.n_frames / self.fps if self.n_frames else 0.0
        self.frame_count = 0
        self.last_score = self.cfg.default_score
        self.t_reset = time.perf_counter()
        self.work_sec = 0.0
        self.stopped: str | None = None           # why the guard stopped the work
        self._tracker = None
        self._hist: dict[int, list[tuple[float, float, float, float]]] = {}
        self._raw: list[float] = []

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
            self.last_score = float(min(self.cfg.max_score, max(0.0, score)))
            return self.last_score
        except Exception:
            # A crash here costs the whole risk curve. Hold the last value.
            return self.last_score

    # -- the guard ----------------------------------------------------------
    def _guard(self) -> bool:
        """False once the work must stop for this video (and stays stopped)."""
        if self.stopped:
            return False
        if self.duration <= 0:
            self.stopped = "unknown duration"
            return False
        if self.work_sec > self.cfg.work_budget_frac * self.duration:
            self.stopped = f"own work {self.work_sec:.1f}s > {self.cfg.work_budget_frac:.2f}x"
            return False
        progress = self.frame_count / max(1, self.n_frames)
        if progress >= self.cfg.guard_min_progress:
            projected = (time.perf_counter() - self.t_reset) / progress
            if projected > self.cfg.part_b_wall_limit_x * self.duration:
                self.stopped = (f"Part B projected {projected / self.duration:.2f}x "
                                f"> {self.cfg.part_b_wall_limit_x:.2f}x")
                return False
        return True

    # -- the actual signal --------------------------------------------------
    def _score(self, frame: np.ndarray, t_sec: float) -> float:
        if not self._guard():
            # Back to the quiet default, not the last score: holding a high
            # value from the moment the guard tripped would be one alarm (and
            # a run of wrong high-score frames) to the end of the video.
            return self.cfg.default_score
        t0 = time.perf_counter()
        try:
            return self._signal(frame, t_sec)
        finally:
            self.work_sec += time.perf_counter() - t0

    def _signal(self, frame, t_sec: float) -> float:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[0] < 16 \
                or frame.shape[1] < 16:
            return self.last_score
        model = _load_model()
        if model is None:
            self.stopped = "no detector"
            return self.cfg.default_score
        import cv2

        from .. import tracker as trk
        from ..yolo import _fp16_kwarg, device_string

        h, w = frame.shape[:2]
        size = self.cfg.detect_imgsz
        if w > size:
            nh = max(2, int(round(h * size / w / 2)) * 2)
            small = cv2.resize(frame, (size, nh), interpolation=cv2.INTER_AREA)
        else:
            small = frame
        sx, sy = w / small.shape[1], h / small.shape[0]
        device = device_string()
        kw = dict(imgsz=size, conf=self.cfg.detect_conf, iou=0.7,
                  classes=list(CFG.perception.classes), max_det=300, device=device, verbose=False)
        if device != "cpu":
            kw.update(_fp16_kwarg())
        r = model.predict(small, **kw)[0]
        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy() * np.array([sx, sy, sx, sy], dtype=np.float32)
            det = trk.Detections(xyxy, boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy())
        else:
            det = trk.Detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0))
        if self._tracker is None:
            self._tracker = trk.make_tracker(CFG.perception, self.cfg.work_every_n_frames, self.fps)
        rows = trk.update(self._tracker, det)

        # Per-track history of (t, ground x, ground y, box height), last 1.5 s.
        live = set()
        for x1, y1, x2, y2, tid, _s, _c in rows:
            tid = int(tid)
            live.add(tid)
            hist = self._hist.setdefault(tid, [])
            hist.append((t_sec, (x1 + x2) / 2.0, y2, y2 - y1))
            while hist and t_sec - hist[0][0] > 1.5:
                hist.pop(0)
        for tid in [k for k, v in self._hist.items() if t_sec - v[-1][0] > 2.0]:
            del self._hist[tid]

        pos, vel, hts, person = [], [], [], []
        cls_of = {int(r_[4]): int(r_[6]) for r_ in rows}
        for tid in live:
            hist = self._hist[tid]
            t1, x1_, y1_, h1 = hist[-1]
            old = [e for e in hist if t1 - e[0] >= self.cfg.velocity_min_span_sec]
            if not old:
                continue
            t0_, x0, y0, _h0 = old[-1]
            dt = t1 - t0_
            pos.append((x1_, y1_))
            vel.append(((x1_ - x0) / dt, (y1_ - y0) / dt))
            hts.append(h1)
            person.append(cls_of.get(tid, -1) == PERSON)
        raw = pair_scores(np.asarray(pos, float).reshape(-1, 2), np.asarray(vel, float).reshape(-1, 2),
                          np.asarray(hts, float), np.asarray(person, bool), self.cfg)
        self._raw.append(raw)
        del self._raw[:-self.cfg.persist_samples]
        # Held for persist_samples consecutive samples: one jittery sample
        # never reaches an alarm on its own.
        return min(self._raw) if len(self._raw) >= self.cfg.persist_samples else 0.0
