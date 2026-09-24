"""Wall-clock budget manager.

The harness gives ONE budget for Part A and Part B together::

    duration = n_frames / fps           # run_submission.video_meta
    budget   = 3.0 * duration
    deadline = t0 + budget              # t0 set just before detect_events()

If the total is exceeded -- or if Part A alone blows past the deadline -- the
harness replaces the whole entry with {"events": [], "risk": []}. A slow
Part B therefore does not cost Part B, it costs Part A as well. Everything in
this module exists to make that outcome impossible.

The policy (CP4):

1. The keyframe BASELINE pass always runs to completion. Part B is the default
   RiskEstimator and scores the same whether or not the video is wiped, so a
   Part A starved to zero and a wiped video both score nothing: running the
   baseline is never worse. It costs ~0.1-0.3x realtime on the 4K clips.
2. Part B's measured decode (median of three probes, measure_part_b_floor)
   decides only whether there is room for OPTIONAL density beyond the
   baseline (headroom_x). Lighter clips measure lighter and buy it
   automatically; the 4K test clips on an 8-core CPU box do not.
3. The cv2 fallback decoder (no PyAV) still stops at part_a_hard: it decodes
   every frame and cannot be trusted to finish.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import CFG, BudgetConfig


PROBE_POSITIONS = (0.1, 0.5, 0.85)


def measure_part_b_floor(video_path: str | Path, n_frames: int,
                         n_probe: int = 40, max_probe_sec: float = 3.0,
                         positions: tuple[float, ...] = PROBE_POSITIONS,
                         rates_out: list | None = None,
                         ) -> tuple[float | None, float]:
    """Probe the REAL, unavoidable cost of run_submission.py's Part B decode.

    run_risk (unmodified, official stride=1) does exactly one thing per frame:
    ``cap.read()`` on the original file. That call is outside our control --
    nothing in solution.py can make it faster -- so the only honest way to
    budget for it is to time a sample of it, on the real file, on this machine,
    right now. It is NOT ``cap.grab()``/``cap.retrieve()``: that pair is Stage
    1's own optimisation (see run_perception) and understates run_risk's real
    cost, which always pays the full decode + BGR conversion + copy.

    THREE short probes, at different positions in the file, and the MEDIAN
    rate: decode cost is not flat across a clip (a 100-frame probe from frame 0
    and a 1500-frame one differed by 21% on sample_001), and a single probe
    lets one unlucky stretch -- or one hiccup on a shared machine -- decide the
    whole budget. Each probe seeks, reads one frame untimed (the seek decodes
    from the previous keyframe, which run_risk never pays), then times plain
    reads.

    Bounded so the probe itself cannot eat the budget it is trying to protect:
    ``n_probe`` frames in total, and ``max_probe_sec`` of wall clock in total,
    split across the positions. On a very short or corrupt clip this can read
    fewer frames than asked, including zero.

    Returns (estimated_total_part_b_sec, probe_wall_sec). The estimate is None
    when nothing could be read at all -- callers should fall back to the fixed
    BudgetConfig.part_b_reserve multiplier in that case, not treat None as zero.
    Per-probe seconds-per-frame are appended to ``rates_out`` when given.
    """
    import cv2

    t0 = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, time.perf_counter() - t0
    rates: list[float] = []
    positions = tuple(positions) or (0.0,)
    per_probe = max(1, int(n_probe) // len(positions))
    per_sec = max_probe_sec / len(positions)
    try:
        # n_probe <= 0 still reads one frame at each position.
        for pos in positions:
            start = int(max(0, min(n_frames - per_probe - 1, pos * n_frames)))
            if start > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                if not cap.read()[0]:            # seek warm-up, untimed
                    continue
            t = time.perf_counter()
            read = 0
            for _ in range(per_probe):
                ok, _frame = cap.read()          # exactly run_risk's call, not grab()
                if not ok:
                    break
                read += 1
                if time.perf_counter() - t >= per_sec:
                    break
            if read:
                rates.append((time.perf_counter() - t) / read)
            if time.perf_counter() - t0 >= max_probe_sec:
                break
    finally:
        cap.release()

    probe_wall = time.perf_counter() - t0
    if rates_out is not None:
        rates_out.extend(rates)
    if not rates:
        return None, probe_wall
    return float(np.median(rates)) * max(0, n_frames), probe_wall


def probe_duration(video_path: str | Path) -> tuple[float, float, int]:
    """Return (duration_sec, fps, n_frames) exactly as the harness derives them.

    The harness uses n_frames / fps from OpenCV's properties, NOT the container
    duration. The two disagree often enough (VFR sources, broken headers,
    trimmed files) that using the wrong one would silently shift the whole
    budget. We reproduce the harness formula, including its fps fallback.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0          # harness: `or 25.0`
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = n_frames / float(fps) if fps else 0.0
    return duration, float(fps), n_frames


@dataclass
class Stage:
    name: str
    seconds: float
    note: str = ""


@dataclass
class Budget:
    """Tracks elapsed wall-clock against the harness deadline."""

    duration: float
    fps: float
    n_frames: int
    t0: float
    cfg: BudgetConfig = field(default_factory=lambda: CFG.budget)
    stages: list[Stage] = field(default_factory=list)
    _stop_reason: str | None = None
    _measured_part_b_sec: float | None = None

    # -- construction -------------------------------------------------------
    @classmethod
    def for_video(
        cls,
        video_path: str | Path,
        t0: float | None = None,
        cfg: BudgetConfig | None = None,
    ) -> "Budget":
        """Build from a video. Call this FIRST in detect_events().

        t0 defaults to now. The harness sets its own t0 immediately before
        calling us, so ours trails it by a function call; safety_margin_sec
        absorbs that and any interpreter hiccup.
        """
        dur, fps, n = probe_duration(video_path)
        return cls(duration=dur, fps=fps, n_frames=n,
                   t0=t0 if t0 is not None else time.perf_counter(),
                   cfg=cfg or CFG.budget)

    # -- derived limits -----------------------------------------------------
    @property
    def total_budget(self) -> float:
        """What the harness allows for Part A + Part B."""
        return self.cfg.harness_factor * self.duration

    @property
    def part_b_reserve(self) -> float:
        """Wall-clock Part B's mandatory full-decode pass will need.

        Prefers a REAL measurement (set_measured_part_b) taken from this file
        on this machine over the fixed multiplier: run_risk's decode cost is
        outside our control and a guess about it can be wrong in either
        direction. Falls back to the fixed cfg.part_b_reserve multiplier only
        when no measurement was taken (measure_part_b_floor was never called,
        or it could not read a single frame).
        """
        if self._measured_part_b_sec is not None:
            return self._measured_part_b_sec * self.cfg.part_b_measured_safety
        return self.cfg.part_b_reserve * self.duration

    def set_measured_part_b(self, seconds: float) -> None:
        """Record a real Part B cost estimate from measure_part_b_floor().

        Call once, early in detect_events, before part_a_hard is consulted:
        part_a_hard derives from part_b_reserve, so this must land before any
        stop decision is made or it has no effect on that run.
        """
        self._measured_part_b_sec = max(0.0, float(seconds))

    @property
    def part_a_target(self) -> float:
        """Soft target: past this we start shedding optional work."""
        return self.cfg.part_a_target * self.duration

    @property
    def safety_margin(self) -> float:
        """Slack for harness overhead outside our t0 and check granularity.

        Capped at a fraction of the total budget. A flat 2 s is right for the
        multi-minute clips the test set is made of, but on a 1 s clip it exceeds
        the entire Part A allowance and would stop perception before it started.
        """
        return min(self.cfg.safety_margin_sec,
                   self.cfg.safety_margin_frac * self.total_budget)

    @property
    def part_a_hard(self) -> float:
        """Absolute stop for Part A.

        Whichever is tighter: the configured 1.5x, or whatever is left after
        reserving Part B's pass and the safety margin.
        """
        by_factor = self.cfg.part_a_hard * self.duration
        by_reserve = self.total_budget - self.part_b_reserve - self.safety_margin
        return max(0.0, min(by_factor, by_reserve))

    # -- live state ---------------------------------------------------------
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def remaining_to_hard(self) -> float:
        return self.part_a_hard - self.elapsed()

    def headroom_x(self) -> float | None:
        """Time left for OPTIONAL Part A work, as a multiple of the video
        duration: the total budget minus what Part B's measured decode needs
        (with its safety factor), the safety margin and what has been used.

        None without a real Part B measurement: optional work is never bought
        on the strength of the fixed fallback guess.
        """
        if self._measured_part_b_sec is None or not self.duration:
            return None
        left = self.total_budget - self.part_b_reserve - self.safety_margin - self.elapsed()
        return left / self.duration

    def over_target(self) -> bool:
        return self.elapsed() > self.part_a_target

    def should_stop(self) -> bool:
        """True once Part A must stop and emit what it has.

        Cheap enough to call in the per-frame loop, but callers should still
        check it every few frames rather than every frame.
        """
        if self._stop_reason is not None:
            return True
        if self.elapsed() >= self.part_a_hard:
            self._stop_reason = (
                f"hard stop at {self.elapsed():.1f}s "
                f"(limit {self.part_a_hard:.1f}s of {self.total_budget:.1f}s total)"
            )
            return True
        return False

    def project_overrun(self, done: int, total: int,
                        loop_t0: float | None = None) -> bool:
        """Extrapolate: at the current rate, will finishing blow the hard stop?

        Lets us bail at frame 200 of 4000 instead of discovering the problem at
        frame 3900, when there is no time left to emit anything.

        ``loop_t0`` is when the per-frame loop began. It matters: Part A pays a
        large one-off cost before the first frame (loading the detector, warming
        CUDA), and amortising that across the frames processed so far makes the
        projection wildly pessimistic early on -- pessimistic enough to abort a
        run that would have finished in a fifth of its budget. With loop_t0 the
        fixed cost is counted ONCE, in elapsed(), and only the marginal
        per-frame cost is extrapolated over the frames that remain.
        """
        if done <= 0 or total <= 0 or done >= total:
            return False
        now = time.perf_counter()
        loop_elapsed = (now - loop_t0) if loop_t0 is not None else (now - self.t0)
        if loop_elapsed <= 0:
            return False
        per_frame = loop_elapsed / done
        projected = self.elapsed() + (total - done) * per_frame
        return projected > self.part_a_hard

    def force_stop(self, reason: str) -> None:
        self._stop_reason = reason

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    # -- instrumentation ----------------------------------------------------
    @contextmanager
    def stage(self, name: str):
        """Time a named stage into the breakdown."""
        t = time.perf_counter()
        try:
            yield
        finally:
            self.stages.append(Stage(name, time.perf_counter() - t))

    def note(self, name: str, seconds: float, note: str = "") -> None:
        self.stages.append(Stage(name, seconds, note))

    def report(self) -> dict:
        el = self.elapsed()
        return {
            "duration_sec": round(self.duration, 2),
            "fps": round(self.fps, 3),
            "n_frames": self.n_frames,
            "total_budget_sec": round(self.total_budget, 1),
            "part_a_target_sec": round(self.part_a_target, 1),
            "part_a_hard_sec": round(self.part_a_hard, 1),
            "part_b_reserve_sec": round(self.part_b_reserve, 1),
            "part_b_reserve_measured": self._measured_part_b_sec is not None,
            "part_a_elapsed_sec": round(el, 2),
            "part_a_x_realtime": round(el / self.duration, 3) if self.duration else None,
            "stopped_early": self._stop_reason is not None,
            "stop_reason": self._stop_reason,
            "stages": [{"name": s.name, "sec": round(s.seconds, 3), "note": s.note}
                       for s in self.stages],
        }

    def format_report(self) -> str:
        r = self.report()
        head = (f"[budget] {r['duration_sec']}s video @ {r['fps']}fps | "
                f"A used {r['part_a_elapsed_sec']}s "
                f"({r['part_a_x_realtime']}x realtime) | "
                f"A-hard {r['part_a_hard_sec']}s | total {r['total_budget_sec']}s")
        lines = [head]
        for s in r["stages"]:
            lines.append(f"[budget]   {s['name']:<22} {s['sec']:>8.3f}s {s['note']}")
        if r["stop_reason"]:
            lines.append(f"[budget]   ! {r['stop_reason']}")
        return "\n".join(lines)
