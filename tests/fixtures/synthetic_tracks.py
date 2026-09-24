"""Build TrackTables by hand, in the exact CP0 cache schema.

Rules consume TRACKS + ZONES, never video. This module is the TRACKS half:
a composable builder for trajectories, plus the tracker failure modes that CP0
established ByteTrack actually has.

    b = TrackBuilder(fps=25.0, duration=60.0)
    (b.vehicle()
      .start_at(point_in_lane("ns_nb_outer", 0.0), t=0.0)
      .drive_to(point_in_lane("ns_nb_outer", 0.4), speed=220)
      .stop_for(14.0)
      .drive_to(point_in_lane("ns_nb_outer", 1.0), speed=220))
    tracks = b.build()

Motion model: an actor accumulates (t, x, y) keyframes and positions are
linearly interpolated between them at each processed frame time. "Stopping" is
two keyframes at the same place; a turn is an arc of keyframes. Everything is
expressed in GROUND points (bottom-centre of the box), because that is what
every rule reads.

DERIVED COLUMNS ARE NOT FAKED. After the rows are assembled, the real
src.tracks.compute_kinematics fills gx, gy, vx, vy, speed and heading with the
same centred-difference smoothing the live pipeline uses. A rule tested here
sees velocities produced the same way it will in production, including the
smoothing lag at the ends of a track.

=============================================================================
TRACKER FAILURE MODES
=============================================================================
ByteTrack is motion-only, no appearance model (see README). Every rule must be
tested BOTH clean and degraded, because a rule that only works on clean tracks
will not survive real footage:

  .with_id_switch(t)        same physical object, new track_id from t onwards.
                            Happens when two boxes cross with high IoU.
  .with_dropout(t0, t1)     detections missing for a window, id preserved
                            after. Models a short occlusion inside track_buffer.
  .with_occlusion(t0, t1)   detections missing AND a new id afterwards. Models
                            an occlusion longer than track_buffer, which is the
                            realistic outcome past ~2.4 s at stride 2.
  .with_jitter(px)          gaussian noise on the ground point: detector box
                            wobble.
  .with_heading_noise(deg)  lateral wobble sized so the induced heading error
                            has roughly this standard deviation. Heading is
                            DERIVED from position by compute_kinematics, so it
                            cannot be perturbed directly without making the
                            fixture inconsistent with the real pipeline.

=============================================================================
WHAT THIS DOES NOT MODEL
=============================================================================
  1. No false-positive detections: every row here belongs to a real object.
     Real footage has ghost boxes on shadows, signage and reflections.
  2. No class confusion: a car is always cls=2. Real detectors flip car/truck
     on the same object between frames.
  3. No two objects merging into one box, which is what actually happens in
     dense traffic and is the main source of congestion-rule error.
  4. Constant speed between keyframes. No acceleration profile, so braking is
     a step change, not a ramp.
  5. Perspective box scaling is a crude linear function of y (see _box_scale).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.config import CFG
from src.tracks import COL, COLUMNS, TrackTable, compute_kinematics

Point = tuple[float, float]

# Box size at the near edge of the frame, in pixels, per COCO class.
_BASE_BOX = {
    0: (46.0, 150.0),    # person
    1: (60.0, 120.0),    # bicycle
    2: (170.0, 130.0),   # car
    3: (70.0, 130.0),    # motorcycle
    5: (230.0, 260.0),   # bus
    7: (220.0, 220.0),   # truck
}


def _box_scale(y: float, height: float = 1080.0) -> float:
    """Crude perspective: objects at the top of the frame are smaller.

    Linear in y, clamped. A real camera's scaling is projective, not linear --
    this only needs to be monotonic and roughly right so that far-field boxes
    are small enough for dropout tests to be meaningful.
    """
    t = (y - 0.28 * height) / (height - 0.28 * height)
    return float(np.clip(0.22 + 0.78 * t, 0.12, 1.15))


@dataclass
class _Keyframe:
    t: float
    x: float
    y: float


@dataclass
class Actor:
    """One physical road user. Chainable; every method returns self."""

    builder: "TrackBuilder"
    track_id: int
    cls: int = 2
    conf: float = 0.88
    keys: list[_Keyframe] = field(default_factory=list)

    # degradations
    _id_switch_at: list[float] = field(default_factory=list)
    _dropouts: list[tuple[float, float, bool]] = field(default_factory=list)
    _jitter_px: float = 0.0
    _heading_noise_deg: float = 0.0

    # -- trajectory -------------------------------------------------------
    @property
    def t_now(self) -> float:
        return self.keys[-1].t if self.keys else 0.0

    @property
    def pos_now(self) -> Point:
        if not self.keys:
            raise ValueError("call start_at() first")
        return (self.keys[-1].x, self.keys[-1].y)

    def start_at(self, pt: Point, t: float = 0.0) -> "Actor":
        self.keys.append(_Keyframe(float(t), float(pt[0]), float(pt[1])))
        return self

    def drive_to(self, pt: Point, speed: float | None = None,
                 duration: float | None = None) -> "Actor":
        """Travel in a straight line. Give either speed (px/s) or duration (s)."""
        if not self.keys:
            raise ValueError("call start_at() first")
        x0, y0 = self.pos_now
        dist = math.hypot(pt[0] - x0, pt[1] - y0)
        if duration is None:
            if speed is None or speed <= 0:
                raise ValueError("give speed > 0 or duration")
            duration = dist / speed
        self.keys.append(_Keyframe(self.t_now + float(duration),
                                   float(pt[0]), float(pt[1])))
        return self

    def stop_for(self, seconds: float) -> "Actor":
        """Hold position. Two keyframes at the same place."""
        x, y = self.pos_now
        self.keys.append(_Keyframe(self.t_now + float(seconds), x, y))
        return self

    def wait_until(self, t: float) -> "Actor":
        if t < self.t_now:
            raise ValueError(f"cannot wait until {t}, already at {self.t_now}")
        x, y = self.pos_now
        self.keys.append(_Keyframe(float(t), x, y))
        return self

    def turn_to(self, pt: Point, via: Point, duration: float,
                steps: int = 12) -> "Actor":
        """Curve to `pt` through a quadratic Bezier with control point `via`.

        Used for legal turns, which wrong_way must NOT fire on: the heading
        sweeps through 90 degrees relative to the entry lane while the vehicle
        is inside it.
        """
        if not self.keys:
            raise ValueError("call start_at() first")
        x0, y0 = self.pos_now
        t0 = self.t_now
        for i in range(1, steps + 1):
            s = i / steps
            mx = (1 - s) ** 2 * x0 + 2 * (1 - s) * s * via[0] + s ** 2 * pt[0]
            my = (1 - s) ** 2 * y0 + 2 * (1 - s) * s * via[1] + s ** 2 * pt[1]
            self.keys.append(_Keyframe(t0 + duration * s, mx, my))
        return self

    def reverse_heading(self, distance: float, duration: float) -> "Actor":
        """Back up along the last direction of travel -- a wrong-way manoeuvre."""
        if len(self.keys) < 2:
            raise ValueError("need a direction of travel to reverse")
        a, b = self.keys[-2], self.keys[-1]
        dx, dy = b.x - a.x, b.y - a.y
        n = math.hypot(dx, dy) or 1.0
        return self.drive_to((b.x - dx / n * distance, b.y - dy / n * distance),
                             duration=duration)

    # -- degradations -----------------------------------------------------
    def with_id_switch(self, at_t: float) -> "Actor":
        """Same object, new track_id from at_t onwards. No gap in detections."""
        self._id_switch_at.append(float(at_t))
        return self

    def with_dropout(self, t0: float, t1: float) -> "Actor":
        """Detections missing in [t0, t1); the id survives the gap."""
        self._dropouts.append((float(t0), float(t1), False))
        return self

    def with_occlusion(self, t0: float, t1: float) -> "Actor":
        """Detections missing in [t0, t1) AND a new id afterwards.

        The realistic outcome when an occlusion outlasts ByteTrack's
        track_buffer, which at stride 2 on 25 fps is about 2.4 s.
        """
        self._dropouts.append((float(t0), float(t1), True))
        return self

    def with_jitter(self, px: float) -> "Actor":
        self._jitter_px = float(px)
        return self

    def with_heading_noise(self, deg: float) -> "Actor":
        self._heading_noise_deg = float(deg)
        return self

    # -- sampling ---------------------------------------------------------
    def _position_at(self, t: float) -> Point | None:
        if len(self.keys) < 2:
            if self.keys and abs(t - self.keys[0].t) < 1e-9:
                return (self.keys[0].x, self.keys[0].y)
            return None
        if t < self.keys[0].t - 1e-9 or t > self.keys[-1].t + 1e-9:
            return None
        for a, b in zip(self.keys, self.keys[1:]):
            if a.t - 1e-9 <= t <= b.t + 1e-9:
                span = b.t - a.t
                s = 0.0 if span <= 1e-9 else (t - a.t) / span
                return (a.x + (b.x - a.x) * s, a.y + (b.y - a.y) * s)
        return (self.keys[-1].x, self.keys[-1].y)


@dataclass
class TrackBuilder:
    """Assembles Actors into a TrackTable in the CP0 schema."""

    fps: float = 25.0
    duration: float = 60.0
    width: int = 1920
    height: int = 1080
    frame_stride: int = 2
    seed: int = 20260923

    actors: list[Actor] = field(default_factory=list)
    _next_id: int = 1

    # -- actors -----------------------------------------------------------
    def vehicle(self, cls: int = 2, track_id: int | None = None,
                conf: float = 0.88) -> Actor:
        return self._actor(cls, track_id, conf)

    def pedestrian(self, track_id: int | None = None,
                   conf: float = 0.72) -> Actor:
        return self._actor(0, track_id, conf)

    def _actor(self, cls: int, track_id: int | None, conf: float) -> Actor:
        if track_id is None:
            track_id = self._next_id
            self._next_id = max(self._next_id + 1, track_id + 1)
        else:
            self._next_id = max(self._next_id, int(track_id) + 1)
        a = Actor(builder=self, track_id=int(track_id), cls=int(cls), conf=conf)
        self.actors.append(a)
        return a

    def degrade_all(self, jitter_px: float = 0.0,
                    heading_noise_deg: float = 0.0) -> "TrackBuilder":
        """Apply the same noise to every actor. The 'realistic' switch."""
        for a in self.actors:
            if jitter_px:
                a.with_jitter(jitter_px)
            if heading_noise_deg:
                a.with_heading_noise(heading_noise_deg)
        return self

    # -- build ------------------------------------------------------------
    @property
    def n_frames(self) -> int:
        return int(round(self.duration * self.fps))

    def _frame_grid(self) -> np.ndarray:
        """Processed frame indices: multiples of frame_stride, as perception emits."""
        return np.arange(0, self.n_frames, max(1, self.frame_stride), dtype=np.int64)

    def build(self) -> TrackTable:
        rng = np.random.default_rng(self.seed)
        grid = self._frame_grid()
        rows: list[np.ndarray] = []
        next_free_id = max([a.track_id for a in self.actors], default=0) + 1000

        for actor in self.actors:
            if len(actor.keys) < 1:
                continue
            # id switches partition the timeline into id segments
            switch_ts = sorted(actor._id_switch_at)
            # occlusions that reacquire also force a new id at their end
            for t0, t1, new_id in actor._dropouts:
                if new_id:
                    switch_ts.append(t1)
            switch_ts = sorted(set(switch_ts))
            id_for_segment = [actor.track_id]
            for _ in switch_ts:
                id_for_segment.append(next_free_id)
                next_free_id += 1

            for frame_idx in grid:
                t = float(frame_idx) / self.fps
                pos = actor._position_at(t)
                if pos is None:
                    continue
                if any(t0 <= t < t1 for t0, t1, _ in actor._dropouts):
                    continue

                x, y = pos
                if actor._heading_noise_deg > 0.0:
                    x, y = self._lateral_wobble(actor, t, x, y, rng)
                if actor._jitter_px > 0.0:
                    x += float(rng.normal(0.0, actor._jitter_px))
                    y += float(rng.normal(0.0, actor._jitter_px))

                seg = sum(1 for st in switch_ts if t >= st)
                tid = id_for_segment[seg]
                rows.append(self._row(frame_idx, t, tid, actor, x, y))

        data = (np.vstack(rows).astype(np.float32) if rows
                else np.zeros((0, len(COLUMNS)), dtype=np.float32))
        # The real derivation, not a hand-written one.
        data = compute_kinematics(
            data, CFG.perception.velocity_window_samples(self.fps, self.frame_stride))
        return TrackTable(
            data=data, fps=self.fps, duration=self.duration,
            n_frames=self.n_frames, width=self.width, height=self.height,
            frame_stride=self.frame_stride, complete=True,
            processed_until_sec=self.duration,
        )

    def _lateral_wobble(self, actor: Actor, t: float, x: float, y: float,
                        rng: np.random.Generator) -> Point:
        """Displace perpendicular to travel, sized to induce ~N(0, deg) heading.

        Over one sample step the vehicle moves `step` px; a lateral offset `d`
        tilts the apparent heading by atan(d / step). Inverting that gives the
        offset for a target angular error, so the noise scales with speed the
        way real bbox noise does: slow objects get noisier headings.
        """
        dt = max(1, self.frame_stride) / self.fps
        ahead = actor._position_at(t + dt)
        behind = actor._position_at(t - dt)
        ref = ahead or behind
        if ref is None:
            return (x, y)
        dx, dy = (ref[0] - x, ref[1] - y) if ahead else (x - ref[0], y - ref[1])
        step = math.hypot(dx, dy)
        if step < 1e-6:
            return (x, y)
        sigma = step * math.tan(math.radians(actor._heading_noise_deg))
        d = float(rng.normal(0.0, sigma))
        nx, ny = -dy / step, dx / step
        return (x + nx * d, y + ny * d)

    def _row(self, frame_idx: int, t: float, tid: int, actor: Actor,
             gx: float, gy: float) -> np.ndarray:
        bw, bh = _BASE_BOX.get(actor.cls, (140.0, 120.0))
        s = _box_scale(gy, float(self.height))
        w, h = bw * s, bh * s
        r = np.zeros(len(COLUMNS), dtype=np.float32)
        r[COL["frame_idx"]] = frame_idx
        r[COL["t_sec"]] = t
        r[COL["track_id"]] = tid
        r[COL["cls"]] = actor.cls
        r[COL["conf"]] = actor.conf
        r[COL["x1"]] = gx - w / 2.0
        r[COL["y1"]] = gy - h
        r[COL["x2"]] = gx + w / 2.0
        r[COL["y2"]] = gy
        # gx/gy/vx/vy/speed/heading are filled by compute_kinematics.
        return r


# ---------------------------------------------------------------------------
# convenience scenarios
# ---------------------------------------------------------------------------
def empty_tracks(fps: float = 25.0, duration: float = 60.0) -> TrackTable:
    return TrackBuilder(fps=fps, duration=duration).build()


def straight_drive(lane_id: str, speed: float = 220.0,
                   builder: TrackBuilder | None = None,
                   cls: int = 2, t0: float = 0.0) -> tuple[TrackBuilder, Actor]:
    """A vehicle driving lawfully along `lane_id` from entry to exit."""
    from .synthetic_scene import point_in_lane

    b = builder or TrackBuilder()
    a = (b.vehicle(cls=cls)
          .start_at(point_in_lane(lane_id, 0.0), t=t0)
          .drive_to(point_in_lane(lane_id, 1.0), speed=speed))
    return b, a
