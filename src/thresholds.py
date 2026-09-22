"""Every rule threshold, in one place.

=============================================================================
READ THIS BEFORE TRUSTING ANY NUMBER BELOW
=============================================================================
NOT ONE of these values is calibrated against real footage. Every one was
chosen against tests/fixtures/synthetic_scene.py, which is an invented scene
with invented perspective, so:

  * Every PIXEL threshold is meaningless until the real zones.json exists.
    Pixels-per-metre on an oblique CCTV view varies by an order of magnitude
    between the near and far edge of the frame, and the synthetic scene's
    linear widening is not a real projective mapping.
  * Every SPEED threshold (px/s) inherits that problem twice over, because it
    is a pixel distance divided by time.
  * Every DURATION threshold is the most transferable kind here, because
    seconds are seconds. The ones taken straight from the task definition
    (stopped_vehicle >= 10 s) are the only values with real authority.
  * Every ANGLE threshold is scale-free and should mostly survive, but depends
    on the lane direction vectors being authored correctly.

Each entry carries a `# CALIBRATION:` note saying what it is grounded in.
"SPEC" means the task PDF defines it. "GUESS" means it is a placeholder that
must be retuned the moment a real clip is labelled.

The retuning workflow once real labels exist: run the rules over the cached
tracks, compare to ground_truth.json with evaluate.py --per-video, and sweep
the handful of values flagged GUESS. Stage 2 runs off the cache in under a
second, so a sweep is cheap -- that is what the CP0 cache was for.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StoppedVehicleThresholds:
    """A vehicle stationary on the carriageway, not queued at a signal."""

    # CALIBRATION: SPEC. The task defines stopped_vehicle as ">= 10 s".
    min_duration_sec: float = 10.0

    # CALIBRATION: MEASURED against synthetic jitter, then set with margin.
    # This is a cheap PRE-FILTER, not the decision -- max_drift_px below is
    # what actually decides whether a vehicle is stopped.
    #
    # Why it cannot be tight: speed is a differenced position, so bbox jitter
    # is amplified by 1/dt. With velocity_window=5 and frame_stride=2 at 25 fps
    # the centred difference spans 0.32 s, and measurement gives an apparent
    # speed of roughly 5.6 * jitter_px for a PERFECTLY STATIONARY vehicle:
    #
    #     jitter 1 px -> mean  5.6 px/s (p95  11)
    #     jitter 3 px -> mean 16.7 px/s (p95  34)
    #     jitter 5 px -> mean 27.9 px/s (p95  56)
    #
    # Real CCTV boxes on a stationary vehicle move several pixels frame to
    # frame, so a threshold below ~30 rejects genuinely stopped vehicles. An
    # earlier 12.0 here tolerated barely 2 px of jitter and lost the class.
    # Raising it is safe precisely because displacement, not speed, decides.
    speed_px_s: float = 45.0

    # CALIBRATION: GUESS. A "stop" is keyed off spatial continuity, not
    # track_id, so an id switch mid-stop does not reset the clock. Two samples
    # belong to the same physical stop if they are within this radius. Must
    # exceed the bbox jitter amplitude comfortably.
    same_stop_radius_px: float = 60.0

    # CALIBRATION: GUESS. A stop survives this much missing data (detector
    # dropout or occlusion) before it is considered ended.
    max_gap_sec: float = 2.0

    # CALIBRATION: GUESS, and THE ACTUAL DECISION. A sample joins a stop only
    # if it is within this distance of where the stop began. Jitter is
    # zero-mean so it does not accumulate, but real motion does: a vehicle
    # creeping at 40 px/s clears 90 px in 2.3 s and breaks the cluster long
    # before the 10 s bar. That is why the speed gate above can be loose.
    max_drift_px: float = 90.0


@dataclass(frozen=True)
class WrongWayThresholds:
    """Sustained travel against the authored lane direction."""

    # CALIBRATION: GUESS, but angle thresholds are scale-free so this should
    # mostly survive. 120 deg leaves a 90 deg turn (a car crossing a lane while
    # turning) comfortably below the bar, which is the specific false positive
    # this must avoid.
    min_angle_deg: float = 120.0

    # CALIBRATION: GUESS. Below this speed the heading is dominated by box
    # noise -- and src.geometry.angle_between deliberately returns 180 deg for a
    # zero vector, so WITHOUT this gate every stationary vehicle reads as
    # driving the wrong way. This gate is load-bearing, not cosmetic.
    min_speed_px_s: float = 25.0

    # CALIBRATION: GUESS. Debounce: the divergence must persist this long.
    min_duration_sec: float = 2.0

    # CALIBRATION: GUESS. ...and the vehicle must actually travel this far
    # against the lane. Duration alone is not enough: a slow vehicle nosing
    # sideways in a queue can hold a bad heading for seconds without going
    # anywhere. Pixel value, so this is one of the first to retune.
    min_distance_px: float = 120.0

    # CALIBRATION: GUESS. Brief gaps in the flag (a frame where the heading
    # dips below the angle bar, or the ground point leaves the lane polygon)
    # are bridged rather than splitting one event into two.
    debounce_gap_sec: float = 1.0


@dataclass(frozen=True)
class JaywalkingThresholds:
    """A pedestrian on the carriageway outside a crossing."""

    # CALIBRATION: GUESS. Debounce against a pedestrian's ground point
    # flickering over the kerb line, and against a single bad detection.
    min_duration_sec: float = 1.0

    # CALIBRATION: GUESS. Bridge short gaps so one crossing of the road is one
    # event rather than three.
    debounce_gap_sec: float = 1.0

    # CALIBRATION: GUESS. Person detections at CCTV range are small and
    # low-confidence; too high a floor here loses the class entirely, too low
    # and street furniture becomes a pedestrian. Needs a real clip.
    min_confidence: float = 0.35


@dataclass(frozen=True)
class CongestionThresholds:
    """Standstill or crawling traffic across all lanes of one direction."""

    # CALIBRATION: GUESS. "Crawling" in pixels per second. Higher than the
    # stopped_vehicle bar because congestion includes slow movement, not just
    # standstill.
    crawl_speed_px_s: float = 45.0

    # CALIBRATION: GUESS. A lane needs at least this many vehicles present
    # (outside the queue zone) before it can be judged congested at all.
    # Without it, a single slow car in an empty lane reads as a jam.
    min_vehicles_per_lane: int = 2

    # CALIBRATION: GUESS. Fraction of the vehicles in a lane that must be slow
    # for that lane to count as congested.
    slow_fraction: float = 0.7

    # CALIBRATION: GUESS. Sustained duration. A red phase can legitimately last
    # 60-90 s, so duration ALONE cannot separate a queue from a jam -- the
    # spatial test (ignoring vehicles inside signal_queue_zones) is what does
    # that. This is a secondary guard.
    min_duration_sec: float = 15.0

    # CALIBRATION: GUESS. Bridge gaps where a lane momentarily drops below the
    # vehicle count, e.g. between detector frames.
    debounce_gap_sec: float = 3.0

    # CALIBRATION: GUESS. Lanes are grouped into "directions" by clustering
    # their authored direction vectors; two lanes within this angle are the
    # same direction. Wide enough to group lanes that fan out with perspective,
    # narrow enough to keep opposing directions apart.
    direction_group_tolerance_deg: float = 45.0


@dataclass(frozen=True)
class SignalThresholds:
    """HSV classification of a traffic-light ROI crop."""

    # CALIBRATION: GUESS. OpenCV hue is 0-179. Red wraps, hence two bands.
    red_hue_lo: tuple[int, int] = (0, 10)
    red_hue_hi: tuple[int, int] = (170, 179)
    amber_hue: tuple[int, int] = (11, 27)
    green_hue: tuple[int, int] = (40, 95)

    # CALIBRATION: GUESS. A pixel must be this saturated and bright to count as
    # lit at all. The single most footage-dependent pair here: a washed-out
    # daytime lamp has low saturation, a night-time one is tiny but vivid.
    min_saturation: int = 90
    min_value: int = 110

    # CALIBRATION: GUESS. Below this many lit pixels the crop is treated as
    # unknown rather than guessed -- an unlit or badly-cropped ROI must not
    # produce a confident answer.
    min_lit_pixels: int = 12

    # CALIBRATION: GUESS. The winner must hold this share of the lit pixels...
    min_winner_fraction: float = 0.5

    # CALIBRATION: GUESS. ...and beat the runner-up by this ratio. Both exist
    # so an amber/red ambiguity resolves to unknown instead of a coin flip.
    min_winner_ratio: float = 1.6


@dataclass(frozen=True)
class Thresholds:
    stopped_vehicle: StoppedVehicleThresholds = field(
        default_factory=StoppedVehicleThresholds)
    wrong_way: WrongWayThresholds = field(default_factory=WrongWayThresholds)
    jaywalking: JaywalkingThresholds = field(default_factory=JaywalkingThresholds)
    congestion: CongestionThresholds = field(default_factory=CongestionThresholds)
    signal: SignalThresholds = field(default_factory=SignalThresholds)


#: The single instance every rule reads. Override per call in tests.
TH = Thresholds()
