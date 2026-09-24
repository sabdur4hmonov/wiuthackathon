"""Every rule threshold, in one place.

=============================================================================
READ THIS BEFORE TRUSTING ANY NUMBER BELOW
=============================================================================
There are still NO labels. As of CP3 (2026-09-24) the values were checked
against real tracked footage from the four sample clips -- what fires, and
what it looks like in the frame -- but never scored against ground truth. So:

  * Distances and speeds are in BOX HEIGHTS, not pixels (see UNITS below): on
    this oblique 4K camera a car is ~80 px tall at the far end and ~350 px near
    the camera, so no fixed pixel bar means the same thing across the frame.
  * Every DURATION threshold transfers best, because seconds are seconds. The
    ones taken from the task definition (stopped_vehicle >= 10 s) are the only
    values with real authority.
  * Every ANGLE threshold is scale-free, but depends on the lane arrows.
  * With no labels, every GUESS is biased toward NOT firing: under macro F1 a
    wrong class costs as much as a missed one, and a quiet rule is the safer
    failure.

Each entry carries a `# CALIBRATION:` note saying what it is grounded in.
"SPEC" means the task PDF defines it. "MEASURED" means read off the real
footage. "GUESS" means a placeholder to retune the moment a clip is labelled.

The retuning workflow once real labels exist: run the rules over the cached
tracks, compare to ground_truth.json with evaluate.py --per-video, and sweep
the handful of values flagged GUESS. Stage 2 runs off the cache in under a
second, so a sweep is cheap -- that is what the CP0 cache was for.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# UNITS. Distances and speeds below are in BOX HEIGHTS ("L"), the object's own
# bounding-box height at that moment, not pixels. The real camera is 4K and
# strongly oblique: a car is ~80 px tall at the far end of the avenue and
# ~350 px near the camera, so any fixed pixel bar is several times too strict
# at one end of the frame and several times too loose at the other. Box jitter
# scales with box size too, which is what makes L the right yardstick for
# "did it really move". Speeds are L per second.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StoppedVehicleThresholds:
    """A vehicle stationary on the carriageway, not queued at a signal."""

    # CALIBRATION: SPEC. The task defines stopped_vehicle as ">= 10 s".
    min_duration_sec: float = 10.0

    # CALIBRATION: GUESS, biased quiet. Weak detections of a static object are
    # where phantom "vehicles" come from.
    min_confidence: float = 0.40

    # CALIBRATION: GUESS. A loose PRE-FILTER only, on net movement over 2 s
    # (rules.util.sustained_speed). max_drift_L is the real decision.
    speed_L_s: float = 0.6

    # CALIBRATION: GUESS. The clock is keyed on spatial continuity, not
    # track_id, so an id switch mid-stop does not reset it. Samples within this
    # radius of a stop's centre belong to that stop.
    same_stop_radius_L: float = 0.5

    # CALIBRATION: GUESS. A stop survives this much missing data (dropout or
    # occlusion) before it is considered ended.
    max_gap_sec: float = 2.0

    # CALIBRATION: GUESS, and THE ACTUAL DECISION. A sample joins a stop only
    # if it is within this distance of where the stop began. Jitter does not
    # accumulate; genuine creep does, and breaks the stop early.
    max_drift_L: float = 0.6

    # CALIBRATION: GUESS, biased quiet. The stop must be backed by detections
    # for at least this share of its span, so a "stop" stitched together from
    # a handful of scattered hits does not count.
    min_coverage: float = 0.6

    # CALIBRATION: GUESS, biased quiet. A stationary sample this close to a
    # signal queue zone is treated as queued: a vehicle at the back of a queue
    # jitters across the zone edge, and the part outside must not add up to a
    # stop.
    queue_margin_L: float = 0.5

    # CALIBRATION: GUESS, biased quiet. Some id in the stop must be seen at
    # least this far from where it stopped -- i.e. it arrived or left during
    # the clip. A detection that never moves (a mis-detected object, or a car
    # parked all clip whose ground point leaks over the kerb line) never fires.
    motion_evidence_L: float = 1.5

    # CALIBRATION: GUESS, biased quiet. A stop is PART OF A QUEUE, not a stopped
    # vehicle, when another stationary vehicle sits within queue_neighbour_L of
    # it for at least queue_share of the stop. The first real run fired on a
    # taxi in a line of five stopped cars downstream of the intersection.
    queue_neighbour_L: float = 1.5
    queue_share: float = 0.5

    # CALIBRATION: GUESS, biased quiet. The stop must sit this far inside a real
    # kerb edge: a car pulled up on the kerb line at the corner plaza is
    # dropping off, not stopped in the road. A car double-parked in a lane is
    # well clear of it.
    kerb_margin_L: float = 0.3

    # CALIBRATION: MEASURED, biased quiet. Boxes smaller than this share of the
    # frame height are ignored. On the four real clips the only stops left at
    # this point were distant cars 67-80 px tall (4K) at the far end of the
    # avenue, half behind a lamp post, one switching track id 7 times in 21 s:
    # their ground points are not reliable enough to accuse anyone.
    min_box_frac: float = 0.04


@dataclass(frozen=True)
class WrongWayThresholds:
    """Sustained travel against the authored lane direction."""

    # CALIBRATION: GUESS, biased quiet (was 120). Scale-free. A legal turn
    # crosses its entry lane at ~90 deg; 135 keeps every turn well clear.
    min_angle_deg: float = 135.0

    # CALIBRATION: GUESS. Below this the heading is box noise -- and
    # geometry.angle_between returns 180 deg for a zero vector, so without
    # this gate every stationary vehicle reads as driving the wrong way.
    min_speed_L_s: float = 0.5

    # CALIBRATION: GUESS, biased quiet (was 2 s). The divergence must persist.
    min_duration_sec: float = 2.5

    # CALIBRATION: GUESS, biased quiet. Net progress AGAINST the lane over the
    # run, measured along the lane's own axis. Straight-line distance alone
    # would count a vehicle nosing sideways in a queue.
    min_progress_L: float = 2.0

    # CALIBRATION: GUESS. Bridge brief gaps in the flag so one manoeuvre is
    # one event.
    debounce_gap_sec: float = 1.0

    # CALIBRATION: GUESS, biased quiet.
    min_confidence: float = 0.40


@dataclass(frozen=True)
class JaywalkingThresholds:
    """A pedestrian on the carriageway outside a crossing."""

    # CALIBRATION: GUESS, biased quiet (was 1 s).
    min_duration_sec: float = 2.0

    # CALIBRATION: GUESS. Bridge short gaps so one crossing is one event.
    debounce_gap_sec: float = 1.0

    # CALIBRATION: GUESS, biased quiet (was 0.35).
    min_confidence: float = 0.45

    # CALIBRATION: biased quiet. The ground point must be at least kerb_margin_L
    # inside the carriageway (from a real kerb edge) and crossing_margin_L away
    # from every crossing polygon. Someone waiting on the kerb, or walking along
    # the edge of the zebra, is not jaywalking at this bar.
    #
    # KNOWN FALSE POSITIVE, ACCEPTED: people who walk parallel to a zebra but
    # well outside its stripes -- common on the diagonal X_leg crossing -- are
    # still flagged. Whether a labeller calls that jaywalking is unknown.
    # MEASURED 2026-09-24 (sample_001+002): at 0.5 L, 14 % of the surviving
    # candidate rows were people at the bus-stop kerb and boarding buses.
    kerb_margin_L: float = 1.0
    crossing_margin_L: float = 0.5

    # CALIBRATION: GUESS. A "person" box with at least this share of its area
    # inside a vehicle box in the same frame is a rider or an occupant, not a
    # pedestrian. Motorcyclists and cyclists are the main source of person
    # detections on the carriageway.
    rider_overlap: float = 0.5


@dataclass(frozen=True)
class CongestionThresholds:
    """Standstill or crawling traffic across one DIRECTION of travel as a whole."""

    # CALIBRATION: GUESS. "Crawling", measured as net movement over 2 s
    # (rules.util.sustained_speed) so box jitter does not read as motion.
    crawl_speed_L_s: float = 0.35

    # CALIBRATION: GUESS, biased quiet. Vehicles of the direction that must be
    # present OUTSIDE every queue zone before it can be judged congested.
    min_vehicles: int = 4

    # CALIBRATION: GUESS, biased quiet. Share of those vehicles that must crawl.
    # At 0.8, one lane crawling beside a lane still flowing at ~1 car per frame
    # sits exactly on the bar; a direction that is still moving must not fire.
    slow_fraction: float = 0.85

    # CALIBRATION: GUESS, biased quiet (was 15 s). A red phase can run 60-90 s,
    # so duration alone never separates a queue from a jam -- excluding every
    # vehicle inside a signal_queue_zone does that. This is a second guard.
    min_duration_sec: float = 30.0

    # CALIBRATION: GUESS. Vehicle count and slow share are judged over this
    # sliding window, so steady flow past a crawling lane always shows up.
    window_sec: float = 5.0

    # CALIBRATION: GUESS. Bridge brief dips below the vehicle count.
    debounce_gap_sec: float = 3.0

    # CALIBRATION: GUESS. Only used when zones.json names no direction_group:
    # lanes whose arrows agree within this angle are then one direction.
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
