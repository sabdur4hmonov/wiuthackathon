"""The four rule-based classes, each tested clean AND degraded.

A rule that only works on clean tracks will not survive real footage, so every
behavioural test has a degraded twin: id switches, dropouts, occlusions, bbox
jitter and heading noise, all produced by the real tracker failure modes that
CP0 established ByteTrack has.

Every rule also has a NEGATIVE test for the specific false positive it is
designed to avoid. Those matter more than the positive ones: under macro F1 a
class we emit that never occurs scores 0 for that class AND enlarges the
denominator, so a rule that over-fires is worse than one that under-fires.
"""
from __future__ import annotations

import numpy as np
import pytest

from fixtures.synthetic_scene import (JAYWALK_POINT, QUEUE_POINT, STOPPED_POINT,
                                      point_in_lane, scene_zones)
from fixtures.synthetic_tracks import TrackBuilder
from src.postprocess import assert_no_overlap, finalise
from src.rules import RULES, FrameSegment, run_rules
from src.rules import congestion as congestion_mod
from src.rules import jaywalking as jaywalking_mod
from src.rules import stopped_vehicle as stopped_mod
from src.rules import wrong_way as wrong_way_mod


@pytest.fixture(scope="module")
def zones(tmp_path_factory):
    return scene_zones(tmp_path_factory.mktemp("zones"))


def seconds(segs, tracks):
    """FrameSegments -> (start, end) seconds, the way run_rules converts them."""
    step = max(1, tracks.frame_stride)
    return [(s.start_frame / tracks.fps, (s.end_frame + step) / tracks.fps)
            for s in segs]


# ===========================================================================
# registry wiring
# ===========================================================================
def test_all_four_classes_are_registered():
    assert set(RULES) == {"congestion", "stopped_vehicle", "jaywalking", "wrong_way"}


def test_solution_classes_track_the_registry():
    import solution

    assert set(solution.CLASSES) == set(RULES)


def test_every_rule_emits_nothing_without_zones():
    """A guessed lane is worse than no lane."""
    b = TrackBuilder(duration=40.0)
    (b.vehicle().start_at(STOPPED_POINT, t=0.0).stop_for(30.0))
    tracks = b.build()
    for label, fn in RULES.items():
        assert fn(tracks, None) == [], f"{label} emitted without zones"
    assert run_rules(tracks, None) == []


def test_every_rule_survives_empty_tracks(zones):
    empty = TrackBuilder(duration=10.0).build()
    for label, fn in RULES.items():
        assert fn(empty, zones) == [], label


# ===========================================================================
# stopped_vehicle
# ===========================================================================
def _stopped_builder(stop_sec: float = 16.0, duration: float = 40.0):
    b = TrackBuilder(duration=duration)
    a = (b.vehicle()
          .start_at(point_in_lane("ns_nb_outer", 0.02), t=0.0)
          .drive_to(STOPPED_POINT, speed=220.0)
          .stop_for(stop_sec)
          .drive_to(point_in_lane("ns_nb_outer", 0.9), speed=220.0))
    return b, a


def test_stopped_vehicle_fires_on_a_long_stop(zones):
    b, _ = _stopped_builder(stop_sec=16.0)
    tracks = b.build()
    segs = stopped_mod.detect(tracks, zones)
    assert len(segs) == 1
    s, e = seconds(segs, tracks)[0]
    assert (e - s) == pytest.approx(16.0, abs=1.5)


def test_stopped_vehicle_ignores_a_short_stop(zones):
    """8 s is below the 10 s the task definition requires."""
    b, _ = _stopped_builder(stop_sec=8.0)
    assert stopped_mod.detect(b.build(), zones) == []


def test_stopped_vehicle_ignores_a_queue_at_the_signal(zones):
    """Sitting still in a signal queue zone is lawful, however long."""
    b = TrackBuilder(duration=60.0)
    (b.vehicle()
      .start_at(point_in_lane("ns_nb_outer", 0.02), t=0.0)
      .drive_to(QUEUE_POINT, speed=220.0)
      .stop_for(40.0)
      .drive_to(point_in_lane("ns_nb_outer", 0.9), speed=220.0))
    assert stopped_mod.detect(b.build(), zones) == []


def test_stopped_vehicle_ignores_a_vehicle_off_the_carriageway(zones):
    b = TrackBuilder(duration=40.0)
    b.vehicle().start_at((120.0, 200.0), t=0.0).stop_for(30.0)
    assert stopped_mod.detect(b.build(), zones) == []


def test_stopped_vehicle_ignores_moving_traffic(zones):
    b = TrackBuilder(duration=40.0)
    (b.vehicle().start_at(point_in_lane("ns_nb_outer", 0.0), t=0.0)
       .drive_to(point_in_lane("ns_nb_outer", 1.0), speed=180.0))
    assert stopped_mod.detect(b.build(), zones) == []


# -- the headline requirement ----------------------------------------------
def test_an_id_switch_mid_stop_does_not_reset_the_clock(zones):
    """THE reason the clock is keyed on position, not track_id.

    A 16 s stop interrupted by an id switch at 10 s would, under id-keyed
    timing, become two fragments of ~6 s and ~10 s -- neither reliably over the
    bar. Keyed spatially it stays one 16 s stop.
    """
    b, a = _stopped_builder(stop_sec=16.0)
    a.with_id_switch(10.0)
    tracks = b.build()
    assert len(tracks.track_ids) == 2, "fixture did not actually switch the id"

    segs = stopped_mod.detect(tracks, zones)
    assert len(segs) == 1
    s, e = seconds(segs, tracks)[0]
    assert (e - s) == pytest.approx(16.0, abs=1.5)
    # ...and the segment knows it spanned both ids.
    assert len(segs[0].track_ids) == 2


def test_two_id_switches_mid_stop_still_one_event(zones):
    b, a = _stopped_builder(stop_sec=20.0)
    a.with_id_switch(8.0).with_id_switch(14.0)
    tracks = b.build()
    assert len(tracks.track_ids) == 3
    segs = stopped_mod.detect(tracks, zones)
    assert len(segs) == 1
    s, e = seconds(segs, tracks)[0]
    assert (e - s) == pytest.approx(20.0, abs=1.5)


def test_a_short_dropout_mid_stop_does_not_split_the_event(zones):
    b, a = _stopped_builder(stop_sec=18.0)
    a.with_dropout(9.0, 10.5)
    segs = stopped_mod.detect(b.build(), zones)
    assert len(segs) == 1


def test_an_occlusion_mid_stop_survives_both_the_gap_and_the_new_id(zones):
    """The realistic case: occluded longer than track_buffer, so it comes back
    with a new id AND a hole in the data."""
    b, a = _stopped_builder(stop_sec=20.0)
    a.with_occlusion(9.0, 10.8)
    tracks = b.build()
    assert len(tracks.track_ids) == 2
    segs = stopped_mod.detect(tracks, zones)
    assert len(segs) == 1
    s, e = seconds(segs, tracks)[0]
    assert (e - s) == pytest.approx(20.0, abs=2.0)


def test_stopped_vehicle_survives_bbox_jitter(zones):
    b, a = _stopped_builder(stop_sec=16.0)
    a.with_jitter(4.0).with_heading_noise(8.0)
    segs = stopped_mod.detect(b.build(), zones)
    assert len(segs) == 1


def test_a_long_gap_does_split_two_different_stops(zones):
    """Beyond max_gap_sec the cluster is closed: two vehicles, two events."""
    b = TrackBuilder(duration=90.0)
    for tid, t0 in ((1, 0.0), (2, 40.0)):
        (b.vehicle(track_id=tid)
          .start_at(point_in_lane("ns_nb_outer", 0.02), t=t0)
          .drive_to(STOPPED_POINT, speed=220.0)
          .stop_for(14.0)
          .drive_to(point_in_lane("ns_nb_outer", 0.9), speed=220.0))
    segs = stopped_mod.detect(b.build(), zones)
    assert len(segs) == 2


def test_stopped_vehicle_ignores_something_that_never_moves(zones):
    """Biased quiet: a detection that never arrives or leaves -- street
    furniture read as a car, or a car parked all clip whose ground point leaks
    over the kerb -- is not a stop event."""
    b = TrackBuilder(duration=60.0)
    b.vehicle().start_at(STOPPED_POINT, t=0.0).stop_for(55.0).with_jitter(3.0)
    assert stopped_mod.detect(b.build(), zones) == []


def test_stopped_vehicle_ignores_the_back_of_a_queue_at_the_zone_edge(zones):
    """A vehicle stopped just outside a queue zone, jittering over its edge, is
    the back of the queue -- not a separate stop."""
    from fixtures.synthetic_scene import ns_x
    from src.geometry import distances_to_boundary

    q = zones.signal_queue_zones[0].polygon
    # Walk down the lane from the queue point until just outside the zone.
    y = QUEUE_POINT[1]
    while zones.in_signal_queue((ns_x(y, 0.5), y)):
        y += 2.0
    edge = (ns_x(y + 6.0, 0.5), y + 6.0)
    assert not zones.in_signal_queue(edge)
    assert distances_to_boundary([edge], q)[0] < 30.0
    b = TrackBuilder(duration=60.0)
    (b.vehicle()
      .start_at(point_in_lane("ns_nb_outer", 0.02), t=0.0)
      .drive_to(edge, speed=220.0)
      .stop_for(30.0).with_jitter(3.0)
      .drive_to(point_in_lane("ns_nb_outer", 0.9), speed=220.0))
    assert stopped_mod.detect(b.build(), zones) == []


# ===========================================================================
# wrong_way
# ===========================================================================
def test_wrong_way_fires_on_sustained_opposed_travel(zones):
    """Drive backwards along a northbound lane.

    Asserted as COVERAGE of the track's own lifetime rather than an absolute
    number of seconds: the latter depends on the fixture's lane length and
    silently becomes a test of arithmetic rather than of the rule.
    """
    b = TrackBuilder(duration=40.0)
    (b.vehicle()
      .start_at(point_in_lane("ns_nb_outer", 0.95), t=0.0)
      .drive_to(point_in_lane("ns_nb_outer", 0.02), speed=90.0))
    tracks = b.build()
    segs = wrong_way_mod.detect(tracks, zones)
    assert len(segs) >= 1
    total = sum(e - s for s, e in seconds(segs, tracks))
    lifetime = float(tracks.col("t_sec").max() - tracks.col("t_sec").min())
    assert lifetime > 5.0
    assert total > 0.9 * lifetime, f"only covered {total:.2f}s of {lifetime:.2f}s"


def test_wrong_way_silent_on_lawful_travel(zones):
    for lane_id in ("ns_nb_outer", "ns_sb_outer", "ew_eb_outer", "ew_wb_outer"):
        b = TrackBuilder(duration=30.0)
        (b.vehicle().start_at(point_in_lane(lane_id, 0.0), t=0.0)
           .drive_to(point_in_lane(lane_id, 1.0), speed=180.0))
        assert wrong_way_mod.detect(b.build(), zones) == [], lane_id


def test_wrong_way_silent_on_a_stationary_vehicle(zones):
    """The speed gate. angle_between returns 180 deg for a zero vector, so
    without it every parked car reads as driving the wrong way."""
    b = TrackBuilder(duration=40.0)
    b.vehicle().start_at(STOPPED_POINT, t=0.0).stop_for(35.0)
    assert wrong_way_mod.detect(b.build(), zones) == []


def test_wrong_way_silent_on_a_stationary_vehicle_with_jitter(zones):
    """Jitter gives a stationary vehicle a random heading every frame."""
    b = TrackBuilder(duration=40.0, seed=5)
    b.vehicle().start_at(STOPPED_POINT, t=0.0).stop_for(35.0).with_jitter(6.0)
    assert wrong_way_mod.detect(b.build(), zones) == []


def test_wrong_way_silent_on_a_legal_turn(zones):
    """THE false positive this rule is shaped around.

    A vehicle turning is inside its entry lane while already heading across it
    -- a 90 degree divergence that is entirely lawful.
    """
    b = TrackBuilder(duration=30.0)
    entry = point_in_lane("ns_nb_inner", 0.15)
    corner = point_in_lane("ns_nb_inner", 0.55)
    (b.vehicle()
      .start_at(point_in_lane("ns_nb_inner", 0.0), t=0.0)
      .drive_to(entry, speed=180.0)
      .turn_to(point_in_lane("ew_wb_inner", 0.62), via=corner, duration=4.0)
      .drive_to(point_in_lane("ew_wb_inner", 0.95), speed=180.0))
    assert wrong_way_mod.detect(b.build(), zones) == []


def test_wrong_way_silent_on_a_brief_heading_glitch(zones):
    """Debounce: noise must not clear the persistence bar."""
    b = TrackBuilder(duration=30.0, seed=2)
    (b.vehicle().start_at(point_in_lane("ns_nb_outer", 0.0), t=0.0)
       .drive_to(point_in_lane("ns_nb_outer", 1.0), speed=180.0)
       .with_heading_noise(25.0).with_jitter(6.0))
    assert wrong_way_mod.detect(b.build(), zones) == []


def test_wrong_way_survives_noise_on_a_real_violation(zones):
    b = TrackBuilder(duration=30.0, seed=8)
    (b.vehicle()
      .start_at(point_in_lane("ns_nb_outer", 0.75), t=0.0)
      .drive_to(point_in_lane("ns_nb_outer", 0.05), speed=160.0)
      .with_jitter(4.0).with_heading_noise(7.0))
    assert len(wrong_way_mod.detect(b.build(), zones)) >= 1


def test_wrong_way_survives_an_id_switch(zones):
    """An id switch splits the track, so each half must still clear the bar --
    and Stage 3 merges whatever fragments remain."""
    b = TrackBuilder(duration=40.0)
    (b.vehicle()
      .start_at(point_in_lane("ns_nb_outer", 0.95), t=0.0)
      .drive_to(point_in_lane("ns_nb_outer", 0.02), speed=110.0)
      .with_id_switch(12.0))
    tracks = b.build()
    segs = wrong_way_mod.detect(tracks, zones)
    assert len(segs) >= 1
    events = finalise([s for s in run_rules(tracks, zones)], tracks.duration)
    assert any(lbl == "wrong_way" for _s, _e, lbl in events)


def test_wrong_way_ignores_a_pedestrian_going_the_wrong_way(zones):
    """Pedestrians are not bound by lane direction."""
    b = TrackBuilder(duration=30.0)
    (b.pedestrian()
      .start_at(point_in_lane("ns_nb_outer", 0.75), t=0.0)
      .drive_to(point_in_lane("ns_nb_outer", 0.05), speed=120.0))
    assert wrong_way_mod.detect(b.build(), zones) == []


# ===========================================================================
# jaywalking
# ===========================================================================
def test_jaywalking_fires_on_a_pedestrian_on_the_carriageway(zones):
    b = TrackBuilder(duration=20.0)
    (b.pedestrian()
      .start_at((JAYWALK_POINT[0] - 260.0, JAYWALK_POINT[1]), t=0.0)
      .drive_to((JAYWALK_POINT[0] + 260.0, JAYWALK_POINT[1]), duration=8.0))
    tracks = b.build()
    segs = jaywalking_mod.detect(tracks, zones)
    assert len(segs) == 1
    s, e = seconds(segs, tracks)[0]
    assert (e - s) > 2.0


def test_jaywalking_silent_on_the_pavement(zones):
    b = TrackBuilder(duration=20.0)
    (b.pedestrian().start_at((100.0, 180.0), t=0.0)
       .drive_to((400.0, 200.0), duration=10.0))
    assert jaywalking_mod.detect(b.build(), zones) == []


def test_jaywalking_silent_on_a_marked_crossing(zones):
    """The whole point of the crossing polygons."""
    from fixtures.synthetic_scene import ns_x

    b = TrackBuilder(duration=20.0)
    (b.pedestrian()
      .start_at((ns_x(870.0, -0.9), 870.0), t=0.0)
      .drive_to((ns_x(870.0, 0.9), 870.0), duration=9.0))
    assert jaywalking_mod.detect(b.build(), zones) == []


def test_jaywalking_ignores_vehicles(zones):
    b = TrackBuilder(duration=20.0)
    (b.vehicle().start_at((JAYWALK_POINT[0] - 200.0, JAYWALK_POINT[1]), t=0.0)
       .drive_to((JAYWALK_POINT[0] + 200.0, JAYWALK_POINT[1]), duration=8.0))
    assert jaywalking_mod.detect(b.build(), zones) == []


def test_jaywalking_ignores_low_confidence_detections(zones):
    b = TrackBuilder(duration=20.0)
    (b.pedestrian(conf=0.10)
      .start_at((JAYWALK_POINT[0] - 200.0, JAYWALK_POINT[1]), t=0.0)
      .drive_to((JAYWALK_POINT[0] + 200.0, JAYWALK_POINT[1]), duration=8.0))
    assert jaywalking_mod.detect(b.build(), zones) == []


def test_jaywalking_ignores_a_rider_on_a_motorcycle(zones):
    """A person box inside a two-wheeler's box is a rider, not a pedestrian --
    the main source of "people" on a real carriageway."""
    b = TrackBuilder(duration=20.0)
    path = ((JAYWALK_POINT[0] - 260.0, JAYWALK_POINT[1]),
            (JAYWALK_POINT[0] + 260.0, JAYWALK_POINT[1]))
    b.vehicle(cls=3).start_at(path[0], t=0.0).drive_to(path[1], duration=8.0)
    b.pedestrian().start_at(path[0], t=0.0).drive_to(path[1], duration=8.0)
    tracks = b.build()
    # The same rider box scaled to sit inside the motorcycle's box.
    person = tracks.data[:, 3] == 0
    moto = tracks.data[tracks.data[:, 3] == 3]
    tracks.data[person, 5:9] = moto[:, 5:9]
    assert jaywalking_mod.detect(tracks, zones) == []


def test_jaywalking_silent_just_outside_the_zebra_edge(zones):
    """Walking along the edge of the stripes, inside the margin, is not
    jaywalking at this bar."""
    from fixtures.synthetic_scene import ns_x

    xw = zones.crossings[0].polygon
    y_top = float(xw[:, 1].min())
    b = TrackBuilder(duration=20.0)
    (b.pedestrian()
      .start_at((ns_x(y_top - 12.0, -0.9), y_top - 12.0), t=0.0)
      .drive_to((ns_x(y_top - 12.0, 0.9), y_top - 12.0), duration=9.0))
    assert jaywalking_mod.detect(b.build(), zones) == []


def test_jaywalking_does_not_blip_on_a_pedestrian_hovering_at_the_kerb(zones):
    """Jitter across the kerb line must not produce a shower of sub-second
    events, each of which would be a false positive."""
    from fixtures.synthetic_scene import ns_x

    kerb_y = 1000.0
    b = TrackBuilder(duration=30.0, seed=6)
    (b.pedestrian()
      .start_at((ns_x(kerb_y, -1.0), kerb_y), t=0.0)
      .stop_for(25.0)
      .with_jitter(9.0))
    segs = jaywalking_mod.detect(b.build(), zones)
    assert len(segs) <= 1


def test_jaywalking_survives_a_dropout_mid_crossing(zones):
    b = TrackBuilder(duration=25.0)
    (b.pedestrian()
      .start_at((JAYWALK_POINT[0] - 260.0, JAYWALK_POINT[1]), t=0.0)
      .drive_to((JAYWALK_POINT[0] + 260.0, JAYWALK_POINT[1]), duration=12.0)
      .with_dropout(5.0, 5.8))
    segs = jaywalking_mod.detect(b.build(), zones)
    assert len(segs) == 1


def test_jaywalking_id_switch_fragments_are_merged_by_stage_3(zones):
    """This rule is id-keyed, so an id switch splits it -- and Stage 3's
    merge_gap_sec is what repairs that. Documents the seam honestly."""
    b = TrackBuilder(duration=25.0)
    (b.pedestrian()
      .start_at((JAYWALK_POINT[0] - 260.0, JAYWALK_POINT[1]), t=0.0)
      .drive_to((JAYWALK_POINT[0] + 260.0, JAYWALK_POINT[1]), duration=12.0)
      .with_id_switch(6.0))
    tracks = b.build()
    assert len(jaywalking_mod.detect(tracks, zones)) == 2      # split by the rule
    events = [e for e in finalise(run_rules(tracks, zones), tracks.duration)
              if e[2] == "jaywalking"]
    assert len(events) == 1                                     # repaired downstream


# ===========================================================================
# congestion
# ===========================================================================
def _fill_lane(builder, lane_id, n, along_from, along_to, speed, duration,
               t0=0.0):
    """n vehicles spread along a lane, all crawling."""
    for i in range(n):
        a0 = along_from + (along_to - along_from) * i / max(1, n - 1)
        start = point_in_lane(lane_id, a0)
        end = point_in_lane(lane_id, min(1.0, a0 + 0.02))
        (builder.vehicle()
                .start_at(start, t=t0)
                .drive_to(end, duration=duration))


def test_congestion_fires_when_every_lane_of_a_direction_crawls(zones):
    b = TrackBuilder(duration=60.0)
    for lane_id in ("ns_nb_inner", "ns_nb_outer"):
        _fill_lane(b, lane_id, 4, 0.72, 0.98, speed=10.0, duration=45.0)
    tracks = b.build()
    segs = congestion_mod.detect(tracks, zones)
    assert len(segs) >= 1
    total = sum(e - s for s, e in seconds(segs, tracks))
    assert total > 20.0


def test_congestion_silent_when_the_direction_keeps_flowing(zones):
    """Judged for the direction as a whole: one crawling lane beside a lane
    that flows the whole time is not a congested direction."""
    b = TrackBuilder(duration=60.0)
    _fill_lane(b, "ns_nb_outer", 4, 0.72, 0.98, speed=10.0, duration=45.0)
    for i in range(22):
        (b.vehicle().start_at(point_in_lane("ns_nb_inner", 0.05), t=2.0 * i)
           .drive_to(point_in_lane("ns_nb_inner", 0.95), speed=200.0))
    assert congestion_mod.detect(b.build(), zones) == []


def _named_zones(zones, groups):
    """The same scene with explicit direction_group names on the lanes."""
    import dataclasses

    lanes = tuple(dataclasses.replace(ln, direction_group=groups.get(ln.id))
                  for ln in zones.lanes)
    return dataclasses.replace(zones, lanes=lanes)


def test_congestion_uses_named_direction_groups(zones):
    """zones.json can name the directions; then clustering is not used and a
    lane with no group takes no part."""
    named = _named_zones(zones, {"ns_nb_inner": "northbound",
                                 "ns_nb_outer": "northbound"})
    groups = congestion_mod.direction_groups(named, 45.0)
    assert [name for name, _ in groups] == ["northbound"]

    b = TrackBuilder(duration=60.0)
    for lane_id in ("ns_nb_inner", "ns_nb_outer"):
        _fill_lane(b, lane_id, 4, 0.72, 0.98, speed=10.0, duration=45.0)
    tracks = b.build()
    segs = congestion_mod.detect(tracks, named)
    assert len(segs) >= 1 and segs[0].debug["direction"] == "northbound"

    # The same crawl in lanes that belong to no named direction is ignored.
    other = _named_zones(zones, {"ew_eb_inner": "eastbound"})
    assert congestion_mod.detect(b.build(), other) == []


def test_congestion_silent_on_a_normal_red_light_queue(zones):
    """THE false positive this rule is shaped around.

    A queue waiting on red sits inside the signal queue zone. Those vehicles
    are excluded from the statistics entirely, so the lanes have no measurable
    traffic and cannot be judged congested -- however long the red lasts.
    """
    from fixtures.synthetic_scene import SL_NS_SOUTH_Y, ns_x

    b = TrackBuilder(duration=80.0)
    for lane_off in (0.25, 0.75):
        for i in range(4):
            y = SL_NS_SOUTH_Y + 12.0 + i * 28.0     # inside q_ns_south
            pt = (ns_x(y, lane_off), y)
            b.vehicle().start_at(pt, t=0.0).stop_for(60.0)

    tracks = b.build()
    # Sanity: the fixture really does put them in the queue zone.
    assert zones.in_signal_queue((ns_x(SL_NS_SOUTH_Y + 12.0, 0.75),
                                  SL_NS_SOUTH_Y + 12.0))
    assert congestion_mod.detect(tracks, zones) == []


def test_congestion_silent_on_free_flowing_traffic(zones):
    b = TrackBuilder(duration=60.0)
    for lane_id in ("ns_nb_inner", "ns_nb_outer"):
        for i in range(4):
            (b.vehicle().start_at(point_in_lane(lane_id, 0.02 + 0.08 * i), t=0.0)
               .drive_to(point_in_lane(lane_id, 1.0), speed=240.0))
    assert congestion_mod.detect(b.build(), zones) == []


def test_congestion_silent_on_a_single_slow_vehicle(zones):
    """min_vehicles_per_lane: one slow car in an empty lane is not a jam."""
    b = TrackBuilder(duration=60.0)
    for lane_id in ("ns_nb_inner", "ns_nb_outer"):
        (b.vehicle().start_at(point_in_lane(lane_id, 0.8), t=0.0)
           .drive_to(point_in_lane(lane_id, 0.85), duration=45.0))
    assert congestion_mod.detect(b.build(), zones) == []


def test_congestion_silent_on_a_brief_slowdown(zones):
    b = TrackBuilder(duration=60.0)
    for lane_id in ("ns_nb_inner", "ns_nb_outer"):
        _fill_lane(b, lane_id, 4, 0.72, 0.98, speed=10.0, duration=6.0)
    assert congestion_mod.detect(b.build(), zones) == []


def test_congestion_survives_jitter_and_dropouts(zones):
    b = TrackBuilder(duration=60.0, seed=12)
    for lane_id in ("ns_nb_inner", "ns_nb_outer"):
        _fill_lane(b, lane_id, 4, 0.72, 0.98, speed=10.0, duration=45.0)
    b.degrade_all(jitter_px=4.0, heading_noise_deg=10.0)
    b.actors[0].with_dropout(12.0, 14.0)
    b.actors[5].with_occlusion(20.0, 22.5)
    assert len(congestion_mod.detect(b.build(), zones)) >= 1


def test_direction_grouping_finds_four_directions(zones):
    groups = congestion_mod.group_lanes_by_direction(zones, 45.0)
    assert len(groups) == 4
    for grp in groups:
        assert len(grp) == 2
    # Opposing lanes must never share a group.
    for grp in groups:
        dirs = [zones.lanes[i].direction for i in grp]
        assert np.dot(dirs[0], dirs[1]) > 0


# ===========================================================================
# the Stage 3 seam
# ===========================================================================
def test_rules_return_frame_segments_not_seconds(zones):
    b, _ = _stopped_builder(stop_sec=16.0)
    segs = stopped_mod.detect(b.build(), zones)
    assert all(isinstance(s, FrameSegment) for s in segs)
    assert all(isinstance(s.start_frame, (int, np.integer)) for s in segs)


def test_run_rules_converts_frames_to_seconds(zones):
    b, _ = _stopped_builder(stop_sec=16.0)
    tracks = b.build()
    raw = run_rules(tracks, zones)
    assert raw
    for r in raw:
        assert 0.0 <= r.start < r.end <= tracks.duration + 1.0
        assert r.label in RULES


def test_no_rule_invents_its_own_merge_logic(zones):
    """Everything goes through finalise, so the output is always submittable."""
    b = TrackBuilder(duration=60.0)
    (b.vehicle().start_at(point_in_lane("ns_nb_outer", 0.02), t=0.0)
       .drive_to(STOPPED_POINT, speed=220.0).stop_for(30.0))
    (b.pedestrian()
      .start_at((JAYWALK_POINT[0] - 260.0, JAYWALK_POINT[1]), t=2.0)
      .drive_to((JAYWALK_POINT[0] + 260.0, JAYWALK_POINT[1]), duration=9.0))
    tracks = b.build()
    events = finalise(run_rules(tracks, zones), tracks.duration)
    assert_no_overlap(events)
    for s, e, lbl in events:
        assert 0.0 <= s < e <= tracks.duration
        assert lbl in RULES


def test_output_passes_the_official_validator(zones):
    import evaluate

    b, _ = _stopped_builder(stop_sec=18.0)
    tracks = b.build()
    events = finalise(run_rules(tracks, zones), tracks.duration)
    pred = {"team": "t", "videos": {"v.mp4": {"events": events, "risk": []}}}
    errors, _ = evaluate.validate(pred)
    assert errors == []


def test_stage_2_is_fast(zones):
    """The architectural promise: Stage 2 runs off the cache in well under a
    second, which is what makes fifty rule iterations a day possible.

    Asserted as a RATIO against one ZoneIndex build rather than as wall-clock
    seconds. An absolute bar measures the machine and whatever else is running
    on it -- this same check passed at 0.58 s alone and failed at 1.03 s inside
    a loaded full-suite run, which is a property of the CI box, not of the code.
    The ratio is immune to both.

    What it actually guards: that the zone index is built ONCE and shared by all
    four rules. Before that fix each rule rebuilt it and the ratio was ~8.8x;
    after, it is ~1.7x. A 4x bar catches a regression without being flaky.
    """
    import time

    from src.rules.zoneindex import ZoneIndex

    # A 3-minute clip at stride 2 is 2250 sampled frames; ~15 objects present
    # throughout is a busy but ordinary intersection, so ~34k rows.
    b = TrackBuilder(duration=180.0)
    for lane_id in ("ns_nb_inner", "ns_nb_outer", "ns_sb_inner", "ns_sb_outer"):
        for i in range(4):
            (b.vehicle().start_at(point_in_lane(lane_id, 0.02 + 0.05 * i), t=0.0)
               .drive_to(point_in_lane(lane_id, 1.0), duration=178.0))
    for i in range(4):
        (b.pedestrian().start_at((300.0 + 80 * i, 980.0), t=0.0)
           .drive_to((1500.0, 980.0), duration=175.0))
    tracks = b.build()
    assert len(tracks) > 30000, f"not a realistic row count: {len(tracks)}"

    veh = tracks.vehicles()
    t0 = time.perf_counter()
    ZoneIndex.build(veh.data, zones)
    index_cost = time.perf_counter() - t0
    assert index_cost > 0.0

    fresh = b.build()               # no memoised index
    t0 = time.perf_counter()
    run_rules(fresh, zones)
    elapsed = time.perf_counter() - t0

    ratio = elapsed / index_cost
    assert ratio < 4.0, (
        f"Stage 2 cost {elapsed:.2f}s = {ratio:.1f}x one index build "
        f"({index_cost:.2f}s) on {len(tracks)} rows; the index is probably "
        f"being rebuilt per rule"
    )


# ===========================================================================
# lessons from the real footage (sample_001/002, 2026-09-24)
# ===========================================================================
def _stop_at(b, pt, stop_sec=16.0):
    (b.vehicle()
      .start_at(point_in_lane("ns_nb_outer", 0.02), t=0.0)
      .drive_to(pt, speed=220.0)
      .stop_for(stop_sec)
      .drive_to(point_in_lane("ns_nb_outer", 0.9), speed=220.0))


def test_stopped_vehicle_silent_when_part_of_a_queue(zones):
    """Two cars standing side by side are traffic standing -- a queue or a jam
    -- not a stopped vehicle. Real run: a taxi in a line of five stopped cars."""
    beside = (STOPPED_POINT[0] - 90.0, STOPPED_POINT[1])
    assert zones.on_carriageway(beside) and not zones.in_signal_queue(beside)
    b = TrackBuilder(duration=40.0)
    _stop_at(b, STOPPED_POINT)
    _stop_at(b, beside)
    assert stopped_mod.detect(b.build(), zones) == []

    alone = TrackBuilder(duration=40.0)          # control: the same stop, alone, fires
    _stop_at(alone, STOPPED_POINT)
    assert len(stopped_mod.detect(alone.build(), zones)) == 1


def test_stopped_vehicle_ignores_a_box_cut_by_the_frame(zones):
    """A box truncated by the frame has the frame edge for a ground point."""
    edge = (STOPPED_POINT[0], 1080.0)
    assert zones.on_carriageway(edge)
    b = TrackBuilder(duration=40.0)
    _stop_at(b, edge, stop_sec=20.0)
    assert stopped_mod.detect(b.build(), zones) == []


def test_wrong_way_only_judges_painted_lanes(zones):
    """An unpainted lane's direction is not trustworthy enough to accuse anyone.
    Real run: 11 lawful vehicles flagged in the unpainted area right of the refuge."""
    import dataclasses

    from src.zones import LaneMarking

    painted = {lid for m in zones.lane_markings for lid in m.separates}
    assert "ns_sb_outer" not in painted
    b = TrackBuilder(duration=40.0)
    (b.vehicle()
      .start_at(point_in_lane("ns_sb_outer", 0.95), t=0.0)
      .drive_to(point_in_lane("ns_sb_outer", 0.02), speed=90.0))
    tracks = b.build()
    assert wrong_way_mod.detect(tracks, zones) == []

    # Control: paint that lane and the same drive is judged.
    mark = LaneMarking("m_sb", ((0.0, 0.0), (1.0, 1.0)), "dashed", ("ns_sb_outer",))
    marked = dataclasses.replace(zones, lane_markings=zones.lane_markings + (mark,))
    assert len(wrong_way_mod.detect(tracks, marked)) >= 1
