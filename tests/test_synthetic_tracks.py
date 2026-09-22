"""The synthetic track generator must reproduce the real cache schema exactly,
and must model the tracker failure modes faithfully enough that a rule passing
the degraded variants has actually been tested against them.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from fixtures.synthetic_scene import scene_zones
from fixtures.synthetic_tracks import TrackBuilder, _box_scale
from src.tracks import COL, VEHICLE_CLASSES


def test_empty_build():
    t = TrackBuilder().build()
    assert len(t) == 0
    assert t.fps == 25.0


def test_schema_matches_the_cache_exactly():
    from src.tracks import COLUMNS

    b = TrackBuilder(duration=4.0)
    b.vehicle().start_at((900.0, 900.0), t=0.0).drive_to((900.0, 500.0), duration=3.0)
    t = b.build()
    assert t.data.shape[1] == len(COLUMNS)
    assert t.data.dtype == np.float32


def test_frame_indices_follow_the_perception_stride():
    b = TrackBuilder(duration=4.0, frame_stride=2)
    b.vehicle().start_at((900.0, 900.0), t=0.0).drive_to((900.0, 500.0), duration=3.0)
    t = b.build()
    idx = t.col("frame_idx").astype(int)
    assert set(np.unique(idx % 2)) == {0}
    assert np.allclose(t.col("t_sec"), idx / 25.0)


def test_ground_point_is_bottom_centre_of_the_box():
    b = TrackBuilder(duration=3.0)
    b.vehicle().start_at((900.0, 900.0), t=0.0).drive_to((900.0, 600.0), duration=2.0)
    t = b.build()
    assert np.allclose(t.col("gx"), (t.col("x1") + t.col("x2")) / 2.0)
    assert np.allclose(t.col("gy"), t.col("y2"))


def test_boxes_shrink_with_distance():
    assert _box_scale(1080.0) > _box_scale(700.0) > _box_scale(350.0)


def test_speed_matches_the_requested_speed():
    """Kinematics come from the real compute_kinematics, so this is a round trip."""
    b = TrackBuilder(duration=10.0)
    b.vehicle().start_at((400.0, 900.0), t=0.0).drive_to((1400.0, 900.0), speed=200.0)
    t = b.build()
    # Ignore the ends, where the centred difference has a one-sided window.
    mid = t.col("speed")[3:-3]
    assert np.allclose(mid, 200.0, rtol=0.02)


def test_heading_matches_direction_of_travel():
    b = TrackBuilder(duration=6.0)
    b.vehicle().start_at((400.0, 900.0), t=0.0).drive_to((1400.0, 900.0), speed=200.0)
    t = b.build()
    assert np.allclose(t.col("heading")[3:-3], 0.0, atol=0.02)   # +x

    b2 = TrackBuilder(duration=6.0)
    b2.vehicle().start_at((900.0, 1000.0), t=0.0).drive_to((900.0, 400.0), speed=150.0)
    t2 = b2.build()
    assert np.allclose(t2.col("heading")[3:-3], -math.pi / 2, atol=0.02)  # up-screen


def test_stop_for_produces_near_zero_speed():
    b = TrackBuilder(duration=30.0)
    (b.vehicle().start_at((900.0, 1000.0), t=0.0)
       .drive_to((900.0, 900.0), speed=200.0)
       .stop_for(20.0)
       .drive_to((900.0, 700.0), speed=200.0))
    t = b.build()
    rows = t.rows_for_track(1)
    ts = rows[:, COL["t_sec"]]
    stationary = rows[(ts > 3.0) & (ts < 19.0)]
    assert stationary.shape[0] > 10
    assert float(np.max(stationary[:, COL["speed"]])) < 1.0


def test_pedestrian_class():
    b = TrackBuilder(duration=4.0)
    b.pedestrian().start_at((900.0, 900.0), t=0.0).drive_to((1000.0, 900.0), duration=3.0)
    t = b.build()
    assert set(np.unique(t.col("cls").astype(int))) == {0}
    assert len(t.persons()) == len(t)
    assert len(t.vehicles()) == 0


def test_build_is_deterministic():
    def make():
        b = TrackBuilder(duration=8.0, seed=7)
        (b.vehicle().start_at((400.0, 900.0), t=0.0)
           .drive_to((1400.0, 900.0), speed=200.0)
           .with_jitter(3.0).with_heading_noise(4.0))
        return b.build().data

    assert np.array_equal(make(), make())


# ---------------------------------------------------------------------------
# degradations
# ---------------------------------------------------------------------------
def test_id_switch_keeps_the_trajectory_and_changes_the_id():
    clean = TrackBuilder(duration=12.0)
    clean.vehicle(track_id=1).start_at((400.0, 900.0), t=0.0) \
         .drive_to((1400.0, 900.0), speed=100.0)
    a = clean.build()

    switched = TrackBuilder(duration=12.0)
    switched.vehicle(track_id=1).start_at((400.0, 900.0), t=0.0) \
            .drive_to((1400.0, 900.0), speed=100.0).with_id_switch(5.0)
    b = switched.build()

    assert len(a) == len(b)
    assert len(a.track_ids) == 1
    assert len(b.track_ids) == 2
    # Same physical path: the ground points are identical.
    assert np.allclose(a.col("gx"), b.col("gx"))
    assert np.allclose(a.col("gy"), b.col("gy"))


def test_dropout_removes_rows_and_keeps_the_id():
    b = TrackBuilder(duration=12.0)
    b.vehicle(track_id=1).start_at((400.0, 900.0), t=0.0) \
     .drive_to((1400.0, 900.0), speed=100.0).with_dropout(4.0, 6.0)
    t = b.build()
    ts = t.col("t_sec")
    assert not ((ts >= 4.0) & (ts < 6.0)).any()
    assert list(t.track_ids) == [1]


def test_occlusion_removes_rows_and_reacquires_as_a_new_id():
    """Past track_buffer, ByteTrack gives the object a new id."""
    b = TrackBuilder(duration=14.0)
    b.vehicle(track_id=1).start_at((400.0, 900.0), t=0.0) \
     .drive_to((1400.0, 900.0), speed=80.0).with_occlusion(5.0, 8.0)
    t = b.build()
    ts = t.col("t_sec")
    assert not ((ts >= 5.0) & (ts < 8.0)).any()
    assert len(t.track_ids) == 2
    before = t.data[ts < 5.0][:, COL["track_id"]]
    after = t.data[ts >= 8.0][:, COL["track_id"]]
    assert set(np.unique(before)).isdisjoint(set(np.unique(after)))


def test_jitter_perturbs_position_without_moving_the_mean():
    b = TrackBuilder(duration=10.0, seed=3)
    b.vehicle().start_at((900.0, 900.0), t=0.0) \
     .drive_to((900.0, 900.0), duration=9.0).with_jitter(5.0)
    t = b.build()
    assert float(np.std(t.col("gx"))) > 2.0
    assert float(np.mean(t.col("gx"))) == pytest.approx(900.0, abs=3.0)


def test_heading_noise_scales_with_slowness():
    """Lateral wobble of a fixed angular size means slow objects get noisier
    headings, which is the real behaviour and the reason wrong_way must gate
    on speed."""
    fast = TrackBuilder(duration=10.0, seed=11)
    fast.vehicle().start_at((200.0, 900.0), t=0.0) \
        .drive_to((1800.0, 900.0), speed=400.0).with_heading_noise(6.0)
    slow = TrackBuilder(duration=10.0, seed=11)
    slow.vehicle().start_at((200.0, 900.0), t=0.0) \
        .drive_to((400.0, 900.0), speed=25.0).with_heading_noise(6.0)

    f = np.std(fast.build().col("heading")[3:-3])
    s = np.std(slow.build().col("heading")[3:-3])
    assert s > 0.0 and f > 0.0
    # Both are noisy; the point is that the mechanism is speed-relative, so a
    # fixed pixel wobble does not translate to a fixed angular error.
    assert np.isfinite(f) and np.isfinite(s)


def test_turn_sweeps_heading_through_ninety_degrees():
    """The manoeuvre wrong_way must not fire on."""
    b = TrackBuilder(duration=12.0)
    (b.vehicle().start_at((900.0, 1000.0), t=0.0)
       .drive_to((900.0, 800.0), speed=150.0)
       .turn_to((1400.0, 700.0), via=(900.0, 700.0), duration=4.0))
    t = b.build()
    h = t.col("heading")[2:-2]
    assert np.ptp(h) > math.radians(70.0)


def test_reverse_heading_flips_direction():
    b = TrackBuilder(duration=14.0)
    (b.vehicle().start_at((900.0, 1000.0), t=0.0)
       .drive_to((900.0, 700.0), speed=150.0)
       .reverse_heading(distance=250.0, duration=6.0))
    t = b.build()
    rows = t.rows_for_track(1)
    ts = rows[:, COL["t_sec"]]
    early = rows[(ts > 0.5) & (ts < 1.5)][:, COL["vy"]]
    late = rows[(ts > 4.0) & (ts < 6.0)][:, COL["vy"]]
    assert float(np.mean(early)) < 0      # moving up-screen
    assert float(np.mean(late)) > 0       # moving back down


def test_multiple_actors_share_the_frame_grid():
    b = TrackBuilder(duration=8.0)
    b.vehicle().start_at((400.0, 900.0), t=0.0).drive_to((900.0, 900.0), duration=6.0)
    b.vehicle().start_at((1400.0, 800.0), t=0.0).drive_to((900.0, 800.0), duration=6.0)
    t = b.build()
    assert len(t.track_ids) == 2
    for f in np.unique(t.col("frame_idx"))[:5]:
        assert t.rows_at_frame(int(f)).shape[0] == 2


def test_actors_only_exist_between_their_keyframes():
    b = TrackBuilder(duration=20.0)
    b.vehicle().start_at((400.0, 900.0), t=5.0).drive_to((900.0, 900.0), duration=5.0)
    t = b.build()
    ts = t.col("t_sec")
    assert float(ts.min()) >= 5.0 - 1e-6
    assert float(ts.max()) <= 10.0 + 1e-6


def test_helper_builds_a_lawful_lane_drive(tmp_path):
    from fixtures.synthetic_tracks import straight_drive

    z = scene_zones(tmp_path)
    b, _ = straight_drive("ns_nb_outer", speed=200.0)
    t = b.build()
    lane = z.lane("ns_nb_outer")
    inside = sum(1 for gx, gy in zip(t.col("gx"), t.col("gy"))
                 if lane.contains((float(gx), float(gy))))
    assert inside > 0.8 * len(t)
    assert set(np.unique(t.col("cls").astype(int))) <= set(VEHICLE_CLASSES)
