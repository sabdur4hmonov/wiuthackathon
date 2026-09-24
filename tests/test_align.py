"""Per-video camera-pose alignment (src/align.py).

The failure this guards against is silent: a clip framed slightly differently
from the one zones.json was drawn on puts every lane on its neighbour. The
fail-safe matters as much as the fit -- a wrong warp is worse than none -- so
every fallback path is tested, not just the happy one.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src import align
from src.align import IDENTITY, FrameMatch, Pose, estimate, match_frame, prepare, warp_zones
from src.zones import load_zones


def _similarity(scale, angle_deg, tx, ty) -> np.ndarray:
    a = math.radians(angle_deg)
    return np.array([[scale * math.cos(a), -scale * math.sin(a), tx],
                     [scale * math.sin(a), scale * math.cos(a), ty]])


@pytest.fixture(scope="module")
def reference():
    img = cv2.imread(str(align.REFERENCE_PATH))
    assert img is not None, "config/pose_reference.jpg must ship with the repo"
    return img


# ---------------------------------------------------------------------------
# estimation
# ---------------------------------------------------------------------------
def test_the_reference_matched_to_itself_is_identity(reference):
    m = match_frame(prepare(reference))
    assert m.inliers >= 200
    pose = estimate([], matches=[m])
    assert pose.reason == "aligned"
    assert pose.scale == pytest.approx(1.0, abs=2e-3)
    assert pose.angle_deg == pytest.approx(0.0, abs=0.05)
    assert math.hypot(pose.tx, pose.ty) < 0.5


def test_recovers_a_known_pan_zoom_and_roll(reference):
    """The camera moved: the same scene, shifted, zoomed 1.2 % and rolled 0.8 deg."""
    h, w = reference.shape[:2]
    true = _similarity(1.012, -0.8, -14.0, 9.0)
    moved = cv2.warpAffine(reference, true, (w, h), borderMode=cv2.BORDER_REFLECT)
    pose = estimate([prepare(moved)])
    assert pose.reason == "aligned"
    assert pose.scale == pytest.approx(1.012, abs=3e-3)
    assert pose.angle_deg == pytest.approx(-0.8, abs=0.1)
    assert pose.tx == pytest.approx(-14.0, abs=1.0)
    assert pose.ty == pytest.approx(9.0, abs=1.0)


def test_matrix_scales_translation_with_the_frame_width():
    p = Pose(1.0, 0.0, 10.0, -5.0, reason="aligned")
    m = p.matrix(3840)
    assert m[0, 2] == pytest.approx(40.0) and m[1, 2] == pytest.approx(-20.0)
    assert np.allclose(m[:, :2], np.eye(2))


def test_pose_round_trips_through_a_dict():
    p = Pose(1.01, -0.5, 3.0, 4.0, good_frames=5, frames=9, median_inliers=120,
             drift_px=1.5, reason="aligned")
    assert Pose.from_dict(p.to_dict()) == p
    assert Pose.from_dict(None) is None


# ---------------------------------------------------------------------------
# fail safe: every path that must fall back to identity
# ---------------------------------------------------------------------------
def _good(m) -> FrameMatch:
    return FrameMatch(np.asarray(m, dtype=float), inliers=300, matches=400)


def test_no_frames_is_identity():
    pose = estimate([])
    assert pose.is_identity and pose.reason == "no frames"


def test_a_different_scene_is_identity():
    """Noise shares no structure with the reference: weak match -> identity."""
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)
    pose = estimate([prepare(noise)])
    assert pose.is_identity
    assert pose.reason.startswith("weak match")


def test_too_few_inliers_is_identity():
    m = FrameMatch(_similarity(1.0, 0.0, 5, 5), inliers=12, matches=400)
    pose = estimate([], matches=[m])
    assert pose.is_identity and pose.reason.startswith("weak match")


@pytest.mark.parametrize("matrix, reason", [
    (_similarity(1.0, 9.0, 0, 0), "implausible rotation"),
    (_similarity(1.2, 0.0, 0, 0), "implausible scale"),
    (_similarity(1.0, 0.0, 300, 0), "implausible shift"),
])
def test_an_implausible_transform_is_identity(matrix, reason):
    pose = estimate([], matches=[_good(matrix)])
    assert pose.is_identity
    assert pose.reason.startswith(reason)


def test_frames_that_disagree_are_identity():
    """Half the frames say one pose, half another: the camera moved mid-clip,
    and no single warp is right for the whole video."""
    a, b = _similarity(1.0, 0.0, 0, 0), _similarity(1.0, 0.0, 40, 0)
    pose = estimate([], matches=[_good(a), _good(b), _good(a), _good(b)])
    assert pose.is_identity and pose.reason.startswith("frames disagree")


def test_the_median_ignores_one_settling_frame():
    """sample_001 drifts at t=0 while the camera settles; the median must not."""
    steady = _similarity(1.0, 0.0, 0.0, 0.0)
    settling = _similarity(1.0, 0.0, 6.0, 0.0)
    pose = estimate([], matches=[_good(settling)] + [_good(steady)] * 8)
    assert pose.reason == "aligned"
    assert abs(pose.tx) < 0.5


# ---------------------------------------------------------------------------
# warping zones
# ---------------------------------------------------------------------------
@pytest.fixture
def zones(tmp_path):
    d = {
        "schema_version": 1,
        "authored_against": {"image_width": 1920, "image_height": 1080,
                             "source_video": "clip.mp4", "source_frame_idx": 0},
        "carriageway": [{"id": "main", "polygon": [[0, 500], [1920, 500], [1920, 1080], [0, 1080]]}],
        "lanes": [{"id": "nb", "polygon": [[0, 500], [960, 500], [960, 1080], [0, 1080]],
                   "direction_from": [100, 1000], "direction_to": [100, 600],
                   "permitted_manoeuvres": [], "governed_by_stop_line": "sl"}],
        "stop_lines": [{"id": "sl", "segment": [[0, 700], [960, 700]],
                        "approach_side": 1, "governs_lanes": ["nb"]}],
        "crossings": [{"id": "x", "polygon": [[0, 620], [1920, 620], [1920, 690], [0, 690]]}],
        "signal_queue_zones": [{"id": "q", "polygon": [[0, 700], [960, 700], [960, 950], [0, 950]],
                                "lanes": ["nb"]}],
        "traffic_lights": [{"id": "sig", "roi": [1500, 100, 1560, 220], "controls_lanes": ["nb"]}],
        "lane_markings": [{"id": "m", "segment": [[960, 500], [960, 1080]], "style": "solid",
                           "separates": ["nb"]}],
        "notes": "",
    }
    p = tmp_path / "zones.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return load_zones(p)


def test_identity_pose_leaves_zones_untouched(zones):
    assert warp_zones(zones, IDENTITY, 1920) is zones
    assert warp_zones(zones, None, 1920) is zones


def test_polygons_move_with_the_similarity(zones):
    pose = Pose(1.01, -1.0, 12.0, -6.0, reason="aligned")
    m = pose.matrix(1920)
    w = warp_zones(zones, pose, 1920)
    expect = zones.carriageway[0].polygon @ m[:, :2].T + m[:, 2]
    assert np.allclose(w.carriageway[0].polygon, expect)
    assert np.allclose(w.lanes[0].polygon, zones.lanes[0].polygon @ m[:, :2].T + m[:, 2])


def test_lane_directions_rotate_with_the_frame(zones):
    """The arrow must turn exactly as the polygons do -- and stay a unit vector."""
    pose = Pose(1.014, -1.08, -8.3, 10.4, reason="aligned")
    w = warp_zones(zones, pose, 1920)
    before, after = np.array(zones.lanes[0].direction), np.array(w.lanes[0].direction)
    assert np.hypot(*after) == pytest.approx(1.0)
    turned = math.degrees(math.atan2(after[1], after[0]) - math.atan2(before[1], before[0]))
    assert turned == pytest.approx(-1.08, abs=1e-6)
    # Same as carrying the drawn arrow's two endpoints through the warp.
    m = pose.matrix(1920)
    tail, head = np.array([100.0, 1000.0]), np.array([100.0, 600.0])
    via_points = (m[:, :2] @ head + m[:, 2]) - (m[:, :2] @ tail + m[:, 2])
    assert np.allclose(after, via_points / np.hypot(*via_points))


def test_stop_line_keeps_its_approach_side(zones):
    pose = Pose(1.014, 2.5, 30.0, -20.0, reason="aligned")
    w = warp_zones(zones, pose, 1920)
    m = pose.matrix(1920)
    waiting = np.array([480.0, 800.0])          # queued upstream of the line
    moved = m[:, :2] @ waiting + m[:, 2]
    assert zones.stop_lines[0].signed_distance(tuple(waiting)) > 0
    assert w.stop_lines[0].signed_distance(tuple(moved)) > 0


def test_signal_roi_follows_the_lamps(zones):
    pose = Pose(1.0, 0.0, 10.0, 5.0, reason="aligned")
    w = warp_zones(zones, pose, 1920)
    assert w.traffic_lights[0].roi == pytest.approx((1520.0, 110.0, 1580.0, 230.0))   # (10, 5) @960 = (20, 10) @1920


# ---------------------------------------------------------------------------
# plumbing: the cache keeps the pose; the pipeline never raises on it
# ---------------------------------------------------------------------------
def test_the_cache_keeps_the_pose(tmp_path, monkeypatch):
    from src import cache
    from src.tracks import TrackTable

    monkeypatch.setenv(cache.CACHE_ENV_VAR, str(tmp_path))
    monkeypatch.delenv(cache.DISABLE_ENV_VAR, raising=False)
    pose = Pose(1.01, -0.7, -30.0, 18.0, good_frames=9, frames=9, reason="aligned")
    t = TrackTable.empty(fps=30.0, duration=10.0, n_frames=300, width=3840, height=2160)
    t.pose = pose.to_dict()
    assert cache.store("v", "c", t)
    assert Pose.from_dict(cache.load("v", "c").pose) == pose


def test_pipeline_uses_the_pose_riding_on_the_tracks(zones):
    from src.budget import Budget
    from src.pipeline import _align_zones
    from src.tracks import TrackTable

    pose = Pose(1.0, 0.0, 10.0, 0.0, reason="aligned")
    t = TrackTable.empty(width=1920, height=1080)
    t.pose = pose.to_dict()
    b = Budget(duration=60.0, fps=30.0, n_frames=1800, t0=__import__("time").perf_counter())
    w = _align_zones("unused.mp4", t, zones, b, verbose=False)
    assert w.carriageway[0].polygon[0][0] == pytest.approx(20.0)


def test_pipeline_leaves_zones_unwarped_when_alignment_is_impossible(zones, tmp_path):
    """No stored pose and no readable video: identity, never an exception."""
    from src.budget import Budget
    from src.pipeline import _align_zones
    from src.tracks import TrackTable

    t = TrackTable.empty(width=1920, height=1080)
    b = Budget(duration=60.0, fps=30.0, n_frames=1800, t0=__import__("time").perf_counter())
    w = _align_zones(str(tmp_path / "missing.mp4"), t, zones, b, verbose=False)
    assert np.allclose(w.carriageway[0].polygon, zones.carriageway[0].polygon)
