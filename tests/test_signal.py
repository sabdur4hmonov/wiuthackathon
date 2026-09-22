"""Traffic-light classification.

The bias under test is asymmetry: a missed red costs one FN, a FALSE red turns
every lawful vehicle through the junction into a violation. So most of these
tests assert that the classifier says `unknown` rather than guessing.
"""
from __future__ import annotations

import numpy as np
import pytest

from fixtures.synthetic_scene import scene_zones
from src.signal import (AMBER, GREEN, RED, UNKNOWN, SignalReader, SignalState,
                        classify_crop)
from src.thresholds import SignalThresholds

cv2 = pytest.importorskip("cv2")


# ---------------------------------------------------------------------------
# synthetic crops
# ---------------------------------------------------------------------------
# BGR, as OpenCV delivers frames.
BGR = {
    RED: (0, 0, 255),
    AMBER: (0, 170, 255),
    GREEN: (0, 200, 0),
}


def lamp(colour: str, size: int = 40, fill: float = 1.0,
         bg: tuple[int, int, int] = (18, 18, 18)) -> np.ndarray:
    """A dark housing with a lit circular lamp covering `fill` of the crop."""
    img = np.full((size, size, 3), bg, dtype=np.uint8)
    if fill <= 0:
        return img
    radius = int(round(size * 0.5 * np.sqrt(fill)))
    cv2.circle(img, (size // 2, size // 2), radius, BGR[colour], -1)
    return img


def with_noise(img: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32) + rng.normal(0.0, sigma, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def blurred(img: np.ndarray, k: int = 7) -> np.ndarray:
    return cv2.GaussianBlur(img, (k | 1, k | 1), 0)


def dimmed(img: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("colour", [RED, AMBER, GREEN])
def test_solid_lamp_is_classified(colour):
    st = classify_crop(lamp(colour))
    assert st.state == colour, st.reason
    assert st.known and bool(st) is True
    assert 0.0 < st.confidence <= 1.0


@pytest.mark.parametrize("colour", [RED, AMBER, GREEN])
def test_survives_sensor_noise(colour):
    st = classify_crop(with_noise(lamp(colour), sigma=18.0, seed=1))
    assert st.state == colour, st.reason


@pytest.mark.parametrize("colour", [RED, AMBER, GREEN])
def test_survives_blur(colour):
    st = classify_crop(blurred(lamp(colour), k=9))
    assert st.state == colour, st.reason


@pytest.mark.parametrize("colour", [RED, AMBER, GREEN])
def test_survives_a_small_far_field_lamp(colour):
    """A signal at CCTV range is a handful of pixels."""
    st = classify_crop(lamp(colour, size=12))
    assert st.state == colour, st.reason


@pytest.mark.parametrize("colour", [RED, AMBER, GREEN])
def test_survives_noise_and_blur_together(colour):
    st = classify_crop(blurred(with_noise(lamp(colour), 14.0, seed=2), k=5))
    assert st.state == colour, st.reason


# ---------------------------------------------------------------------------
# the reluctance -- what must NOT produce an answer
# ---------------------------------------------------------------------------
def test_unlit_housing_is_unknown():
    """All three lamps off. Guessing here would invent a phase."""
    st = classify_crop(np.full((40, 40, 3), 20, dtype=np.uint8))
    assert st.state == UNKNOWN
    assert bool(st) is False


def test_very_dark_lamp_is_unknown():
    """Night-time underexposure: too dim to be sure."""
    st = classify_crop(dimmed(lamp(RED), 0.25))
    assert st.state == UNKNOWN, st.reason


def test_washed_out_lamp_is_unknown():
    """Blown-out daylight: bright but desaturated, so the hue is meaningless."""
    st = classify_crop(np.full((40, 40, 3), 250, dtype=np.uint8))
    assert st.state == UNKNOWN, st.reason


def test_a_tiny_number_of_lit_pixels_is_unknown():
    img = np.full((40, 40, 3), 18, dtype=np.uint8)
    img[0:2, 0:2] = BGR[RED]                      # 4 lit pixels
    st = classify_crop(img)
    assert st.state == UNKNOWN
    assert "lit pixels" in st.reason


def test_an_ambiguous_red_amber_mix_is_unknown():
    """Both lamps apparently lit. A coin flip here is the worst outcome."""
    img = np.full((40, 40, 3), 18, dtype=np.uint8)
    img[5:20, 5:35] = BGR[RED]
    img[20:35, 5:35] = BGR[AMBER]
    st = classify_crop(img)
    assert st.state == UNKNOWN, st.reason


def test_a_hue_outside_every_band_is_unknown():
    """A bright blue sign in the ROI is not a traffic light."""
    img = np.full((40, 40, 3), 18, dtype=np.uint8)
    img[8:32, 8:32] = (255, 40, 0)                # blue in BGR
    st = classify_crop(img)
    assert st.state == UNKNOWN, st.reason


@pytest.mark.parametrize("bad", [
    None,
    np.zeros((0, 0, 3), dtype=np.uint8),
    np.zeros((10, 10), dtype=np.uint8),           # not 3-channel
    np.zeros((4, 4, 4), dtype=np.uint8),          # 4-channel
])
def test_malformed_input_is_unknown_not_an_exception(bad):
    st = classify_crop(bad)
    assert st.state == UNKNOWN
    assert st.reason


def test_thresholds_are_configurable():
    """Loosening the bars is how this gets retuned against real crops."""
    dim = dimmed(lamp(RED), 0.25)
    assert classify_crop(dim).state == UNKNOWN
    loose = SignalThresholds(min_saturation=40, min_value=30, min_lit_pixels=4)
    assert classify_crop(dim, loose).state == RED


# ---------------------------------------------------------------------------
# SignalReader against the scene
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def zones(tmp_path_factory):
    return scene_zones(tmp_path_factory.mktemp("sigzones"))


def frame_with_signal(zones, colour: str | None) -> np.ndarray:
    """A full frame with the scene's signal ROI filled by `colour`."""
    frame = np.full((1080, 1920, 3), 60, dtype=np.uint8)
    lt = zones.traffic_lights[0]
    x1, y1, x2, y2 = (int(v) for v in lt.roi)
    frame[y1:y2, x1:x2] = 18
    if colour is not None:
        patch = lamp(colour, size=min(x2 - x1, y2 - y1))
        frame[y1:y1 + patch.shape[0], x1:x1 + patch.shape[1]] = patch
    return frame


def test_reader_is_available_when_the_scene_declares_a_light(zones):
    assert SignalReader(zones).available is True


@pytest.mark.parametrize("colour", [RED, AMBER, GREEN])
def test_reader_reads_the_roi(zones, colour):
    reader = SignalReader(zones)
    states = reader.read(frame_with_signal(zones, colour))
    assert set(states) == {zones.traffic_lights[0].id}
    assert states[zones.traffic_lights[0].id].state == colour


def test_state_for_a_controlled_lane(zones):
    reader = SignalReader(zones)
    frame = frame_with_signal(zones, RED)
    st = reader.state_for_lane(frame, "ns_nb_outer")
    assert st.state == RED
    assert reader.is_red(frame, "ns_nb_outer") is True


def test_a_lane_no_signal_controls_is_unknown(zones):
    reader = SignalReader(zones)
    frame = frame_with_signal(zones, RED)
    st = reader.state_for_lane(frame, "ew_eb_outer")
    assert st.state == UNKNOWN
    assert "no signal controls" in st.reason
    assert reader.is_red(frame, "ew_eb_outer") is None


def test_is_red_is_tri_state(zones):
    """None must be distinguishable from False, or a rule written as
    `if not is_red(...)` treats an unreadable signal as a green light."""
    reader = SignalReader(zones)
    assert reader.is_red(frame_with_signal(zones, GREEN), "ns_nb_outer") is False
    assert reader.is_red(frame_with_signal(zones, None), "ns_nb_outer") is None
    assert reader.is_red(frame_with_signal(zones, RED), "ns_nb_outer") is True


# ---------------------------------------------------------------------------
# degrading to "no signal available"
# ---------------------------------------------------------------------------
def test_no_traffic_lights_in_the_scene_degrades_cleanly(tmp_path):
    """The expected state until someone confirms a signal is in frame at all.

    config/zones.json treats an empty traffic_lights list as a legal, meaningful
    value: 'no signal visible'. Nothing downstream may break on it.
    """
    import json

    from fixtures.synthetic_scene import build_zones_dict
    from src.zones import load_zones

    d = build_zones_dict()
    d["traffic_lights"] = []
    p = tmp_path / "z.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    z = load_zones(p)

    reader = SignalReader(z)
    assert reader.available is False
    frame = np.full((1080, 1920, 3), 60, dtype=np.uint8)
    assert reader.read(frame) == {}
    assert reader.state_for_lane(frame, "ns_nb_outer").state == UNKNOWN
    assert reader.is_red(frame, "ns_nb_outer") is None


def test_reader_with_no_zones_at_all():
    reader = SignalReader(None)
    assert reader.available is False
    assert reader.read(np.zeros((10, 10, 3), dtype=np.uint8)) == {}
    assert reader.is_red(np.zeros((10, 10, 3), dtype=np.uint8), "any") is None


def test_roi_outside_the_frame_is_unknown(zones):
    """A 720p frame with a 1080p-authored ROI, before any rescaling."""
    reader = SignalReader(zones)
    small = np.full((240, 320, 3), 60, dtype=np.uint8)
    states = reader.read(small)
    assert all(st.state == UNKNOWN for st in states.values())


def test_reader_never_raises_on_a_malformed_frame(zones):
    reader = SignalReader(zones)
    for bad in (None, np.zeros((10, 10), dtype=np.uint8), "not a frame"):
        states = reader.read(bad)
        assert all(st.state == UNKNOWN for st in states.values())


def test_signal_state_defaults_are_safe():
    st = SignalState()
    assert st.state == UNKNOWN
    assert st.known is False
    assert bool(st) is False
