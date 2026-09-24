"""src/avdecode.py: keyframe-only decode must index the harness's frame sequence."""
import numpy as np
import pytest

from src import avdecode

pytestmark = pytest.mark.skipif(not avdecode.is_available(), reason="PyAV not installed")

N, FPS, W, H, GOP = 45, 25.0, 320, 240, 15


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """Structured like the real footage: H.264, fixed 15-frame GOP, B-frames.
    Frame i is a flat grey of level 20 + 5 i, so every frame is identifiable."""
    import av

    p = tmp_path_factory.mktemp("av") / "numbered.mp4"
    with av.open(str(p), "w") as out:
        s = out.add_stream("libx264", rate=int(FPS))
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.codec_context.gop_size = GOP
        s.codec_context.max_b_frames = 2
        s.codec_context.options = {"sc_threshold": "0", "bf": "2", "g": str(GOP),
                                   "keyint_min": str(GOP)}
        for i in range(N):
            img = np.full((H, W, 3), 20 + 5 * i, np.uint8)
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
                out.mux(pkt)
        for pkt in s.encode():
            out.mux(pkt)
    return p


@pytest.fixture(scope="module")
def cv2_means(clip):
    """What the harness sees: cv2's frame sequence."""
    import cv2

    cap = cv2.VideoCapture(str(clip))
    means = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        means.append(float(f.mean()))
    cap.release()
    return np.array(means)


def _check_against_cv2(got, cv2_means):
    for idx, img in got:
        # The frame decoded is cv2's frame `idx`, not a neighbour.
        assert int(np.argmin(np.abs(cv2_means - img.mean()))) == idx


def test_every_frame_mode_indexes_like_cv2(clip, cv2_means):
    assert len(cv2_means) == N
    with avdecode.KeyframeReader(clip, FPS, "DEFAULT", target_width=W) as r:
        got = list(r.frames())
    assert [i for i, _ in got] == list(range(N))
    _check_against_cv2(got, cv2_means)


def test_keyframes_only_is_a_correctly_indexed_subset(clip, cv2_means):
    with avdecode.KeyframeReader(clip, FPS, "NONKEY", target_width=W) as r:
        got = list(r.frames())
    idx = [i for i, _ in got]
    assert len(idx) == N // GOP
    assert np.all(np.diff(idx) == GOP)
    _check_against_cv2(got, cv2_means)


def test_downscale_and_coordinate_scale(clip):
    with avdecode.KeyframeReader(clip, FPS, "NONKEY", target_width=160) as r:
        idx, img = next(r.frames())
        assert img.shape == (120, 160, 3)
        assert (r.native_width, r.native_height) == (W, H)
        assert r.scale == pytest.approx((2.0, 2.0))


def test_never_upscales_and_keeps_height_even():
    assert avdecode.output_size(320, 240, 1280) == (320, 240)
    assert avdecode.output_size(3840, 2160, 960) == (960, 540)
    assert avdecode.output_size(1920, 1081, 960)[1] % 2 == 0


def test_does_not_use_frame_threading(clip):
    # Frame threads decode every packet and make skip_frame worthless
    # (measured 0.61x vs 0.14x realtime on the 4K clips).
    with avdecode.KeyframeReader(clip, FPS, "NONKEY") as r:
        assert str(r._s.thread_type).upper().endswith("SLICE")


def test_close_releases_the_file(clip, tmp_path):
    import shutil

    copy = tmp_path / "copy.mp4"
    shutil.copy(clip, copy)
    r = avdecode.KeyframeReader(copy, FPS, "NONKEY")
    next(r.frames())
    r.close()
    copy.unlink()                 # Windows refuses this while a handle is open


def test_unopenable_file_raises_the_fallback_error(tmp_path):
    bad = tmp_path / "not_a_video.mp4"
    bad.write_bytes(b"nope")
    with pytest.raises(avdecode.AVUnavailable):
        avdecode.KeyframeReader(bad, FPS)
