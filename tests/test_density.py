"""CP4 budget policy on the keyframe path (src/perception.py + src/budget.py).

* The keyframe baseline always completes, whatever the budget says.
* Density beyond it is bought only with MEASURED Part B headroom, and dropped
  back to keyframes when the dense pass projects past part_a_hard.
"""
import numpy as np
import pytest

from src import avdecode, perception
from src.budget import Budget

pytestmark = pytest.mark.skipif(not avdecode.is_available(), reason="PyAV not installed")
pytest.importorskip("ultralytics")
pytest.importorskip("lap")

N, FPS, GOP = 120, 25.0, 15                   # 8 GOPs, IBBP like the real clips


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    import av

    p = tmp_path_factory.mktemp("density") / "gop15.mp4"
    with av.open(str(p), "w") as out:
        s = out.add_stream("libx264", rate=int(FPS))
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        s.codec_context.gop_size = GOP
        # Plain IBBP (no B-pyramid, no adaptive B): B-frames are never
        # references, as on the real XAVC clips, so NONREF decodes I+P only.
        s.codec_context.options = {"g": str(GOP), "keyint_min": str(GOP),
                                   "x264-params": "bframes=2:b-pyramid=none:b-adapt=0:scenecut=0"}
        rng = np.random.default_rng(0)
        for _ in range(N):
            img = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
                out.mux(pkt)
        for pkt in s.encode():
            out.mux(pkt)
    return p


class _NoBoxes:
    boxes = None


class _FakeModel:
    """Counts frames; detects nothing. The policy, not the detector, is under test."""

    def __init__(self):
        self.frames = 0

    def predict(self, frame, **kw):
        self.frames += 1
        return [_NoBoxes()]


@pytest.fixture
def model(monkeypatch):
    m = _FakeModel()
    monkeypatch.setattr(perception, "load_model", lambda cfg=None: m)
    return m


def _budget(clip):
    return Budget.for_video(clip)


def test_the_keyframe_baseline_ignores_a_budget_stop(clip, model):
    b = _budget(clip)
    b.force_stop("no time at all")                 # e.g. Part B measured over budget
    t = perception.run_perception(clip, b, verbose=False)
    assert model.frames == N // GOP                # every keyframe, none skipped
    assert t.complete is True
    assert t.frame_stride == GOP


def test_no_density_without_a_real_part_b_measurement(clip, model):
    b = _budget(clip)                              # fixed fallback reserve only
    t = perception.run_perception(clip, b, verbose=False)
    assert t.frame_stride == GOP
    assert model.frames == N // GOP


def test_density_when_the_measured_part_b_leaves_room(clip, model, monkeypatch):
    b = _budget(clip)
    b.set_measured_part_b(0.0)                     # a light clip: Part B is cheap
    assert b.headroom_x() >= b.cfg.dense_min_headroom_x
    # The choice is under test, not the drop-back (its own test below): on a
    # slow first run this 4.8 s clip can project past its budget and drop back.
    monkeypatch.setattr(b, "project_overrun", lambda *a, **k: False)
    t = perception.run_perception(clip, b, verbose=False)
    assert t.frame_stride == 3                     # I+P: one frame in three
    assert model.frames > 3 * (N // GOP)
    assert any("dense" in (s.note or "") for s in b.stages if s.name == "density")


def test_no_density_when_part_b_leaves_no_room(clip, model):
    b = _budget(clip)
    b.set_measured_part_b(2.5 * b.duration)        # the 4K clips on the 8-core box
    assert b.headroom_x() < b.cfg.dense_min_headroom_x
    t = perception.run_perception(clip, b, verbose=False)
    assert t.frame_stride == GOP


def test_dense_pass_drops_back_to_keyframes_and_still_finishes(clip, model, monkeypatch):
    b = _budget(clip)
    b.set_measured_part_b(0.0)
    monkeypatch.setattr(b, "project_overrun", lambda *a, **k: True)
    t = perception.run_perception(clip, b, verbose=False)
    assert t.complete is True
    # Part of the clip was sampled at keyframe rate, so the table reports the
    # sparse step: direction rules must not trust it.
    assert t.frame_stride >= GOP
    assert any("keyframes from here" in (s.note or "") for s in b.stages)


def test_headroom_is_none_without_a_measurement(clip):
    b = _budget(clip)
    assert b.headroom_x() is None
    b.set_measured_part_b(b.duration)
    h = b.headroom_x()
    expected = (b.total_budget - b.duration * b.cfg.part_b_measured_safety
                - b.safety_margin - b.elapsed()) / b.duration
    assert h == pytest.approx(expected, abs=0.05)


def test_alignment_is_never_skipped_for_budget(clip, model):
    b = _budget(clip)
    b.force_stop("no time at all")
    t = perception.run_perception(clip, b, verbose=False)
    assert t.pose is not None
    assert not t.pose["reason"].startswith("skipped")
    assert t.pose["frames"] > 0                    # frames were matched in-loop


def test_fp16_uses_the_current_ultralytics_name():
    from src.config import CFG

    kw = perception.build_track_kwargs(CFG.perception, "0")
    assert "half" not in kw                        # deprecated: warns every call
    assert kw.get("quantize") == 16
    assert "quantize" not in perception.build_track_kwargs(CFG.perception, "cpu")


@pytest.fixture(scope="module")
def long_gop_clip(tmp_path_factory):
    """What phones and re-encodes produce: one keyframe every 250 frames."""
    import av

    p = tmp_path_factory.mktemp("longgop") / "g250.mp4"
    with av.open(str(p), "w") as out:
        s = out.add_stream("libx264", rate=int(FPS))
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        s.codec_context.options = {"g": "250", "x264-params": "scenecut=0"}
        rng = np.random.default_rng(1)
        for _ in range(250):
            img = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
                out.mux(pkt)
        for pkt in s.encode():
            out.mux(pkt)
    return p


def test_long_gop_is_sampled_every_half_second_not_every_keyframe(long_gop_clip, model, monkeypatch):
    b = _budget(long_gop_clip)
    # The sample rate is under test; a 10 s test clip can otherwise project
    # past its own tiny budget on a slow runner (the next test covers that).
    monkeypatch.setattr(b, "project_overrun", lambda *a, **k: False)
    monkeypatch.setattr(b, "should_stop", lambda: False)
    t = perception.run_perception(long_gop_clip, b, verbose=False)
    step = int(round(perception.LONG_GOP_SAMPLE_SEC * FPS))
    assert t.frame_stride == step                  # not 250: one keyframe would be useless
    assert model.frames == pytest.approx(250 / step, abs=1)
    assert t.complete is True
    assert any("long GOP" in (s.note or "") for s in b.stages)


def test_long_gop_full_decode_respects_the_budget(long_gop_clip, model):
    b = _budget(long_gop_clip)
    b.force_stop("no time")                        # unlike the keyframe baseline, this mode stops
    t = perception.run_perception(long_gop_clip, b, verbose=False)
    assert model.frames == perception._BUDGET_CHECK_EVERY
    assert t.complete is False


def test_reader_reports_the_gop(clip, long_gop_clip):
    with avdecode.KeyframeReader(clip, FPS) as r:
        assert r.keyframe_gap == GOP
        assert len(list(r.frames())) == N // GOP   # the probe seeks back: nothing lost
    with avdecode.KeyframeReader(long_gop_clip, FPS) as r:
        assert r.keyframe_gap is None              # < 2 keyframes in the probe window
