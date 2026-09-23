"""FFmpeg-pipe decoder tests.

This module exists to speed up Part A on the real camera format (XAVC H.264
High 4:2:2, 10-bit, 3840x2160). It must never be a correctness dependency: any
failure here has to fall back to cv2, exactly as CP0's cache and zones already
degrade rather than break the pipeline. These tests use only small, fast local
clips -- never the real 4K sample -- so the suite stays quick.
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.ffdecode import (FFmpegFrameReader, FFmpegUnavailable, ffmpeg_exe,
                          is_available, scaled_size)


# ---------------------------------------------------------------------------
# scaled_size
# ---------------------------------------------------------------------------
def test_scaled_size_preserves_aspect_ratio():
    w, h = scaled_size(3840, 2160, 1280)
    assert w == 1280
    assert h == pytest.approx(720, abs=1)


def test_scaled_size_is_always_even():
    for target in (1279, 1280, 1281, 961, 641):
        w, h = scaled_size(3840, 2160, target)
        assert w % 2 == 0
        assert h % 2 == 0


def test_scaled_size_never_below_two():
    w, h = scaled_size(3840, 2160, 1)
    assert w >= 2
    assert h >= 2


def test_scaled_size_rejects_bad_source():
    with pytest.raises(ValueError):
        scaled_size(0, 100, 640)
    with pytest.raises(ValueError):
        scaled_size(100, -5, 640)


def test_scaled_size_matches_a_non_16_9_source():
    w, h = scaled_size(1000, 1000, 500)
    assert (w, h) == (500, 500)


# ---------------------------------------------------------------------------
# ffmpeg_exe / is_available
# ---------------------------------------------------------------------------
def test_ffmpeg_exe_finds_a_binary():
    """imageio-ffmpeg is a declared dependency; this must succeed in CI."""
    exe = ffmpeg_exe()
    assert exe
    from pathlib import Path
    assert Path(exe).exists()


def test_is_available_agrees_with_ffmpeg_exe():
    assert is_available() is True


# ---------------------------------------------------------------------------
# FFmpegFrameReader, against a small local clip
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_clip(tmp_path_factory):
    p = tmp_path_factory.mktemp("ffdecode") / "clip.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (320, 240))
    rng = np.random.default_rng(0)
    for i in range(30):
        frame = np.full((240, 320, 3), 0, dtype=np.uint8)
        # A moving block so consecutive frames are genuinely different --
        # otherwise a broken pipe silently returning frame 0 forever would
        # still "work" by accident.
        x = (i * 8) % 280
        frame[80:160, x:x + 40] = (0, 0, 255)
        w.write(frame)
    w.release()
    return p


def test_reads_the_right_number_of_frames(small_clip):
    with FFmpegFrameReader(str(small_clip), 320, 240) as reader:
        frames = list(reader.frames())
    assert len(frames) == 30
    assert [idx for idx, _f in frames] == list(range(30))


def test_frames_have_the_requested_shape_and_dtype(small_clip):
    with FFmpegFrameReader(str(small_clip), 320, 240) as reader:
        _idx, frame = next(reader.frames())
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8


def test_scaling_changes_the_frame_shape(small_clip):
    w, h = scaled_size(320, 240, 160)
    with FFmpegFrameReader(str(small_clip), w, h) as reader:
        _idx, frame = next(reader.frames())
    assert frame.shape == (h, w, 3)


def test_consecutive_frames_are_not_identical(small_clip):
    """Guards against a reader that silently repeats one frame forever."""
    with FFmpegFrameReader(str(small_clip), 320, 240) as reader:
        frames = [frame for _idx, frame in
                  (next(reader.frames()) for _ in range(5))]
    assert not np.array_equal(frames[0], frames[1])
    assert not np.array_equal(frames[1], frames[2])


def test_pixel_content_is_plausible_not_garbage(small_clip):
    """The block we drew must actually be visible in the decoded frame."""
    with FFmpegFrameReader(str(small_clip), 320, 240) as reader:
        _idx, frame = next(reader.frames())
    # BGR red block: high blue-channel-index-2 (red in BGR is channel 2)
    # value somewhere in the frame, not just uniform background.
    assert frame[..., 2].max() > 100
    assert frame.std() > 1.0


def test_reader_used_outside_context_manager_raises(small_clip):
    reader = FFmpegFrameReader(str(small_clip), 320, 240)
    with pytest.raises(RuntimeError):
        reader.read()


def test_reader_on_a_nonexistent_file_yields_no_frames(tmp_path):
    missing = tmp_path / "does_not_exist.mp4"
    with FFmpegFrameReader(str(missing), 64, 64) as reader:
        frames = list(reader.frames())
    assert frames == []


def test_reader_exit_is_safe_to_call_without_enter():
    reader = FFmpegFrameReader("whatever.mp4", 64, 64)
    reader.__exit__(None, None, None)     # must not raise


def test_reader_can_be_reused_as_a_fresh_context(small_clip):
    """Same object, entered twice sequentially -- both passes must agree."""
    reader = FFmpegFrameReader(str(small_clip), 320, 240)
    with reader:
        first = [frame for _idx, frame in
                 (next(reader.frames()) for _ in range(3))]
    with reader:
        second = [frame for _idx, frame in
                  (next(reader.frames()) for _ in range(3))]
    assert len(first) == len(second) == 3
    for a, b in zip(first, second):
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# unavailability degrades cleanly
# ---------------------------------------------------------------------------
def test_ffmpeg_exe_raises_a_typed_error_when_nothing_is_found(monkeypatch):
    import shutil as shutil_mod

    import src.ffdecode as ffdecode_mod

    def boom(*a, **k):
        raise ImportError("no imageio_ffmpeg")

    monkeypatch.setattr(ffdecode_mod, "shutil", shutil_mod)
    monkeypatch.setattr(shutil_mod, "which", lambda name: None)

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "imageio_ffmpeg":
            raise ImportError("blocked for this test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(FFmpegUnavailable):
        ffdecode_mod.ffmpeg_exe()
    assert ffdecode_mod.is_available() is False


def test_reader_enter_raises_ffmpeg_unavailable_when_no_binary(monkeypatch, small_clip):
    import src.ffdecode as ffdecode_mod

    monkeypatch.setattr(ffdecode_mod, "ffmpeg_exe",
                        lambda: (_ for _ in ()).throw(FFmpegUnavailable("none")))
    reader = FFmpegFrameReader(str(small_clip), 320, 240)
    with pytest.raises(FFmpegUnavailable):
        reader.__enter__()
