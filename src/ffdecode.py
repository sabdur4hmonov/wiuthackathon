"""FFmpeg-pipe frame decoder for Stage 1 -- Part A only, OFF BY DEFAULT.

THE HYPOTHESIS THIS WAS BUILT TO TEST. The real camera format is XAVC: H.264
High 4:2:2, 10-bit (yuv422p10le), 3840x2160, ~140 Mbps. cv2's FFmpeg backend
decodes and colour-converts every PROCESSED frame at full 4K/10-bit resolution
before Stage 1 ever sees a pixel, even though the detector immediately resizes
to PerceptionConfig.imgsz and throws the rest away. The idea was to move that
resize INTO the decoder via ffmpeg's own `scale` filter, decoding once at the
target size instead of decoding 4K and discarding most of it.

MEASURED RESULT, 2026-09-23, real clip, this 8-core machine: the hypothesis
was WRONG. cv2 grab/retrieve at native 4K measured 81.6 ms/processed-frame;
this pipe, scaled to 640/960/1280 width, measured 81.9/93.8/104.0 ms/frame --
equal at best, worse at every wider scale (see PerceptionConfig.decoder in
src/config.py for the full comparison and the reasoning: H.264 decode of this
stream dominates and happens before any scale filter runs, and cv2's
grab()-only skip on non-stride frames is cheaper than this pipe re-paying full
cost on every frame). PerceptionConfig.decoder therefore defaults to "cv2" and
should stay that way unless re-measured on different footage.

THIS ONLY EVER HELPS PART A, never Part B. run_submission.py's run_risk
(unmodified, official stride=1) always calls cv2 VideoCapture.read() on the
ORIGINAL file for every frame; that decode is entirely outside our control and
this module does not touch it, cannot touch it, and was never meant to. The
measured cv2.read() floor on the real file (tools/bench.py --skip-part-a) is a
hard floor on Part B no matter what Stage 1 does -- see
src/budget.measure_part_b_floor for how that measurement feeds the time
budget instead of a guess.

Kept, not deleted, because it is fully opt-in, tested, and falls back to cv2
automatically on any failure -- it may still be worth revisiting for different
footage or with a `select` filter added to skip full processing on non-stride
frames, which the current implementation does not do.

The ffmpeg binary comes from the `imageio-ffmpeg` PyPI package (a wheel with a
bundled static binary) rather than assuming a system install, so this works
fully offline once the package is installed -- no weights/download.sh-style
one-time fetch needed, and no dependency on the evaluation box having ffmpeg
on PATH at all.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


class FFmpegUnavailable(RuntimeError):
    """No ffmpeg binary could be found. Callers should fall back to cv2."""


def ffmpeg_exe() -> str:
    """Locate an ffmpeg binary: imageio-ffmpeg's bundled one first, else PATH.

    imageio-ffmpeg is checked first because it works fully offline; a PATH
    ffmpeg is a bonus, not a requirement.
    """
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise FFmpegUnavailable(
        "no ffmpeg binary found: `pip install imageio-ffmpeg`, or put ffmpeg "
        "on PATH. Callers should catch this and fall back to the cv2 decoder."
    )


def is_available() -> bool:
    try:
        ffmpeg_exe()
        return True
    except FFmpegUnavailable:
        return False


def scaled_size(src_w: int, src_h: int, target_w: int) -> tuple[int, int]:
    """Aspect-preserving (width, height), both EVEN, for the scale filter.

    Computed in Python, not left to ffmpeg's `-2` auto-sizing: the exact
    output size must be known BEFORE the subprocess starts, because it fixes
    the exact byte size of every frame read from the pipe, and it is what gets
    stored as the TrackTable's width/height (the SCALED size, never the source
    4K size -- rules and zones.json both reason in the coordinate frame the
    tracks were actually produced in).
    """
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"bad source size {src_w}x{src_h}")
    w = max(2, int(target_w) - (int(target_w) % 2))
    h = round(w * src_h / src_w)
    h -= h % 2
    return w, max(2, h)


class FFmpegFrameReader:
    """Iterates scaled BGR24 frames from a video via an ffmpeg subprocess pipe.

    Frames are pre-scaled to (width, height) by ffmpeg's own `scale` filter --
    the SCALED size, not the source's. Use as a context manager:

        with FFmpegFrameReader(path, w, h) as reader:
            for idx, frame in reader.frames():
                ...

    No internal frame-stride skipping: the caller still decides which frames
    to run the detector on, exactly as the cv2 grab()/retrieve() loop does, so
    swapping decoders does not change budget accounting semantics. (Skipped
    frames DO still cost a scale+colour-convert+pipe-write here, unlike cv2's
    grab()-only skip -- see the module docstring in src/perception.py for the
    measured comparison and whether that gap matters in practice.)
    """

    def __init__(self, video_path: str | Path, width: int, height: int) -> None:
        self.video_path = str(video_path)
        self.width = int(width)
        self.height = int(height)
        self._frame_bytes = self.width * self.height * 3
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> "FFmpegFrameReader":
        exe = ffmpeg_exe()
        cmd = [
            exe, "-hide_banner", "-loglevel", "error",
            "-i", self.video_path,
            "-vf", f"scale={self.width}:{self.height}",
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "-vsync", "0",          # one output frame per input frame
            "-an", "-sn",           # no audio/subtitle/data streams
            "-",
        ]
        # A pipe buffer sized for a few frames avoids the writer (ffmpeg)
        # blocking on every single frame while we are busy running the
        # detector on the previous one.
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=self._frame_bytes * 4,
        )
        return self

    def __exit__(self, *exc) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdout:
                self._proc.stdout.close()
        except Exception:
            pass
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
        self._proc = None

    def read(self) -> np.ndarray | None:
        """One BGR frame, shape (height, width, 3), or None at EOF/error."""
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("FFmpegFrameReader used outside a `with` block")
        buf = self._proc.stdout.read(self._frame_bytes)
        if len(buf) < self._frame_bytes:
            return None
        return np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width, 3)

    def frames(self):
        """Yield (frame_idx, ndarray) for every source frame, in order."""
        idx = 0
        while True:
            frame = self.read()
            if frame is None:
                return
            yield idx, frame
            idx += 1
