"""Keyframe-only decode for Stage 1, through PyAV.

WHY. The harness's Part B pass decodes every frame of the original file with
cv2 (~2.3-2.5x realtime for the 4K 10-bit XAVC test footage on the 8-core
box). Stage 1 used cv2 too, and cv2.grab() still runs the H.264 decoder on
every frame -- stride only saved the colour conversion and the detector. Two
full decodes can never fit inside the 3x budget together.

HOW. The decoder's own skip_frame switch: "NONKEY" makes FFmpeg drop every
non-keyframe before decoding it. On the real clips (tools/gop_probe.py) the GOP
is a fixed 15 frames, so that is one frame every 0.5 s.

MEASURED, 2026-09-24, 8-core box, sample_004, first 60 s:
    NONKEY decode only                     0.124x realtime
    NONKEY + downscale to 1280 (default)   0.183x
    NONREF (I+P, 10/s) + downscale         0.532x
    DEFAULT (every frame), frame threads   ~0.59-0.75x
Threading matters and is counter-intuitive: with FRAME threading (what
thread_type="AUTO" selects), skip_frame saves NOTHING -- 0.61x for NONKEY vs
0.14x without it -- because every frame thread decodes its packet regardless.
SLICE threading and no threading measure the same (0.14x), so this module
uses SLICE and never AUTO/FRAME.

The downscale happens inside swscale (the reformat step) with AREA
interpolation: 9 ms/frame for 4K 10-bit 4:2:2 -> 960 wide, against ~30 ms for
the default bicubic path, and AREA is the right filter for a 3-4x shrink.

FRAME INDEX. Frames come out with presentation timestamps. The index reported
is round((pts - start) * time_base * fps) with the SAME fps the harness uses,
so it indexes the harness's own frame sequence. Verified against cv2 on
sample_004: the first keyframes land on indices 2, 17, 32, 47 (each GOP opens
with two B-frames in display order) and match cv2's frames there.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import av  # type: ignore
except Exception:  # noqa: BLE001 - PyAV missing => caller falls back to cv2
    av = None


class AVUnavailable(RuntimeError):
    """PyAV is missing, or cannot open this file."""


def is_available() -> bool:
    return av is not None


def output_size(width: int, height: int, target_width: int) -> tuple[int, int]:
    """(w, h) scaled to target_width, height even, never upscaled."""
    if width <= 0 or height <= 0:
        raise ValueError(f"bad frame size {width}x{height}")
    if target_width <= 0 or target_width >= width:
        return width, height - height % 2
    h = int(round(height * target_width / width / 2.0)) * 2
    return int(target_width), max(2, h)


class KeyframeReader:
    """Iterate (frame_index, bgr_image) over a video's decoded frames.

    skip_frame: "NONKEY" (keyframes only), "NONREF" (I+P), or "DEFAULT".
    Always call close() -- or use it as a context manager -- so the demuxer's
    file handle and decoder threads are gone before the harness starts its own
    Part B decode.
    """

    def __init__(self, path: str | Path, fps: float, skip_frame: str = "NONKEY",
                 target_width: int = 960):
        if av is None:
            raise AVUnavailable("PyAV is not installed")
        try:
            self._c = av.open(str(path))
        except Exception as e:  # noqa: BLE001
            raise AVUnavailable(f"cannot open {path}: {e}") from e
        try:
            self._s = self._c.streams.video[0]
        except Exception as e:  # noqa: BLE001
            self._c.close()
            raise AVUnavailable(f"no video stream in {path}: {e}") from e
        self._s.thread_type = "SLICE"          # NOT "AUTO": see module docstring
        self._s.codec_context.skip_frame = skip_frame
        self.fps = float(fps) if fps else float(self._s.average_rate or 25.0)
        self.native_width = int(self._s.codec_context.width)
        self.native_height = int(self._s.codec_context.height)
        self.width, self.height = output_size(self.native_width, self.native_height,
                                              target_width)
        self._tb = float(self._s.time_base)
        self._start = self._s.start_time or 0

    @property
    def scale(self) -> tuple[float, float]:
        """Multiply decoded-frame coordinates by this to get native pixels."""
        return self.native_width / self.width, self.native_height / self.height

    def set_skip_frame(self, mode: str) -> None:
        """Change what is decoded from the next packet on. Safe at any time for
        "NONKEY" (a keyframe needs nothing before it); switching to a denser
        mode only produces frames from the next keyframe on."""
        self._s.codec_context.skip_frame = mode

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        for f in self._c.decode(self._s):
            if f.pts is None:
                continue
            idx = int(round((f.pts - self._start) * self._tb * self.fps))
            img = f.to_ndarray(format="bgr24", width=self.width, height=self.height,
                               interpolation="AREA")
            yield idx, img

    def close(self) -> None:
        try:
            self._c.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "KeyframeReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
