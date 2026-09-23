"""Fail-closed duration check based on frames actually decoded by OpenCV.

The decoder runs in a separate, time-limited Python process. Container movie
duration is deliberately not used as proof that a video is short enough.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Callable

from .jobs import MAX_DURATION_SEC

DECODE_TIMEOUT_SEC = 45.0
MAX_FPS = 240.0
MAX_DECODED_FRAMES = int(MAX_DURATION_SEC * MAX_FPS) + 1
MAX_FRAME_PIXELS = 4096 * 2160


class DurationVerificationError(ValueError):
    """The decoded duration cannot safely be established."""


def decode_duration_seconds(video_path: Path, capture_factory: Callable | None = None) -> float:
    """Decode to EOF and compare frame count and presentation timestamps.

    FPS and reported frame count are used only as cross-checks; neither the
    MP4 movie duration nor an unverified frame count establishes duration.
    A decoder unable to supply progressing timestamps fails closed.
    """
    import cv2

    cap = (capture_factory or cv2.VideoCapture)(str(video_path))
    try:
        if not cap.isOpened():
            raise DurationVerificationError("Decoded video duration could not be verified.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        reported_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or not 0 < fps <= MAX_FPS:
            raise DurationVerificationError("Decoded video duration could not be verified.")
        if not math.isfinite(reported_frames) or reported_frames < 0:
            raise DurationVerificationError("Decoded video duration could not be verified.")

        count = 0
        last_ms = -1.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
            if count > MAX_DECODED_FRAMES:
                raise DurationVerificationError("Video exceeds the 120-second demo limit.")
            if frame is None or len(frame.shape) < 2 or frame.shape[0] * frame.shape[1] > MAX_FRAME_PIXELS:
                raise DurationVerificationError("Decoded video duration could not be verified.")
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if not math.isfinite(pos_ms) or pos_ms < 0 or pos_ms < last_ms:
                raise DurationVerificationError("Decoded video duration could not be verified.")
            last_ms = pos_ms
            if max(count / fps, pos_ms / 1000.0 + 1.0 / fps) > MAX_DURATION_SEC + 1e-6:
                raise DurationVerificationError("Video exceeds the 120-second demo limit.")

        if count == 0 or (count > 1 and last_ms <= 0):
            raise DurationVerificationError("Decoded video duration could not be verified.")
        if reported_frames > 0 and abs(reported_frames - count) > 1:
            raise DurationVerificationError("Decoded video duration could not be verified.")
        return max(count / fps, last_ms / 1000.0 + 1.0 / fps)
    finally:
        cap.release()


class DecodedDurationVerifier:
    """Run the OpenCV decode in a killable process before model execution."""

    def __init__(self, timeout_sec: float = DECODE_TIMEOUT_SEC):
        if not math.isfinite(timeout_sec) or timeout_sec <= 0:
            raise ValueError("Decode timeout must be positive.")
        self.timeout_sec = timeout_sec

    def duration_seconds(self, video_path: Path) -> float:
        command = [sys.executable, "-m", "demo_api.duration", "--verify", str(video_path)]
        try:
            finished = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=self.timeout_sec,
                check=False,
                shell=False,
            )
            if finished.returncode != 0 or len(finished.stdout) > 128:
                raise DurationVerificationError("Decoded video duration could not be verified.")
            value = json.loads(finished.stdout)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0 or value > MAX_DURATION_SEC:
                raise DurationVerificationError("Decoded video duration could not be verified.")
            return float(value)
        except (OSError, subprocess.TimeoutExpired, ValueError, TypeError) as error:
            raise DurationVerificationError("Decoded video duration could not be verified.") from error


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=Path, required=True)
    args = parser.parse_args()
    try:
        value = decode_duration_seconds(args.verify)
    except Exception:
        # This process's stderr is suppressed by the API; never print paths.
        return 1
    print(json.dumps(value))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
