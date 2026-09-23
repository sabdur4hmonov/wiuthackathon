"""Optional, subprocess-isolated adapter for the unchanged organizer harness.

Constructing this adapter does not connect it to the demo API. The API's
normal startup deliberately remains model-disconnected.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

from .validation import UNSAFE_KEYS, safe_filename, sanitized_prediction

HARNESS_TIMEOUT_SEC = 390.0  # 3 * 120-second official budget plus startup margin.
MAX_OUTPUT_BYTES = 8 * 1024 * 1024


class HarnessAdapterError(ValueError):
    """Safe, path-free error at the subprocess boundary."""


def _unique_safe_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result or key in UNSAFE_KEYS:
            raise HarnessAdapterError("Model returned invalid prediction data.")
        result[key] = value
    return result


def _reject_constant(_value: str):
    raise HarnessAdapterError("Model returned invalid prediction data.")


def sanitized_harness_result(raw: object, source_filename: str, display_filename: str) -> dict:
    """Accept one error-free harness result; never forward diagnostics."""
    if not safe_filename(source_filename) or not safe_filename(display_filename):
        raise HarnessAdapterError("Model returned invalid prediction data.")
    if not isinstance(raw, dict) or "log" not in raw or set(raw) - {"team", "videos", "log"}:
        raise HarnessAdapterError("Model returned invalid prediction data.")
    if not isinstance(raw["log"], dict) or set(raw["log"]) != {source_filename}:
        raise HarnessAdapterError("Model returned invalid prediction data.")
    videos = raw.get("videos")
    if not isinstance(videos, dict) or set(videos) != {source_filename}:
        raise HarnessAdapterError("Model returned invalid prediction data.")
    log_entry = raw["log"][source_filename]
    if not isinstance(log_entry, dict) or log_entry.get("errors") != []:
        # The runner can turn exceptions and overruns into empty results.
        raise HarnessAdapterError("Model processing failed.")
    candidate = {"videos": {display_filename: videos[source_filename]}}
    if "team" in raw:
        candidate["team"] = raw["team"]
    return sanitized_prediction(candidate, display_filename)


class HarnessAdapter:
    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        harness_path: Path | None = None,
        timeout_sec: float = HARNESS_TIMEOUT_SEC,
        team: str = "wiut-cv",
    ):
        if not math.isfinite(timeout_sec) or timeout_sec <= 0:
            raise ValueError("Harness timeout must be positive.")
        if not isinstance(team, str) or not team or len(team) > 100 or any(c in team for c in "/\\:\r\n"):
            raise ValueError("Invalid team name.")
        self.repo_root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
        self.harness_path = (harness_path or self.repo_root / "run_submission.py").resolve()
        self.timeout_sec = timeout_sec
        self.team = team

    def predict(self, video_path: Path, filename: str) -> dict:
        if not safe_filename(filename):
            raise HarnessAdapterError("Model returned invalid prediction data.")
        video_path = Path(video_path).resolve()
        output_path = video_path.parent / "harness_predictions.json"
        command = [
            sys.executable, str(self.harness_path), "--videos", str(video_path),
            "--out", str(output_path), "--team", self.team,
        ]
        try:
            if not video_path.is_file() or not self.harness_path.is_file():
                raise HarnessAdapterError("Model processing failed.")
            finished = subprocess.run(
                command,
                cwd=self.repo_root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self.timeout_sec,
                check=False,
                shell=False,
            )
            if finished.returncode != 0 or not output_path.is_file() or output_path.stat().st_size > MAX_OUTPUT_BYTES:
                raise HarnessAdapterError("Model processing failed.")
            raw = json.loads(
                output_path.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_safe_object,
                parse_constant=_reject_constant,
            )
            return sanitized_harness_result(raw, video_path.name, filename)
        except Exception as error:
            # subprocess, JSON, and filesystem exceptions can contain paths.
            raise HarnessAdapterError("Model processing failed or returned invalid predictions.") from error
        finally:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                pass
