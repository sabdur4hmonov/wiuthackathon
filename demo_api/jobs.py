"""Bounded local upload jobs with an isolated, validated model-adapter seam."""

from __future__ import annotations

import math
import shutil
import struct
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import BinaryIO, Callable, Protocol

from .validation import safe_filename, sanitized_prediction

MAX_BYTES = 100 * 1024 * 1024
MAX_DURATION_SEC = 120.0
MAX_UPLOAD_SEC = 60.0
SOCKET_TIMEOUT_SEC = 10.0
READ_CHUNK_BYTES = 64 * 1024
MAX_ACTIVE_JOBS = 2
MAX_RETAINED_JOBS = 32
JOB_TTL_SEC = 15 * 60.0


class UploadError(ValueError):
    pass


class UploadTimeout(UploadError):
    pass


class CapacityError(UploadError):
    pass


class PredictionAdapter(Protocol):
    def predict(self, video_path: Path, filename: str) -> dict:
        """Return an official prediction document for this uploaded file."""


class DurationVerifier(Protocol):
    def duration_seconds(self, video_path: Path) -> float:
        """Independently verify actual duration from decoded video frames."""


def _boxes(stream: BinaryIO, start: int, end: int):
    cursor = start
    while cursor + 8 <= end:
        stream.seek(cursor)
        header = stream.read(8)
        if len(header) != 8:
            return
        size, kind = struct.unpack(">I4s", header)
        header_size = 8
        if size == 1:
            extended = stream.read(8)
            if len(extended) != 8:
                return
            size = struct.unpack(">Q", extended)[0]
            header_size = 16
        elif size == 0:
            size = end - cursor
        if size < header_size or size > end - cursor:
            return
        yield kind, cursor + header_size, cursor + size
        cursor += size


def mp4_duration_seconds(stream: BinaryIO, size: int) -> float:
    """Cheap container-header check; NOT authoritative decoded duration."""
    stream.seek(0)
    if size < 12 or stream.read(8)[4:8] != b"ftyp":
        raise UploadError("File is not a supported MP4 container.")
    for kind, start, end in _boxes(stream, 0, size):
        if kind != b"moov":
            continue
        for child, body, tail in _boxes(stream, start, end):
            if child != b"mvhd" or body + 4 > tail:
                continue
            stream.seek(body)
            version = stream.read(1)[0]
            if version == 0 and body + 20 <= tail:
                stream.seek(body + 12)
                timescale, duration = struct.unpack(">II", stream.read(8))
            elif version == 1 and body + 32 <= tail:
                stream.seek(body + 20)
                timescale, duration = struct.unpack(">IQ", stream.read(12))
            else:
                break
            if timescale == 0 or duration in (0, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF):
                break
            return duration / timescale
    raise UploadError("MP4 duration is unavailable; upload a finalized .mp4 with a movie header.")


class JobStore:
    def __init__(
        self,
        adapter: PredictionAdapter | None = None,
        duration_verifier: DurationVerifier | None = None,
        *,
        max_active_jobs: int = MAX_ACTIVE_JOBS,
        max_retained_jobs: int = MAX_RETAINED_JOBS,
        ttl_sec: float = JOB_TTL_SEC,
        janitor_interval_sec: float = 15.0,
        upload_timeout_sec: float = MAX_UPLOAD_SEC,
    ):
        if adapter is not None and duration_verifier is None:
            raise ValueError("Model execution requires an independent decoded-duration verifier.")
        if max_active_jobs < 1 or max_retained_jobs < 1 or ttl_sec <= 0 or janitor_interval_sec <= 0 or upload_timeout_sec <= 0:
            raise ValueError("Job limits and expiry must be positive.")
        self.adapter = adapter
        self.duration_verifier = duration_verifier
        self._temporary = tempfile.TemporaryDirectory(prefix="wiut-demo-")
        self.root = Path(self._temporary.name)
        self._jobs: dict[str, dict] = {}
        self._results: dict[str, dict] = {}
        self._created: dict[str, float] = {}
        self._cleanup_pending: set[str] = set()
        self._lock = threading.Lock()
        self._active = threading.BoundedSemaphore(max_active_jobs)
        self._max_retained_jobs = max_retained_jobs
        self._ttl_sec = ttl_sec
        self._upload_timeout_sec = upload_timeout_sec
        self._stop = threading.Event()
        self._janitor = threading.Thread(target=self._janitor_loop, args=(janitor_interval_sec,), daemon=True)
        self._janitor.start()

    def _janitor_loop(self, interval: float):
        while not self._stop.wait(interval):
            self.sweep()

    def _cleanup_dir(self, job_id: str):
        folder = self.root / job_id
        try:
            shutil.rmtree(folder)
        except FileNotFoundError:
            pass
        except OSError:
            # A running adapter may hold the file on Windows. Retry later.
            with self._lock:
                self._cleanup_pending.add(job_id)
            return False
        with self._lock:
            self._cleanup_pending.discard(job_id)
        return True

    def sweep(self):
        now = time.monotonic()
        with self._lock:
            expired = [job_id for job_id, created in self._created.items() if now - created >= self._ttl_sec]
            for job_id in expired:
                self._created.pop(job_id, None)
                self._jobs.pop(job_id, None)
                self._results.pop(job_id, None)
            cleanup = set(expired) | set(self._cleanup_pending)
        for job_id in cleanup:
            self._cleanup_dir(job_id)

    def close(self):
        self._stop.set()
        self._janitor.join(timeout=2)
        self._temporary.cleanup()

    def create(
        self,
        filename: str,
        stream: BinaryIO,
        length: int,
        *,
        set_read_timeout: Callable[[float], None] | None = None,
    ) -> dict:
        if not safe_filename(filename):
            raise UploadError("Filename must be a simple .mp4 name.")
        if length <= 0 or length > MAX_BYTES:
            raise UploadError("Upload must be a nonempty MP4 no larger than 100 MiB.")
        self.sweep()
        if not self._active.acquire(blocking=False):
            raise CapacityError("Demo is busy; retry after an active upload or job finishes.")

        job_id = uuid.uuid4().hex
        folder = self.root / job_id
        worker_started = False
        try:
            with self._lock:
                if len(self._jobs) >= self._max_retained_jobs:
                    raise CapacityError("Demo job list is full; retry after older jobs expire.")
                self._jobs[job_id] = {
                    "job_id": job_id,
                    "status": "uploading",
                    "filename": filename,
                    "result_available": False,
                    "message": "Upload in progress.",
                }
                self._created[job_id] = time.monotonic()
            folder.mkdir()
            video_path = folder / "upload.mp4"
            remaining = length
            deadline = time.monotonic() + self._upload_timeout_sec
            read_chunk = getattr(stream, "read1", stream.read)
            with video_path.open("xb") as output:
                while remaining:
                    time_left = deadline - time.monotonic()
                    if time_left <= 0:
                        raise UploadTimeout("Upload timed out; retry with a shorter video.")
                    if set_read_timeout is not None:
                        set_read_timeout(min(SOCKET_TIMEOUT_SEC, time_left))
                    try:
                        chunk = read_chunk(min(READ_CHUNK_BYTES, remaining))
                    except TimeoutError as error:
                        raise UploadTimeout("Upload timed out; retry with a shorter video.") from error
                    if time.monotonic() > deadline:
                        raise UploadTimeout("Upload timed out; retry with a shorter video.")
                    if not chunk:
                        raise UploadError("Upload ended before the declared length.")
                    output.write(chunk)
                    remaining -= len(chunk)
            with video_path.open("rb") as uploaded:
                header_duration = mp4_duration_seconds(uploaded, length)
            if header_duration > MAX_DURATION_SEC:
                raise UploadError("Video exceeds the 120-second demo limit.")
            if self.duration_verifier is not None:
                try:
                    decoded_duration = self.duration_verifier.duration_seconds(video_path)
                except Exception as error:
                    raise UploadError("Decoded video duration could not be verified.") from error
                if not isinstance(decoded_duration, (int, float)) or isinstance(decoded_duration, bool) or not math.isfinite(decoded_duration) or decoded_duration <= 0:
                    raise UploadError("Decoded video duration could not be verified.")
                if decoded_duration > MAX_DURATION_SEC:
                    raise UploadError("Video exceeds the 120-second demo limit.")

            with self._lock:
                if job_id not in self._jobs:
                    raise UploadTimeout("Upload expired before completion.")
                job = self._jobs[job_id]
                job.update(
                    status="queued" if self.adapter else "awaiting_model",
                    message="Queued for model processing." if self.adapter else "Upload accepted; model adapter is not connected yet.",
                )
                response = dict(job)
            if self.adapter:
                thread = threading.Thread(target=self._run, args=(job_id, video_path, filename), daemon=True)
                thread.start()
                worker_started = True
            else:
                self._cleanup_dir(job_id)
            return response
        except Exception:
            with self._lock:
                self._jobs.pop(job_id, None)
                self._results.pop(job_id, None)
                self._created.pop(job_id, None)
            self._cleanup_dir(job_id)
            raise
        finally:
            if not worker_started:
                self._active.release()

    def _run(self, job_id: str, video_path: Path, filename: str):
        result = None
        try:
            with self._lock:
                if job_id not in self._jobs:
                    return
                self._jobs[job_id].update(status="running", message="Model processing in progress.")
            raw = self.adapter.predict(video_path, filename)
            result = sanitized_prediction(raw, filename)
        except Exception:
            # Model exceptions may contain local paths; never return them.
            pass
        finally:
            cleaned = self._cleanup_dir(job_id)
            with self._lock:
                if job_id in self._jobs:
                    if result is not None and cleaned:
                        self._results[job_id] = result
                        self._jobs[job_id].update(status="completed", result_available=True, message="Prediction ready.")
                    else:
                        self._jobs[job_id].update(status="failed", message="Processing failed or cleanup is pending; check private server diagnostics.")
            self._active.release()

    def get(self, job_id: str) -> dict | None:
        self.sweep()
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def result(self, job_id: str) -> dict | None:
        self.sweep()
        with self._lock:
            return self._results.get(job_id)
