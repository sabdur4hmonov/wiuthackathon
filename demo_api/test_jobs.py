import http.client
import io
import json
import struct
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from demo_api.app import make_handler
from demo_api.jobs import (
    MAX_BYTES, READ_CHUNK_BYTES, CapacityError, JobStore, UploadError,
    UploadTimeout,
    mp4_duration_seconds,
)
from demo_api.validation import OFFICIAL_CLASSES, PredictionError, sanitized_prediction
from evaluate import OFFICIAL_CLASSES as EVALUATOR_CLASSES, validate as evaluator_validate


def atom(kind, body):
    return struct.pack(">I4s", 8 + len(body), kind) + body


def small_mp4(seconds=10):
    movie = bytes(12) + struct.pack(">II", 1000, seconds * 1000)
    return atom(b"ftyp", b"isom0000") + atom(b"moov", atom(b"mvhd", movie))


def upload(store, filename="clip.mp4", blob=None):
    blob = small_mp4() if blob is None else blob
    return store.create(filename, io.BytesIO(blob), len(blob))


def await_status(store, job_id, wanted):
    for _ in range(100):
        job = store.get(job_id)
        if job is None or job["status"] == wanted:
            return job
        time.sleep(0.01)
    return store.get(job_id)


class FakeVerifier:
    def __init__(self, seconds=10):
        self.seconds = seconds

    def duration_seconds(self, video_path):
        assert video_path.is_file()
        return self.seconds


class FakeAdapter:
    def predict(self, video_path, filename):
        assert video_path.is_file()
        return {"team": "test", "videos": {filename: {"events": [], "risk": [[0, 0.0]]}}}


class FailingAdapter:
    def predict(self, video_path, filename):
        raise RuntimeError(f"private path: {video_path}")


class BlockingAdapter:
    def __init__(self):
        self.release = threading.Event()

    def predict(self, video_path, filename):
        self.release.wait(2)
        return FakeAdapter().predict(video_path, filename)


class InvalidAdapter:
    def predict(self, video_path, filename):
        return {"team": "test", "videos": {filename: {"events": []}}, "log": {"path": str(video_path)}}


class ObservingStream(io.BytesIO):
    def __init__(self, blob):
        super().__init__(blob)
        self.calls = 0

    def read(self, count=-1):
        assert 0 < count <= READ_CHUNK_BYTES
        self.calls += 1
        return super().read(count)

    def read1(self, count=-1):
        return self.read(count)


class TimeoutStream:
    def read(self, count):
        raise TimeoutError("socket timed out at C:\\private\\upload.mp4")


class SlowStream:
    def read(self, count):
        time.sleep(0.02)
        return b"x"


class PredictionValidationTests(unittest.TestCase):
    def test_classes_match_evaluator_and_normal_result(self):
        self.assertEqual(OFFICIAL_CLASSES, frozenset(EVALUATOR_CLASSES))
        raw = {"team": "test", "videos": {"clip.mp4": {"events": [[0, 2, "accident"], [1, 3, "near_miss"]]}}}
        clean = sanitized_prediction(raw, "clip.mp4")
        self.assertEqual(clean["videos"]["clip.mp4"]["risk"], [])
        self.assertNotIn("log", clean)
        self.assertEqual(evaluator_validate(clean)[0], [])

    def test_invalid_result_is_rejected(self):
        base = lambda events, risk=[]: {"videos": {"clip.mp4": {"events": events, "risk": risk}}}
        bad = [
            base([[1, 1, "accident"]]),
            base([[0, 1, "unknown"]]),
            base([[0, 2, "accident"], [1, 3, "accident"]]),
            base([], [[0, 1.1]]),
            base([], [[1, 0.5], [0, 0.5]]),
            base([], [[0, float("nan")]]),
            {**base([]), "log": {"path": "C:\\private"}},
            {"team": "C:\\private", **base([])},
            {"videos": {"other.mp4": {"events": []}}},
            {"videos": {"clip.mp4": {"events": [], "server_path": "private"}}},
        ]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(PredictionError):
                sanitized_prediction(raw, "clip.mp4")


class JobTests(unittest.TestCase):
    def test_duration_streaming_limits_and_cleanup(self):
        self.assertEqual(mp4_duration_seconds(io.BytesIO(small_mp4()), len(small_mp4())), 10)
        store = JobStore()
        try:
            for name, blob in (("../clip.mp4", small_mp4()), ("clip.mp4", small_mp4(121)), ("clip.mp4", b"not a video")):
                with self.assertRaises(UploadError):
                    upload(store, name, blob)
            blob = small_mp4() + atom(b"free", b"x" * 200_000)
            stream = ObservingStream(blob)
            job = store.create("clip.mp4", stream, len(blob))
            self.assertGreater(stream.calls, 1)
            self.assertEqual(job["status"], "awaiting_model")
            self.assertFalse((store.root / job["job_id"]).exists())
            self.assertNotIn(str(store.root), json.dumps(job))
            self.assertIsNone(store.result(job["job_id"]))
        finally:
            store.close()

    def test_adapter_requires_independent_duration_verifier(self):
        with self.assertRaises(ValueError):
            JobStore(FakeAdapter())
        store = JobStore(FakeAdapter(), FakeVerifier(121))
        try:
            with self.assertRaises(UploadError):
                upload(store)
            self.assertEqual(list(store.root.iterdir()), [])
        finally:
            store.close()

    def test_upload_timeout_and_retained_record_cap(self):
        store = JobStore(max_retained_jobs=1, ttl_sec=0.1, janitor_interval_sec=0.02, upload_timeout_sec=0.03)
        try:
            with self.assertRaises(UploadTimeout) as caught:
                store.create("clip.mp4", TimeoutStream(), len(small_mp4()))
            self.assertNotIn("C:\\private", str(caught.exception))
            self.assertEqual(list(store.root.iterdir()), [])
            with self.assertRaises(UploadTimeout):
                store.create("clip.mp4", SlowStream(), len(small_mp4()))
            self.assertEqual(list(store.root.iterdir()), [])
            first = upload(store)
            with self.assertRaises(CapacityError):
                upload(store)
            time.sleep(0.16)
            self.assertIsNone(store.get(first["job_id"]))
            second = upload(store)
            self.assertEqual(second["status"], "awaiting_model")
        finally:
            store.close()

    def test_failure_and_invalid_result_never_expose_paths(self):
        for adapter in (FailingAdapter(), InvalidAdapter()):
            store = JobStore(adapter, FakeVerifier())
            try:
                job = upload(store)
                status = await_status(store, job["job_id"], "failed")
                self.assertEqual(status["status"], "failed")
                self.assertNotIn(str(store.root), json.dumps(status))
                self.assertFalse((store.root / job["job_id"]).exists())
                self.assertIsNone(store.result(job["job_id"]))
            finally:
                store.close()

    def test_concurrent_limit_and_expiry(self):
        adapter = BlockingAdapter()
        store = JobStore(adapter, FakeVerifier(), max_active_jobs=1, ttl_sec=0.15, janitor_interval_sec=0.02)
        try:
            job = upload(store)
            with self.assertRaises(CapacityError):
                upload(store)
            time.sleep(0.25)
            self.assertIsNone(store.get(job["job_id"]))
            self.assertFalse((store.root / job["job_id"]).exists())
        finally:
            adapter.release.set()
            time.sleep(0.03)
            store.close()

    def test_http_status_and_adapter_result(self):
        store = JobStore(FakeAdapter(), FakeVerifier())
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/api/health") as response:
                self.assertTrue(json.load(response)["model_connected"])
                self.assertIsNone(response.headers.get("Access-Control-Allow-Origin"))
            request = Request(base + "/api/jobs", data=small_mp4(), headers={"Content-Type": "video/mp4", "X-File-Name": "test.mp4"}, method="POST")
            with urlopen(request) as response:
                job = json.load(response)
                self.assertEqual(response.status, 202)
            for _ in range(100):
                with urlopen(base + "/api/jobs/" + job["job_id"]) as response:
                    status = json.load(response)
                if status["status"] == "completed":
                    break
                time.sleep(0.01)
            self.assertEqual(status["status"], "completed")
            self.assertFalse((store.root / job["job_id"]).exists())
            with urlopen(base + "/api/jobs/" + job["job_id"] + "/result") as response:
                self.assertEqual(json.load(response)["videos"]["test.mp4"]["events"], [])
            with self.assertRaises(HTTPError) as caught:
                urlopen(base + "/api/jobs/not-a-job")
            self.assertEqual(caught.exception.code, 404)
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
            connection.putrequest("POST", "/api/jobs")
            connection.putheader("Content-Length", str(MAX_BYTES + 1))
            connection.endheaders()
            response = connection.getresponse()
            self.assertEqual(response.status, 413)
            response.read()
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            store.close()

    def test_disconnected_http_result_is_unavailable(self):
        store = JobStore()
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            request = Request(base + "/api/jobs", data=small_mp4(), headers={"Content-Type": "video/mp4", "X-File-Name": "test.mp4"}, method="POST")
            with urlopen(request) as response:
                job = json.load(response)
            self.assertEqual(job["status"], "awaiting_model")
            self.assertFalse((store.root / job["job_id"]).exists())
            with self.assertRaises(HTTPError) as caught:
                urlopen(base + "/api/jobs/" + job["job_id"] + "/result")
            self.assertEqual(caught.exception.code, 409)
        finally:
            server.shutdown()
            server.server_close()
            store.close()


if __name__ == "__main__":
    unittest.main()
