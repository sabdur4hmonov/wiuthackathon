"""Phase 2A boundary tests; no real model or camera footage is used."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from demo_api.app import build_store
from demo_api.duration import (
    DecodedDurationVerifier, DurationVerificationError, decode_duration_seconds,
)
from demo_api.harness_adapter import HarnessAdapter, HarnessAdapterError
from demo_api.jobs import JobStore


class FakeCapture:
    def __init__(self, count: int, *, fps: float = 25.0, reported: int | None = None,
                 timestamp_step_ms: float | None = None, opened: bool = True):
        self.count = count
        self.fps = fps
        self.reported = count if reported is None else reported
        self.step_ms = 1000 / fps if timestamp_step_ms is None else timestamp_step_ms
        self.opened = opened
        self.index = 0
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, prop):
        if prop == 1:
            return self.fps
        if prop == 2:
            return self.reported
        if prop == 3:
            return max(0, self.index - 1) * self.step_ms
        raise AssertionError(prop)

    def read(self):
        if self.index >= self.count:
            return False, None
        self.index += 1
        return True, types.SimpleNamespace(shape=(16, 16, 3))

    def release(self):
        self.released = True


class DurationTests(unittest.TestCase):
    def decode(self, cap: FakeCapture):
        fake_cv2 = types.SimpleNamespace(CAP_PROP_FPS=1, CAP_PROP_FRAME_COUNT=2, CAP_PROP_POS_MSEC=3)
        with patch.dict(sys.modules, {"cv2": fake_cv2}):
            try:
                return decode_duration_seconds(Path("not-a-real-file.mp4"), lambda _: cap)
            finally:
                self.assertTrue(cap.released)

    def test_exact_120_second_boundary_from_decoded_frames(self):
        self.assertAlmostEqual(self.decode(FakeCapture(3000)), 120.0)
        with self.assertRaises(DurationVerificationError):
            self.decode(FakeCapture(3001))

    def test_misleading_short_frame_metadata_and_long_timestamps(self):
        with self.assertRaises(DurationVerificationError):
            self.decode(FakeCapture(3001, reported=250))
        with self.assertRaises(DurationVerificationError):
            self.decode(FakeCapture(100, timestamp_step_ms=1300))

    def test_truncated_or_unverifiable_decode_fails_closed(self):
        for cap in (
            FakeCapture(100, reported=3000),
            FakeCapture(10, timestamp_step_ms=0),
            FakeCapture(0),
            FakeCapture(100, opened=False),
        ):
            with self.subTest(cap=cap), self.assertRaises(DurationVerificationError):
                self.decode(cap)

    def test_verifier_has_killable_timeout_and_no_shell(self):
        verifier = DecodedDurationVerifier(timeout_sec=0.1)
        with patch("demo_api.duration.subprocess.run", side_effect=subprocess.TimeoutExpired("decode", 0.1)) as run:
            with self.assertRaises(DurationVerificationError) as caught:
                verifier.duration_seconds(Path("C:/private/upload.mp4"))
        self.assertNotIn("private", str(caught.exception))
        self.assertIs(run.call_args.kwargs["shell"], False)
        self.assertEqual(run.call_args.kwargs["timeout"], 0.1)


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / "upload.mp4"
        self.video.write_bytes(b"test-only")
        self.harness = self.root / "fake_harness.py"

    def write_harness(self, document=None, *, sleep_sec=0):
        if sleep_sec:
            self.harness.write_text(f"import time\ntime.sleep({sleep_sec})\n", encoding="utf-8")
        else:
            self.harness.write_text(
                "import argparse, json\n"
                "p = argparse.ArgumentParser()\n"
                "p.add_argument('--videos')\n"
                "p.add_argument('--out')\n"
                "p.add_argument('--team')\n"
                "a = p.parse_args()\n"
                f"open(a.out, 'w', encoding='utf-8').write({json.dumps(json.dumps(document))})\n",
                encoding="utf-8",
            )

    def adapter(self, timeout_sec=2):
        return HarnessAdapter(repo_root=self.root, harness_path=self.harness, timeout_sec=timeout_sec)

    def test_strips_log_remaps_filename_and_sanitizes(self):
        self.write_harness({
            "team": "wiut-cv",
            "videos": {"upload.mp4": {"events": [[1, 2, "accident"]], "risk": [[0, 0.1]]}},
            "log": {"upload.mp4": {"errors": [], "private_path": "C:/server/model"}},
        })
        result = self.adapter().predict(self.video, "road.mp4")
        self.assertEqual(result["videos"]["road.mp4"]["events"], [[1.0, 2.0, "accident"]])
        self.assertNotIn("log", result)
        self.assertNotIn("C:/server", json.dumps(result))
        self.assertFalse((self.root / "harness_predictions.json").exists())

    def test_invalid_prediction_structures_are_rejected(self):
        cases = [
            {"videos": {"upload.mp4": {"events": [[1, 2, "unknown"]]}}},
            {"videos": {"upload.mp4": {"events": [[2, 1, "accident"]]}}},
            {"videos": {"upload.mp4": {"events": [], "risk": [[0, 2]]}}},
            {"videos": {"upload.mp4": {"events": [], "server_path": "C:/server"}}},
            {"videos": {"upload.mp4": {"events": []}}, "sensitive": "C:/server"},
            {"videos": {"upload.mp4": {"events": []}}, "log": ["unexpected"]},
            {"videos": {"upload.mp4": {"events": []}}, "log": {"upload.mp4": {"errors": ["private C:/server/model"]}}},
            {"videos": {"upload.mp4": {"events": [], "__proto__": {"admin": True}}}},
        ]
        for document in cases:
            with self.subTest(document=document):
                if "log" not in document:
                    document = {**document, "log": {"upload.mp4": {"errors": []}}}
                self.write_harness(document)
                with self.assertRaises(HarnessAdapterError):
                    self.adapter().predict(self.video, "road.mp4")
        duplicate_json = '{"videos":{"upload.mp4":{"events":[]},"upload.mp4":{"events":[]}}}'
        self.harness.write_text(
            "import argparse\np=argparse.ArgumentParser()\np.add_argument('--videos')\n"
            "p.add_argument('--out')\np.add_argument('--team')\na=p.parse_args()\n"
            f"open(a.out,'w').write({duplicate_json!r})\n",
            encoding="utf-8",
        )
        with self.assertRaises(HarnessAdapterError):
            self.adapter().predict(self.video, "road.mp4")
        self.harness.write_text(
            "import argparse\np=argparse.ArgumentParser()\np.add_argument('--videos')\n"
            "p.add_argument('--out')\np.add_argument('--team')\na=p.parse_args()\n"
            "open(a.out,'w').write('{malformed')\n",
            encoding="utf-8",
        )
        with self.assertRaises(HarnessAdapterError):
            self.adapter().predict(self.video, "road.mp4")

    def test_timeout_terminates_process_and_hides_paths(self):
        self.write_harness(sleep_sec=5)
        start = time.monotonic()
        with self.assertRaises(HarnessAdapterError) as caught:
            self.adapter(timeout_sec=0.1).predict(self.video, "road.mp4")
        self.assertLess(time.monotonic() - start, 2)
        self.assertNotIn(str(self.root), str(caught.exception))

    def test_no_shell_invocation(self):
        self.write_harness({"videos": {"upload.mp4": {"events": []}}, "log": {"upload.mp4": {"errors": []}}})
        original_run = subprocess.run
        calls = []

        def observe(*args, **kwargs):
            calls.append((args, kwargs))
            return original_run(*args, **kwargs)

        with patch("demo_api.harness_adapter.subprocess.run", side_effect=observe):
            self.adapter().predict(self.video, "road.mp4")
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][1]["shell"], False)
        self.assertEqual(calls[0][1]["timeout"], 2)

    def test_normal_startup_remains_disconnected(self):
        store = build_store()
        try:
            self.assertIsNone(store.adapter)
            self.assertIsNone(store.duration_verifier)
        finally:
            store.close()
        store = build_store(self.adapter())
        try:
            self.assertIsInstance(store, JobStore)
            self.assertIsInstance(store.duration_verifier, DecodedDurationVerifier)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
