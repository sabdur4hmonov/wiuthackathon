"""Future sample export and evidence-only EDA tests, using no real footage."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from demo_api.sample_export import export_sample
from eda.summary import event_distribution, main as eda_main, track_hooks, video_metadata
from evaluate import validate as official_validate


class SampleExportTests(unittest.TestCase):
    def test_export_strips_log_and_preserves_official_values(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "raw.json"
            output = Path(folder) / "sanitized.json"
            source.write_text(json.dumps({
                "team": "wiut-cv",
                "videos": {"clip.mp4": {"events": [[1.234, 2.345, "accident"]], "risk": [[0.0001, 0.7]]}},
                "log": {"clip.mp4": {"errors": [], "private": "C:/server/weights.pt"}},
            }), encoding="utf-8")
            export_sample(source, "clip.mp4", output)
            text = output.read_text(encoding="utf-8")
            document = json.loads(text)
            self.assertNotIn("log", document)
            self.assertNotIn("C:/server", text)
            self.assertEqual(document["videos"]["clip.mp4"]["events"][0][0], 1.234)
            self.assertEqual(official_validate(document)[0], [])
            self.assertEqual(event_distribution(output, "clip.mp4"), {"accident": 1})
            with self.assertRaises(FileExistsError):
                export_sample(source, "clip.mp4", output)

    def test_export_rejects_harness_error_and_duplicate_video_key(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "raw.json"
            output = Path(folder) / "sanitized.json"
            source.write_text(json.dumps({
                "videos": {"clip.mp4": {"events": [], "risk": []}},
                "log": {"clip.mp4": {"errors": ["model failed at C:/private"]}},
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                export_sample(source, "clip.mp4", output)
            self.assertFalse(output.exists())
            source.write_text('{"videos":{"clip.mp4":{"events":[]},"clip.mp4":{"events":[]}}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                export_sample(source, "clip.mp4", output)
            self.assertFalse(output.exists())


class FakeCapture:
    def __init__(self):
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == 1:
            return 2.0
        if prop == 2:
            return 4.0
        if prop == 3:
            return max(0, self.index - 1) * 500.0
        raise AssertionError(prop)

    def read(self):
        if self.index == 4:
            return False, None
        self.index += 1
        return True, types.SimpleNamespace(shape=(480, 640, 3))

    def release(self):
        self.released = True


class EdaTests(unittest.TestCase):
    def test_real_data_is_required_when_no_video_supplied(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["eda.summary"]), contextlib.redirect_stdout(output):
            self.assertEqual(eda_main(), 2)
        self.assertIn("real data required", output.getvalue())

    def test_metadata_from_decoded_frames_and_track_hooks(self):
        capture = FakeCapture()
        fake_cv2 = types.SimpleNamespace(CAP_PROP_FPS=1, CAP_PROP_FRAME_COUNT=2, CAP_PROP_POS_MSEC=3)
        with patch.dict(sys.modules, {"cv2": fake_cv2}):
            metadata = video_metadata(Path("not-bundled.mp4"), lambda _: capture)
        self.assertTrue(capture.released)
        self.assertEqual((metadata["width"], metadata["height"]), (640, 480))
        self.assertEqual(metadata["decoded_frame_count"], 4)
        self.assertEqual(metadata["decoded_duration_sec"], 2.0)
        with tempfile.TemporaryDirectory() as folder:
            sidecar = Path(folder) / "tracks.json"
            sidecar.write_text(json.dumps({"video": "clip.mp4", "points": [
                {"t_sec": 0.0, "track_id": 7, "x": 0.5, "y": 0.8},
                {"t_sec": 1.0, "track_id": 7, "x": 0.6, "y": 0.7},
            ]}), encoding="utf-8")
            hooks = track_hooks(sidecar, "clip.mp4", 2.0)
            self.assertEqual(hooks["unique_tracks_per_5s"], [[0, 1]])
            self.assertEqual(len(hooks["trajectories"][0]["points"]), 2)
