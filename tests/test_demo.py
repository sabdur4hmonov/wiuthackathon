"""demo/app.py must never crash on a user upload: a readable DemoError or a result."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))
app = pytest.importorskip("app")
cv2 = pytest.importorskip("cv2")


def _clip(path: Path, n: int, fps: float = 25.0, size=(160, 96)) -> Path:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    rng = np.random.default_rng(0)
    for _ in range(n):
        w.write(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    w.release()
    return path


def _bytes(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


@pytest.mark.parametrize("make, message", [
    (lambda d: _bytes(d / "empty.mp4", b""), "empty"),
    (lambda d: _bytes(d / "junk.mp4", b"not a video" * 50),
     "could not be opened|No frame could be decoded"),
    (lambda d: _clip(d / "one.mp4", 1), "too short"),
])
def test_bad_uploads_get_a_readable_error(tmp_path, make, message):
    with pytest.raises(app.DemoError, match=message):
        app.probe(str(make(tmp_path)))


def test_too_long_is_refused_before_any_work(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "MAX_SEC", 1.0)
    with pytest.raises(app.DemoError, match="accepts up to"):
        app.probe(str(_clip(tmp_path / "long.mp4", 50)))     # 2 s


def test_a_clip_from_another_scene_runs_without_the_rules(tmp_path):
    r = app.analyse(str(_clip(tmp_path / "noise.mp4", 50)), log=lambda m: None)
    assert r["events"] == []                                 # zones not applied
    assert "NOT run" in r["note"]
    assert len(r["risk"]) == 50
    for key in ("annotated", "json"):
        assert Path(r[key]).exists() and Path(r[key]).stat().st_size > 0
