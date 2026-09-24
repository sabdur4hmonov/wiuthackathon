"""The causality guard.

RiskEstimator.step() may use only frames it has already been handed. The tracks
cache is built by a pass that has seen the WHOLE video, so reading it inside
step() would leak future information into a score the metric treats as causal.
That is a disqualifying violation, not a bug -- so it is enforced structurally
(the risk package must not even be able to import the batch pipeline) and
checked here in a FRESH interpreter, because an in-process check would pass
trivially once pytest itself has imported the pipeline.
"""
from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
RISK_DIR = ROOT / "src" / "risk"

# Modules that have seen, or can reach, the whole video -- or can open one
# (PyAV and our keyframe reader).
FORBIDDEN_MODULES = {
    "src.perception", "src.cache", "src.pipeline", "src.tracks",
    "src.rules", "src.postprocess", "src.zones", "src.avdecode",
    "perception", "cache", "pipeline", "tracks", "rules", "postprocess", "av",
}
# The estimator runs its OWN detector on the frames it is handed (causal), so
# the source may import cv2 (resize) and ultralytics (YOLO) -- but only lazily:
# importing src.risk must not load them, and it must still work (quietly)
# when they cannot be imported at all. Opening a video is banned separately
# (VideoCapture / imread, below).
IMPORT_TIME_FORBIDDEN = FORBIDDEN_MODULES | {"cv2", "ultralytics"}


# ---------------------------------------------------------------------------
# static: what does the source import?
# ---------------------------------------------------------------------------
def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            # Resolve relative imports: level 1 inside src/risk is src.risk,
            # level 2 is src.
            if node.level == 0:
                names.add(node.module or "")
            else:
                base = "src.risk" if node.level == 1 else "src"
                names.add(f"{base}.{node.module}" if node.module else base)
    return {n for n in names if n}


@pytest.mark.parametrize("path", sorted(RISK_DIR.glob("*.py")), ids=lambda p: p.name)
def test_risk_source_imports_nothing_from_the_batch_pipeline(path):
    imported = _imported_names(path)
    banned = {n for n in imported
              if n in FORBIDDEN_MODULES
              or any(n.startswith(f + ".") for f in FORBIDDEN_MODULES)}
    assert not banned, (
        f"{path.name} imports {sorted(banned)}. The risk estimator must not be "
        f"able to reach anything built with future frames."
    )


def test_risk_source_never_opens_a_video_or_the_cache():
    """No VideoCapture, no cache access, anywhere in the package."""
    needles = ("VideoCapture", "cache", "load_zones", "TrackTable", "imread")
    for path in RISK_DIR.glob("*.py"):
        src = path.read_text(encoding="utf-8")
        # Strip docstrings/comments: the module docstring legitimately explains
        # the ban and names the very things it bans.
        tree = ast.parse(src)
        code_only = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Attribute, ast.Name)):
                code_only.append(getattr(node, "attr", None) or getattr(node, "id", ""))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                code_only.append(node.func.id)
        joined = " ".join(c for c in code_only if c)
        hits = [n for n in needles if n in joined]
        assert not hits, f"{path.name} references {hits} in executable code"


# ---------------------------------------------------------------------------
# dynamic: what actually gets imported, in a clean interpreter?
# ---------------------------------------------------------------------------
def test_importing_risk_does_not_pull_in_the_pipeline():
    """A fresh process imports src.risk and reports sys.modules."""
    code = textwrap.dedent(f"""
        import sys, json
        sys.path.insert(0, {str(ROOT)!r})
        import src.risk                      # noqa: F401
        banned = sorted(m for m in sys.modules if m in {sorted(IMPORT_TIME_FORBIDDEN)!r})
        print(json.dumps(banned))
    """)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    banned = eval(res.stdout.strip().splitlines()[-1])
    assert banned == [], (
        f"importing src.risk transitively loaded {banned}; the estimator must "
        f"be reachable without the batch pipeline"
    )


def test_estimator_runs_without_the_pipeline_importable():
    """Prove it works standalone, not just that it avoids the import."""
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})

        # Make the batch modules un-importable, then use the estimator anyway.
        class Blocker:
            BANNED = {sorted(IMPORT_TIME_FORBIDDEN)!r}
            def find_module(self, name, path=None):
                return self if name in self.BANNED else None
            def find_spec(self, name, path=None, target=None):
                if name in self.BANNED:
                    raise ImportError("blocked by the causality guard: " + name)
                return None
        sys.meta_path.insert(0, Blocker())

        import numpy as np
        from src.risk import RiskEstimator
        est = RiskEstimator()
        est.reset({{"video_id": "x.mp4", "fps": 25.0, "width": 64,
                   "height": 48, "n_frames": 100}})
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        scores = [est.step(frame, i / 25.0) for i in range(60)]
        assert all(0.0 <= s <= 1.0 for s in scores), scores
        print("OK")
    """)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=120)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "OK" in res.stdout


# ---------------------------------------------------------------------------
# behaviour
# ---------------------------------------------------------------------------
def test_default_is_quiet_well_below_the_alarm_threshold():
    """An alarm on an accident-free video is a pure false positive, pooled
    across the whole test set. Silence must be the default."""
    from src.risk import RiskEstimator

    est = RiskEstimator()
    est.reset({"video_id": "v.mp4", "fps": 25.0, "width": 64, "height": 48,
               "n_frames": 250})
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    scores = [est.step(frame, i / 25.0) for i in range(250)]
    assert max(scores) < 0.5, "would raise an alarm with no signal"
    assert max(scores) == 0.0


def test_reset_clears_state_between_videos():
    """The harness reuses one estimator per video via run_submission."""
    from src.risk import RiskEstimator

    est = RiskEstimator()
    est.reset({"video_id": "a.mp4", "fps": 25.0, "width": 8, "height": 8,
               "n_frames": 10})
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    for i in range(10):
        est.step(frame, i / 25.0)
    assert est.frame_count == 10
    est.reset({"video_id": "b.mp4", "fps": 30.0, "width": 8, "height": 8,
               "n_frames": 10})
    assert est.frame_count == 0
    assert est.last_score == 0.0
    assert est.fps == 30.0


def test_step_never_raises_on_hostile_input():
    """An exception costs the whole risk curve for that video."""
    from src.risk import RiskEstimator

    est = RiskEstimator()
    est.reset({})                                   # empty meta
    for bad in (None, np.zeros((0, 0, 3), dtype=np.uint8), "not a frame",
                np.zeros((4, 4), dtype=np.uint8)):
        s = est.step(bad, 0.0)
        assert isinstance(s, float) and 0.0 <= s <= 1.0


def test_step_returns_a_value_for_every_call():
    """The harness records one sample per frame regardless of internal striding."""
    from src.risk import RiskEstimator

    est = RiskEstimator()
    est.reset({"video_id": "v.mp4", "fps": 25.0, "width": 8, "height": 8,
               "n_frames": 37})
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    out = [est.step(frame, i / 25.0) for i in range(37)]
    assert len(out) == 37
    assert all(isinstance(s, float) for s in out)


def test_work_is_actually_strided():
    """The expensive path must run once every work_every_n_frames, not always."""
    from src.config import RiskConfig
    from src.risk import RiskEstimator

    est = RiskEstimator(RiskConfig(work_every_n_frames=5))
    calls = []
    est._score = lambda frame, t: (calls.append(t), 0.0)[1]
    est.reset({"video_id": "v.mp4", "fps": 25.0, "width": 8, "height": 8,
               "n_frames": 20})
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    for i in range(20):
        est.step(frame, i / 25.0)
    assert len(calls) == 4                           # frames 0, 5, 10, 15
