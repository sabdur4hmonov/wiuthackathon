"""Central configuration: tunables, seeds and offline enforcement.

Everything that changes pipeline *behaviour* lives here so that the perception
cache can be keyed on a hash of it (see `perception_config_hash`).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = REPO_ROOT / "weights"
ZONES_PATH = REPO_ROOT / "config" / "zones.json"

SEED = 1337

# --------------------------------------------------------------------------
# offline enforcement
# --------------------------------------------------------------------------
# Ultralytics phones home in three ways: an AMP self-check that downloads
# yolo11n.pt, a version/analytics ping, and a font download inside its plotting
# helpers. All three are disabled here, and this runs at import time of any
# module that touches the detector. See scripts/offline_check.py for the proof.
_OFFLINE_ENV = {
    "YOLO_OFFLINE": "1",
    "ULTRALYTICS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TORCH_HOME": str(WEIGHTS_DIR / ".torch"),
    "NO_ALBUMENTATIONS_UPDATE": "1",
    "DO_NOT_TRACK": "1",
}


def enforce_offline() -> None:
    """Set the env vars and Ultralytics settings that stop any network call."""
    for k, v in _OFFLINE_ENV.items():
        os.environ.setdefault(k, v)
    try:  # ultralytics may not be installed in a metric-only environment
        from ultralytics.utils import SETTINGS  # type: ignore

        # `sync` sends analytics; `datasets_dir`/`weights_dir` are where it would
        # auto-download. Pin them into the repo so a miss fails loudly, offline.
        if SETTINGS.get("sync", False):
            SETTINGS["sync"] = False
        SETTINGS["weights_dir"] = str(WEIGHTS_DIR)
        SETTINGS["runs_dir"] = str(REPO_ROOT / ".runs")
    except Exception:
        pass


def seed_everything(seed: int = SEED) -> None:
    """Fix every RNG we can reach. Two runs must produce identical output."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# --------------------------------------------------------------------------
# budget
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class BudgetConfig:
    """Fractions of the video duration. The harness allows 3.0x for A+B."""

    harness_factor: float = 3.0      # run_submission.py TIME_FACTOR_DEFAULT
    part_a_target: float = 1.2       # aim to finish perception+rules by here
    part_a_hard: float = 1.5         # absolute stop for Part A
    part_b_reserve: float = 1.2      # headroom Part B's full decode pass needs
    safety_margin_sec: float = 2.0   # covers harness overhead outside our t0
    safety_margin_frac: float = 0.15  # ...but never more than this of the budget


# --------------------------------------------------------------------------
# perception
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PerceptionConfig:
    """Anything here changes the tracks, so it is part of the cache key."""

    weights: str = "yolo11s.pt"
    imgsz: int = 960                 # CCTV objects are small; 640 loses them
    conf: float = 0.20               # low: ByteTrack's second stage uses the tail
    iou: float = 0.70                # NMS IoU
    frame_stride: int = 2            # process every Nth frame
    max_det: int = 300
    half: bool = True                # fp16 on CUDA, ignored on CPU
    tracker: str = "bytetrack.yaml"
    # COCO ids we care about. 0 person, 1 bicycle, 2 car, 3 motorcycle,
    # 5 bus, 7 truck. Everything else is noise on a road camera.
    classes: tuple[int, ...] = (0, 1, 2, 3, 5, 7)
    # Kinematics are smoothed over this many *processed* frames before vx/vy.
    velocity_window: int = 5

    def hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class RiskConfig:
    """Part B. Must stay cheap: step() is called on EVERY frame."""

    work_every_n_frames: int = 5     # do real work 1 frame in N, else cache
    default_score: float = 0.0       # default-quiet: silence is free, alarms are not
    max_score: float = 1.0


@dataclass(frozen=True)
class PostConfig:
    min_duration_sec: float = 0.5    # drop blips
    merge_gap_sec: float = 1.0       # stitch fragments of the same class


@dataclass(frozen=True)
class Config:
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    post: PostConfig = field(default_factory=PostConfig)


CFG = Config()


def perception_config_hash() -> str:
    return CFG.perception.hash()
