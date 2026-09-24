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
    part_b_reserve: float = 1.2      # FALLBACK headroom, used only if the real
                                      # probe below never runs or fails. On the
                                      # 4K XAVC test footage this fixed 1.2x is
                                      # wrong -- see part_b_measured_safety.
    safety_margin_sec: float = 2.0   # covers harness overhead outside our t0
    safety_margin_frac: float = 0.15  # ...but never more than this of the budget

    # -- measured Part B reserve --------------------------------------------
    # run_submission.py's run_risk (unmodified) calls cv2 VideoCapture.read()
    # on the ORIGINAL file for every single frame, official stride=1 -- that
    # decode is entirely outside our control and cannot be sped up by anything
    # in solution.py. The fixed part_b_reserve multiplier above was a guess
    # made before any real footage existed. Real footage turned out to be XAVC
    # H.264 High 4:2:2, 10-bit (yuv422p10le), 3840x2160 -- decoding that is far
    # more expensive per frame than the synthetic 1080p8 clips CP0 was tuned
    # against, so a fixed 1.2x is not trustworthy either direction: it could
    # starve Part A on this footage, or on a lighter clip reserve far more than
    # Part B will actually need.
    #
    # So detect_events probes a handful of REAL frames with the harness's own
    # decode call (plain cap.read(), not grab/retrieve) before doing anything
    # else, extrapolates to the full frame count, and Budget.part_b_reserve
    # uses that measurement instead of the fixed multiplier whenever one is
    # available (Budget.set_measured_part_b). See src/budget.measure_part_b_floor.
    part_b_probe_frames: int = 40    # frames to sample for the estimate
    part_b_probe_max_sec: float = 3.0  # hard cap on the probe's own wall time
    # ...but never more than this FRACTION of the video's own total budget --
    # same reasoning as safety_margin_frac: a flat 3.0s cap is fine for the
    # multi-minute real test clips, but on a very short clip (harness contract
    # tests use ~1.6s synthetic clips, budget 4.8s) even a fast few-millisecond
    # probe plus its own VideoCapture open/close overhead is a proportionally
    # large bite, and stacking that on top of a cold model load can be what
    # tips an already tight per-frame budget-check granularity over the edge.
    part_b_probe_max_frac: float = 0.05
    # CALIBRATION: MEASURED, 2026-09-23, real XAVC 4K clip (sample_001.mp4),
    # this 8-core machine. A 100-frame probe from frame 0 gave 49.12 ms/frame
    # (1.472x realtime); a 1500-frame probe from frame 0 gave 59.65 ms/frame
    # (1.788x realtime) -- ~21% higher. Decode cost is NOT flat across a clip
    # (GOP structure, reference buffering, scene motion), and the runtime probe
    # in _calibrate_part_b_reserve only ever samples from frame 0 (cheap: it
    # has to run inside the real budget), so it is likely to UNDERESTIMATE.
    # 1.3x, not 1.15x, covers the ~21% intra-clip variance actually observed
    # plus machine-to-machine slack. Revisit once a full-file decode-floor
    # confirmation exists (tools/bench.py without --max-frames).
    part_b_measured_safety: float = 1.30  # headroom over the measured rate


# --------------------------------------------------------------------------
# perception
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PerceptionConfig:
    """Anything here changes the tracks, so it is part of the cache key."""

    # ======================================================================
    # DETECTOR SPEED KNOBS -- the three values to edit if Stage 1 is over
    # budget. Nothing else in the codebase hard-codes them; everything reads
    # CFG.perception. Example fallback: imgsz=640, frame_stride=4.
    #
    # What changes when you flip them:
    #   * The perception cache key changes, so every clip re-runs Stage 1.
    #   * ByteTrack's track_buffer is 30 PROCESSED frames, so its real-time
    #     length scales with frame_stride: 2.0 s at stride 2 on 29.97 fps,
    #     4.0 s at stride 4. Rules are written in seconds and do not care.
    #   * `weights` must name a file in weights/ that weights/download.sh
    #     also fetches -- the run is offline, and only yolo11s.pt is listed
    #     there today. tests/test_config.py fails if the two disagree.
    # ======================================================================
    weights: str = "yolo11s.pt"      # model variant (yolo11n / s / m ...)
    imgsz: int = 960                 # detector input size; CCTV objects are small, 640 loses them
                                     # (the pyav decoder also downscales to this width)
    skip_frame: str = "NONKEY"       # pyav: NONKEY = keyframes only (every 0.5 s on
                                     # the real clips), NONREF = I+P (10/s)
    frame_stride: int = 2            # cv2/ffmpeg fallback only: every Nth frame
    # ======================================================================

    conf: float = 0.20               # low: ByteTrack's second stage uses the tail
    iou: float = 0.70                # NMS IoU
    max_det: int = 300
    half: bool = True                # fp16 on CUDA, ignored on CPU
    tracker: str = "bytetrack.yaml"
    # Frame source for Stage 1. "pyav" (default since CP4) decodes keyframes
    # only via the decoder's skip_frame (src/avdecode.py): ~0.18x realtime on
    # the 4K clips, against a full decode for cv2. Falls back to "cv2" when
    # PyAV is missing or cannot open the file.
    #
    # "cv2" is cv2.VideoCapture grab/retrieve, the CP0 path: grab() still
    # decodes every frame. "ffmpeg" pipes through an ffmpeg subprocess
    # that scales inside the decoder (src/ffdecode.py), built on the
    # hypothesis that scaling INSIDE the decoder would beat decoding full
    # 4K/10-bit and letting the detector resize it away.
    #
    # MEASURED, 2026-09-23, on the real XAVC clip (sample_001.mp4), this
    # 8-core machine, 300 processed frames at stride=2: the hypothesis was
    # WRONG. cv2 grab/retrieve at native 4K: 81.6 ms/frame. ffmpeg piped and
    # scaled to 640/960/1280 width: 81.9 / 93.8 / 104.0 ms/frame -- equal at
    # best, worse at every wider scale. Two reasons: (1) H.264 decode of this
    # 10-bit 4:2:2 stream dominates the per-frame cost and happens BEFORE any
    # scale filter runs, so scaling the output does not reduce it; (2) cv2's
    # grab()-only skip on non-stride frames is genuinely cheap (decode, no
    # colour-convert/copy), while the ffmpeg pipe must fully decode + scale +
    # convert + write EVERY frame to keep frame-index accounting correct, so
    # it pays full cost on frames cv2 gets almost for free.
    #
    # Kept available (off by default) rather than deleted: it is fully
    # opt-in, falls back to cv2 automatically on any failure, and may still
    # be worth revisiting for a different codec/resolution or with a
    # `select` filter added to skip full processing on non-stride frames.
    # DO NOT default this to "ffmpeg" without re-measuring first.
    decoder: str = "pyav"
    # Target width for the ffmpeg decoder's own scale filter (height follows
    # the source aspect ratio, rounded to even). Chosen above `imgsz` so the
    # detector's own letterboxing/resize still has real pixels to work from,
    # not upscaled ones. Ignored when decoder == "cv2".
    decode_width: int = 1280
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
