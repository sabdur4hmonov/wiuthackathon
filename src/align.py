"""Per-video camera-pose alignment: move zones.json onto each video's framing.

zones.json is drawn once, against one camera pose (config/pose_reference.jpg is
that pose, downscaled). The real clips do not all share it: measured against
the reference, sample_003 sits ~117 px off at the stop line and sample_004
~33 px off, from a slightly different pan, zoom and roll. A static zones.json
silently breaks every rule on such a video -- lanes land on the wrong lanes.

So, per video:
  1. Stage 1 keeps a few frames it decodes anyway, downscaled to MATCH_WIDTH.
  2. Each is matched to the reference (SIFT on contrast-equalised grey, ratio
     test, RANSAC) under a SIMILARITY model: translation, uniform scale and
     rotation. The measured differences between clips are exactly that; a
     homography would add freedom the data does not need and noise it does
     not want.
  3. The per-frame estimates are combined by a median, and the ZONES are warped
     into the video's coordinates. Tracks stay as decoded, so the tracks cache
     stays valid and no rule changes.

FAIL SAFE. A wrong warp is worse than no warp: it moves every lane onto its
neighbour with total confidence. Weak matching (few inliers, low inlier share),
disagreement between frames, or an implausible transform (large rotation,
scale far from 1, a shift beyond a sane bound) all fall back to IDENTITY, and
the reason is logged.

PART B. RiskEstimator does not use zones today. If it ever does, it must align
causally -- only from frames step() has already been handed -- and cannot use
the whole-clip median computed here, which looks at the future.
"""
from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass, replace

import numpy as np

from .config import REPO_ROOT

REFERENCE_PATH = REPO_ROOT / "config" / "pose_reference.jpg"
MATCH_WIDTH = 960


@dataclass(frozen=True)
class AlignConfig:
    """Matching and fail-safe bounds. Measured values are noted where they exist."""

    # Frames Stage 1 keeps for matching, spread evenly over the clip. The
    # median over them ignores a short settle at the start (sample_001 drifts
    # 27.5 px @4K at t=0, 5 px by 10 s, <1 px after 33 s).
    frames: int = 9
    sift_features: int = 3000
    ratio: float = 0.75
    ransac_px: float = 2.0            # at MATCH_WIDTH

    # CALIBRATION: MEASURED 2026-09-24, 142 frames of the four real clips at
    # MATCH_WIDTH: same-pose frames give 733-1452 inliers (share 0.75-0.96),
    # the re-framed dusk and evening clips 72-177 (share >= 0.51). The floors
    # sit at about half the weakest real frame.
    min_inliers: int = 40
    min_inlier_share: float = 0.30
    # At least this many frames must count before a transform is trusted.
    min_good_frames: int = 1

    # Plausibility: anything outside this is a broken match, not a camera.
    max_rotation_deg: float = 4.0
    max_scale_dev: float = 0.06
    max_shift_frac: float = 0.10      # of the frame width
    # Frames must agree: the worst frame may differ from the median by at most
    # this, measured as displacement of the frame corners, in MATCH_WIDTH px.
    # Measured worst on real clips: 6.9 (sample_001's t=0 settle), else <= 3.1.
    max_disagreement_px: float = 8.0


CFG_ALIGN = AlignConfig()


@dataclass(frozen=True)
class Pose:
    """Similarity mapping REFERENCE -> VIDEO coordinates, both at MATCH_WIDTH.

    p_video = scale * R(angle) @ p_ref + (tx, ty). Identity when alignment was
    not possible; `reason` says why.
    """

    scale: float = 1.0
    angle_deg: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    good_frames: int = 0
    frames: int = 0
    median_inliers: int = 0
    drift_px: float = 0.0             # worst frame vs the median, MATCH_WIDTH px
    reason: str = "not estimated"

    @property
    def is_identity(self) -> bool:
        return (self.scale == 1.0 and self.angle_deg == 0.0
                and self.tx == 0.0 and self.ty == 0.0)

    def matrix(self, width: float) -> np.ndarray:
        """2x3 affine in the pixel space of a frame `width` pixels wide."""
        k = width / MATCH_WIDTH
        a = math.radians(self.angle_deg)
        c, s = self.scale * math.cos(a), self.scale * math.sin(a)
        return np.array([[c, -s, self.tx * k], [s, c, self.ty * k]], dtype=float)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Pose | None":
        if not d:
            return None
        try:
            return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})
        except Exception:
            return None


IDENTITY = Pose(reason="identity")


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------
def prepare(frame: np.ndarray) -> np.ndarray:
    """A decoded BGR frame -> MATCH_WIDTH-wide contrast-equalised grey."""
    import cv2

    h, w = frame.shape[:2]
    small = cv2.resize(frame, (MATCH_WIDTH, int(round(h * MATCH_WIDTH / w))),
                       interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    # Contrast equalisation: the clips span noon, golden hour and dusk.
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grey)


_REF: tuple | None = None


def _reference(cfg: AlignConfig):
    """(keypoints, descriptors, shape) of the reference pose, computed once."""
    global _REF
    if _REF is None:
        import cv2

        img = cv2.imread(str(REFERENCE_PATH))
        if img is None:
            raise FileNotFoundError(REFERENCE_PATH)
        ref = prepare(img)
        kp, des = cv2.SIFT_create(cfg.sift_features).detectAndCompute(ref, None)
        _REF = (kp, des, ref.shape)
    return _REF


@dataclass(frozen=True)
class FrameMatch:
    matrix: np.ndarray | None        # 2x3, reference -> this frame, MATCH_WIDTH px
    inliers: int
    matches: int

    @property
    def share(self) -> float:
        return self.inliers / self.matches if self.matches else 0.0


def match_frame(small: np.ndarray, cfg: AlignConfig = CFG_ALIGN) -> FrameMatch:
    """Similarity from the reference to one prepared frame."""
    import cv2

    kp_r, des_r, _ = _reference(cfg)
    kp, des = cv2.SIFT_create(cfg.sift_features).detectAndCompute(small, None)
    if des is None or des_r is None or len(kp) < 3:
        return FrameMatch(None, 0, 0)
    pairs = cv2.BFMatcher().knnMatch(des_r, des, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < cfg.ratio * p[1].distance]
    if len(good) < 3:
        return FrameMatch(None, 0, len(good))
    src = np.float32([kp_r[m.queryIdx].pt for m in good])
    dst = np.float32([kp[m.trainIdx].pt for m in good])
    cv2.setRNGSeed(0)                  # RANSAC must be reproducible run to run
    m, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                         ransacReprojThreshold=cfg.ransac_px,
                                         maxIters=5000, confidence=0.995)
    n_in = int(inl.sum()) if inl is not None else 0
    return FrameMatch(m, n_in, len(good))


def _params(m: np.ndarray) -> tuple[float, float, float, float]:
    scale = float(math.hypot(m[0, 0], m[1, 0]))
    angle = float(math.degrees(math.atan2(m[1, 0], m[0, 0])))
    return scale, angle, float(m[0, 2]), float(m[1, 2])


def _corner_gap(a: np.ndarray, b: np.ndarray, shape) -> float:
    """Worst displacement between two transforms over the frame corners."""
    h, w = shape
    pts = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1],
                    [w / 2, h / 2, 1]], dtype=float)
    return float(np.hypot(*((pts @ a.T) - (pts @ b.T)).T).max())


def estimate(smalls: list[np.ndarray], cfg: AlignConfig = CFG_ALIGN,
             matches: list[FrameMatch] | None = None) -> Pose:
    """Combine per-frame matches into one Pose, or IDENTITY with a reason."""
    if not smalls and not matches:
        return replace(IDENTITY, reason="no frames")
    ms = matches if matches is not None else [match_frame(s, cfg) for s in smalls]
    good = [m for m in ms if m.matrix is not None and m.inliers >= cfg.min_inliers
            and m.share >= cfg.min_inlier_share]
    base = dict(frames=len(ms), good_frames=len(good),
                median_inliers=int(np.median([m.inliers for m in ms])) if ms else 0)
    if len(good) < cfg.min_good_frames:
        return replace(IDENTITY, reason=f"weak match: {len(good)}/{len(ms)} frames "
                                        f"passed (best {max((m.inliers for m in ms), default=0)} inliers)",
                       **base)

    p = np.array([_params(m.matrix) for m in good])
    scale, angle, tx, ty = (float(v) for v in np.median(p, axis=0))
    med = Pose(scale, angle, tx, ty).matrix(MATCH_WIDTH)
    shape = _reference(cfg)[2]
    drift = max(_corner_gap(m.matrix, med, shape) for m in good)
    base["drift_px"] = round(drift, 2)

    if abs(angle) > cfg.max_rotation_deg:
        return replace(IDENTITY, reason=f"implausible rotation {angle:.2f} deg", **base)
    if abs(scale - 1.0) > cfg.max_scale_dev:
        return replace(IDENTITY, reason=f"implausible scale {scale:.4f}", **base)
    if math.hypot(tx, ty) > cfg.max_shift_frac * MATCH_WIDTH:
        return replace(IDENTITY, reason=f"implausible shift ({tx:.1f}, {ty:.1f}) px", **base)
    if drift > cfg.max_disagreement_px:
        return replace(IDENTITY, reason=f"frames disagree by {drift:.1f} px", **base)
    return Pose(scale, angle, tx, ty, reason="aligned", **base)


def sample_times(duration: float, cfg: AlignConfig = CFG_ALIGN) -> list[float]:
    """Timestamps (s) at which Stage 1 keeps a frame for matching."""
    if duration <= 0:
        return [0.0]
    return [float(t) for t in np.linspace(0.0, duration, cfg.frames, endpoint=False)]


def estimate_from_video(video_path: str, duration: float,
                        cfg: AlignConfig = CFG_ALIGN) -> Pose:
    """Decode a few frames and estimate. ONLY for tracks from a cache entry
    written before alignment existed; a live run takes its frames from Stage 1."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    smalls = []
    try:
        for t in sample_times(duration, cfg):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, frame = cap.read()
            if ok:
                smalls.append(prepare(frame))
    finally:
        cap.release()
    return estimate(smalls, cfg)


# ---------------------------------------------------------------------------
# applying a pose to zones
# ---------------------------------------------------------------------------
def warp_zones(zones, pose: Pose | None, frame_width: float):
    """zones (already in this video's resolution) -> this video's framing.

    Polygons, segments and signal ROIs move with the similarity; lane
    direction vectors rotate with it (scale does not change a unit vector).
    A similarity has positive determinant, so every stop line keeps its
    approach side.
    """
    if zones is None or pose is None or pose.is_identity:
        return zones
    m = pose.matrix(frame_width)
    a = math.radians(pose.angle_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])

    def pts(p) -> np.ndarray:
        p = np.asarray(p, dtype=float).reshape(-1, 2)
        return p @ m[:, :2].T + m[:, 2]

    def seg(s):
        q = pts(s)
        return ((float(q[0, 0]), float(q[0, 1])), (float(q[1, 0]), float(q[1, 1])))

    def area(a_):
        return replace(a_, polygon=pts(a_.polygon))

    def roi(r):
        x1, y1, x2, y2 = r
        q = pts([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        return (float(q[:, 0].min()), float(q[:, 1].min()),
                float(q[:, 0].max()), float(q[:, 1].max()))

    def direction(d):
        v = rot @ np.asarray(d, dtype=float)
        n = float(np.hypot(*v)) or 1.0
        return (float(v[0] / n), float(v[1] / n))

    return replace(
        zones,
        carriageway=tuple(area(a_) for a_ in zones.carriageway),
        lanes=tuple(replace(ln, polygon=pts(ln.polygon), direction=direction(ln.direction))
                    for ln in zones.lanes),
        stop_lines=tuple(replace(sl, segment=seg(sl.segment)) for sl in zones.stop_lines),
        crossings=tuple(area(a_) for a_ in zones.crossings),
        signal_queue_zones=tuple(area(a_) for a_ in zones.signal_queue_zones),
        traffic_lights=tuple(replace(t, roi=roi(t.roi)) for t in zones.traffic_lights),
        lane_markings=tuple(replace(lm, segment=seg(lm.segment)) for lm in zones.lane_markings),
    )


def log(pose: Pose, where: str = "") -> None:
    msg = (f"[align]{where} {pose.reason}: scale {pose.scale:.4f}, rot {pose.angle_deg:+.2f} deg, "
           f"shift ({pose.tx:+.1f}, {pose.ty:+.1f}) px@{MATCH_WIDTH}, "
           f"{pose.good_frames}/{pose.frames} frames, median {pose.median_inliers} inliers, "
           f"drift {pose.drift_px:.1f} px")
    print(msg, file=sys.stderr)
