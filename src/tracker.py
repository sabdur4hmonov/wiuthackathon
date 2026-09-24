"""ByteTrack for a 0.5 s sample period.

Stage 1 now sees one frame per GOP (src/avdecode.py): 15 source frames, 0.5 s
on the real clips. Stock ByteTrack associates on box overlap (IoU) alone, and
confirms a new track only if its SECOND detection overlaps the first with IoU
above ~0.3. Measured on the full-rate CP3 tracks, 0.5 s apart:

    moving pedestrians   IoU < 0.2 in 90-94% of cases
    moving cars          IoU < 0.2 in 40-57%
    buses                IoU < 0.2 in  2%

so at keyframe rate a walking pedestrian would never get a confirmed track,
and jaywalking -- which is keyed on track_id -- could not fire. Retuning the
frame-counted parameters is not enough; the association cost has to tolerate
motion.

KeyframeTracker keeps all of ByteTrack (Kalman prediction, the two-stage
high/low score association, Hungarian assignment) and changes one thing: the
similarity between a track and a detection is the larger of their IoU and a
motion similarity,

    s_motion = max(0, 1 - centre_distance / (reach * box_height))

gated to the same class group (person / vehicle) and to a plausible change in
box height. `reach` is per group, in box heights per sample, from the same
measurement: over 0.5 s pedestrians move 0.48 heights at p95, vehicles 1.78.
The Kalman filter supplies the predicted position, so once a track has two
samples the distance is measured from where it should be, not where it was.

Only get_dists is overridden -- it exists, with this signature, in every
Ultralytics release since 8.1 -- so this does not depend on internals that
changed between 8.3 and 8.4.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

PERSON_CLASSES = (0,)


def _group(cls: np.ndarray) -> np.ndarray:
    """0 for people, 1 for everything else (vehicles)."""
    return (~np.isin(cls.astype(np.int64), PERSON_CLASSES)).astype(np.int8)


def motion_similarity(trk_xywh: np.ndarray, det_xywh: np.ndarray,
                      trk_cls: np.ndarray, det_cls: np.ndarray,
                      reach_person: float, reach_vehicle: float,
                      max_size_ratio: float) -> np.ndarray:
    """(n_tracks, n_dets) similarity in [0, 1]; boxes are centre x, y, w, h."""
    if len(trk_xywh) == 0 or len(det_xywh) == 0:
        return np.zeros((len(trk_xywh), len(det_xywh)), dtype=np.float32)
    th = np.maximum(trk_xywh[:, 3:4], 1e-3)
    dh = np.maximum(det_xywh[None, :, 3], 1e-3)
    dist = np.hypot(trk_xywh[:, 0:1] - det_xywh[None, :, 0],
                    trk_xywh[:, 1:2] - det_xywh[None, :, 1]) / ((th + dh) / 2)
    tg, dg = _group(trk_cls), _group(det_cls)
    reach = np.where(tg == 0, reach_person, reach_vehicle).astype(np.float32)[:, None]
    sim = np.clip(1.0 - dist / reach, 0.0, 1.0)
    ratio = dh / th
    ok = (tg[:, None] == dg[None, :]) & (ratio < max_size_ratio) & (ratio > 1.0 / max_size_ratio)
    return np.where(ok, sim, 0.0).astype(np.float32)


class Detections:
    """The Results-like object BYTETracker.update() consumes, built from arrays.

    Lets Stage 1 hand plain numpy detections to the tracker, and lets a
    tracker be replayed offline over stored detections.
    """

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> "Detections":
        return Detections(self.xyxy[idx], self.conf[idx], self.cls[idx])


def make_tracker(cfg, sample_step_frames: int, fps: float):
    """A KeyframeTracker configured from PerceptionConfig for this sample rate.

    track_buffer is given in SECONDS in the config and converted to samples
    here: ByteTrack counts it in update() calls, so its real-time meaning
    depends on how far apart the samples are.
    """
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.trackers.utils import matching

    class KeyframeTracker(BYTETracker):
        def get_dists(self, tracks, detections):
            iou_sim = 1.0 - matching.iou_distance(tracks, detections)
            if len(tracks) and len(detections):
                t_xywh = np.array([t.xywh for t in tracks], dtype=np.float32)
                d_xywh = np.array([d.xywh for d in detections], dtype=np.float32)
                t_cls = np.array([t.cls for t in tracks], dtype=np.float32)
                d_cls = np.array([d.cls for d in detections], dtype=np.float32)
                same = _group(t_cls)[:, None] == _group(d_cls)[None, :]
                sim = np.maximum(np.where(same, iou_sim, 0.0),
                                 motion_similarity(t_xywh, d_xywh, t_cls, d_cls,
                                                   cfg.track_reach_person,
                                                   cfg.track_reach_vehicle,
                                                   cfg.track_max_size_ratio))
            else:
                sim = iou_sim
            dists = 1.0 - sim
            if self.args.fuse_score:
                dists = matching.fuse_score(dists, detections)
            return dists

    args = SimpleNamespace(
        tracker_type="bytetrack",
        track_high_thresh=cfg.track_high_thresh,
        track_low_thresh=cfg.track_low_thresh,
        new_track_thresh=cfg.new_track_thresh,
        track_buffer=1,
        match_thresh=cfg.match_thresh,
        fuse_score=cfg.track_fuse_score,
    )
    tracker = KeyframeTracker(args)
    set_step(tracker, cfg, sample_step_frames, fps)
    tracker.reset_id()
    return tracker


def set_step(tracker, cfg, sample_step_frames: int, fps: float) -> None:
    """Re-express track_buffer_sec in samples once the real step is known.

    8.4 counts lost samples in max_frames_lost; 8.3 scaled track_buffer by
    frame_rate / 30 into max_time_lost. Both are pinned to samples.
    """
    step_sec = max(1, int(sample_step_frames)) / (fps or 25.0)
    buffer = max(1, int(np.ceil(cfg.track_buffer_sec / step_sec)))
    tracker.args.track_buffer = buffer
    tracker.max_frames_lost = buffer
    tracker.max_time_lost = buffer


def update(tracker, det: Detections) -> np.ndarray:
    """Rows of (x1, y1, x2, y2, track_id, score, cls) for the tracks this
    sample updated."""
    out = tracker.update(det)
    if len(out) == 0:
        return np.zeros((0, 7), dtype=np.float32)
    return np.asarray(out, dtype=np.float32)[:, :7]
