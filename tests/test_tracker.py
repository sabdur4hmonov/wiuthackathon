"""src/tracker.py: ByteTrack that can link objects 0.5 s apart."""
import dataclasses

import numpy as np
import pytest

pytest.importorskip("ultralytics")
pytest.importorskip("lap")

from src import tracker as trk
from src.config import CFG

FPS, STEP = 30.0, 15            # one sample every 0.5 s, like the real keyframes
H, W = 60.0, 24.0               # a pedestrian box


def _det(boxes, cls=0, conf=0.8):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return trk.Detections(boxes, np.full(len(boxes), conf), np.full(len(boxes), cls))


def _walker(x0, y0, dx_per_sample, n):
    return [(x0 + k * dx_per_sample, y0, x0 + k * dx_per_sample + W, y0 + H) for k in range(n)]


def _ids(tracker, frames):
    out = []
    for boxes in frames:
        rows = trk.update(tracker, _det(boxes) if boxes else _det(np.zeros((0, 4))))
        out.append(sorted(int(r[4]) for r in rows))
    return out


def test_motion_similarity_basics():
    t = np.array([[100, 100, 20, 50]], np.float32)
    same = trk.motion_similarity(t, t, np.array([0.]), np.array([0.]), 1.0, 2.5, 1.5)
    assert same[0, 0] == pytest.approx(1.0)
    far = trk.motion_similarity(t, np.array([[160, 100, 20, 50]], np.float32),
                                np.array([0.]), np.array([0.]), 1.0, 2.5, 1.5)
    assert far[0, 0] == 0.0                      # 1.2 heights away, reach 1.0
    other_class = trk.motion_similarity(t, t, np.array([0.]), np.array([2.]), 1.0, 2.5, 1.5)
    assert other_class[0, 0] == 0.0              # a person never links to a car
    resized = trk.motion_similarity(t, np.array([[100, 100, 40, 100]], np.float32),
                                    np.array([0.]), np.array([0.]), 1.0, 2.5, 1.5)
    assert resized[0, 0] == 0.0                  # height doubled in 0.5 s: not the same object


def test_a_walker_with_no_overlap_between_samples_keeps_one_id():
    # 0.45 heights per sample (27 px) with a 24 px wide box: IoU is ZERO
    # between consecutive samples, the case stock ByteTrack cannot link.
    frames = _walker(100, 200, 27, 12)
    tr = trk.make_tracker(CFG.perception, STEP, FPS)
    ids = [i for i in _ids(tr, [[b] for b in frames]) if i]
    assert len(ids) >= 10
    assert len({i[0] for i in ids}) == 1


def test_stock_iou_association_breaks_on_the_same_walker():
    """The reason src/tracker.py exists: IoU-only association loses it."""
    from types import SimpleNamespace

    from ultralytics.trackers.byte_tracker import BYTETracker

    stock = BYTETracker(SimpleNamespace(
        tracker_type="bytetrack", track_high_thresh=0.25, track_low_thresh=0.1,
        new_track_thresh=0.25, track_buffer=4, match_thresh=0.8, fuse_score=True))
    frames = _walker(100, 200, 27, 12)
    tracked = [len(trk.update(stock, _det([b]))) for b in frames]
    # Each new detection starts an unconfirmed track that the next sample
    # cannot confirm (no overlap), so the walker is simply never output
    # after the first sample.
    assert sum(tracked) <= 2


def test_two_walkers_side_by_side_do_not_swap():
    a = _walker(100, 200, 25, 12)
    b = _walker(100, 240, 25, 12)                # 0.67 heights below, same pace
    tr = trk.make_tracker(CFG.perception, STEP, FPS)
    per_sample = []
    for ba, bb in zip(a, b):
        rows = trk.update(tr, _det([ba, bb]))
        per_sample.append({int(r[4]): float(r[1]) for r in rows})
    # Each id stays on its own row (y) throughout.
    rows_of = {}
    for s in per_sample[1:]:
        for tid, y in s.items():
            rows_of.setdefault(tid, set()).add(round(y / 40))
    assert len(rows_of) == 2
    assert all(len(v) == 1 for v in rows_of.values())


def test_buffer_is_in_seconds_not_samples():
    walk = _walker(100, 200, 10, 30)
    cfg = dataclasses.replace(CFG.perception, track_buffer_sec=2.0)
    # hidden for 1.5 s (3 samples): the same id comes back
    tr = trk.make_tracker(cfg, STEP, FPS)
    ids = _ids(tr, [[b] if not 5 <= k < 8 else [] for k, b in enumerate(walk)])
    assert len({i[0] for i in ids if i}) == 1
    # hidden for 3.5 s (7 samples): a new id
    tr = trk.make_tracker(cfg, STEP, FPS)
    ids = _ids(tr, [[b] if not 5 <= k < 12 else [] for k, b in enumerate(walk)])
    assert len({i[0] for i in ids if i}) == 2


def test_buffer_follows_the_sample_step():
    tr = trk.make_tracker(CFG.perception, 15, 29.97)
    assert tr.max_frames_lost == 4                # 2.0 s / 0.5 s
    trk.set_step(tr, CFG.perception, 2, 29.97)
    assert tr.max_frames_lost == 30               # 2.0 s at stride 2: the CP3 value
