"""Stage 1 -- video in, tracks out. The only stage that touches pixels.

Detector: YOLO11s (Ultralytics), local weights, loaded once per process.
Decoder:  keyframes only, through PyAV (src/avdecode.py): one frame per GOP,
          0.5 s apart on the real clips.
Tracker:  ByteTrack with a motion-tolerant association (src/tracker.py) --
          stock ByteTrack links boxes by overlap and cannot follow a walking
          pedestrian across 0.5 s. cv2/ffmpeg fallbacks keep model.track.

Why this pair, for a T4 and a 3x-duration budget:

* YOLO11s at imgsz=960 fits a T4 with room to spare in fp16 and leaves headroom
  for Part B's decode pass. YOLO11n is faster but drops the small, distant
  vehicles that a CCTV view is full of; YOLO11m roughly doubles inference for a
  gain concentrated on large objects we already detect. The `s` model at a
  larger imgsz beats the `m` model at 640 on small objects for the same cost,
  which is the trade that matters here.
* RT-DETR was the alternative. It is stronger on crowded scenes but ~2x the
  latency at equal resolution, has no NMS knob to trade recall for speed under
  budget pressure, and Ultralytics' tracker integration is less exercised for
  it. Losing the whole video to a timeout costs more than the accuracy gains.
* ByteTrack associates the LOW-confidence detections in a second pass, which is
  precisely the regime of a partly occluded car at the far end of the frame.
  It needs no ReID model, so it adds no GPU cost to the budget.

ID-SWITCH BEHAVIOUR -- every rule downstream assumes track continuity, so:
  - ByteTrack is motion-only (IoU + Kalman). It has NO appearance model.
  - A track occluded for <= track_buffer_sec (2 s) is re-associated and KEEPS
    its id. Past that it is deleted, and the object gets a NEW id on
    reappearance.
  - At 0.5 s per sample a platoon of cars ALIASES: the track hops back one car
    per sample and drifts backwards. Tracks carry position, not direction or
    speed, at that rate (see thresholds.WrongWayThresholds.max_sample_sec).
  - Two similar boxes crossing with high IoU CAN swap ids. On this camera that
    happens where lanes converge in the distance and boxes are small.
  - Consequence for Stage 2: never assume one id spans a whole manoeuvre.
    Rules must tolerate a track ending and a new one starting in the same place
    (fragment stitching belongs in Stage 3), and must not treat an id change as
    evidence that anything physical happened.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from . import align, avdecode, tracker
from .budget import Budget
from .config import CFG, WEIGHTS_DIR, PerceptionConfig, enforce_offline, seed_everything
from .ffdecode import FFmpegFrameReader, FFmpegUnavailable, scaled_size
from .tracks import COL, COLUMNS, TrackTable, compute_kinematics, make_row

_MODEL = None
_MODEL_KEY: tuple | None = None
# Budget is checked every this many PROCESSED frames. Cheap, but not free, and
# checking every frame would show up in the per-frame cost we are protecting.
_BUDGET_CHECK_EVERY = 10


class PerceptionUnavailable(RuntimeError):
    """Raised when the detector cannot be loaded (missing deps or weights)."""


def weights_path(cfg: PerceptionConfig | None = None) -> Path:
    cfg = cfg or CFG.perception
    p = Path(cfg.weights)
    return p if p.is_absolute() else WEIGHTS_DIR / p


def load_model(cfg: PerceptionConfig | None = None):
    """Load the detector once per process, from an explicit local path.

    Never passes a bare model NAME to Ultralytics: that is the code path that
    downloads. Only a path to a file that already exists, which fails loudly
    offline instead of silently reaching for the network.
    """
    global _MODEL, _MODEL_KEY
    cfg = cfg or CFG.perception
    enforce_offline()
    key = (str(weights_path(cfg)), cfg.imgsz)
    if _MODEL is not None and _MODEL_KEY == key:
        return _MODEL

    wp = weights_path(cfg)
    if not wp.exists():
        raise PerceptionUnavailable(
            f"weights not found at {wp}. Run weights/download.sh once, with "
            f"internet, before the offline evaluation."
        )
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise PerceptionUnavailable(f"ultralytics is not installed: {e}") from e

    model = YOLO(str(wp))
    _disable_amp_check(model)
    _MODEL, _MODEL_KEY = model, key
    return model


def _disable_amp_check(model) -> None:
    """Stop Ultralytics' AMP self-check, which downloads yolo11n.pt.

    The check runs on the first CUDA inference and fetches a nano checkpoint to
    compare fp16 and fp32 outputs. With no internet that is at best a long
    timeout and at worst an exception inside detect_events -- i.e. an empty
    prediction for the video. We pin the flag on every object that owns one.
    """
    for obj in (model, getattr(model, "model", None), getattr(model, "predictor", None)):
        if obj is None:
            continue
        for attr in ("amp", "_amp_checked"):
            try:
                setattr(obj, attr, False if attr == "amp" else True)
            except Exception:
                pass
    try:
        args = getattr(getattr(model, "model", None), "args", None)
        if isinstance(args, dict):
            args["amp"] = False
    except Exception:
        pass


def build_track_kwargs(cfg: PerceptionConfig, device: str) -> dict:
    """Arguments for model.track(), assembled once per video.

    `half` is only passed on CUDA. It is meaningless on CPU, and Ultralytics
    8.4 deprecated the name in favour of `quantize` -- passing it regardless
    would emit a warning on every single frame.
    """
    kwargs = {
        "persist": True,
        "tracker": cfg.tracker,
        "imgsz": cfg.imgsz,
        "conf": cfg.conf,
        "iou": cfg.iou,
        "classes": list(cfg.classes),
        "max_det": cfg.max_det,
        "device": device,
        "verbose": False,
    }
    if cfg.half and device != "cpu":
        kwargs["half"] = True
    return kwargs


def build_predict_kwargs(cfg: PerceptionConfig, device: str) -> dict:
    """Arguments for model.predict() on the keyframe path: detection only,
    tracking is done by src/tracker.py."""
    kwargs = build_track_kwargs(cfg, device)
    kwargs.pop("persist")
    kwargs.pop("tracker")
    return kwargs


def _detections(res, sx: float, sy: float) -> "tracker.Detections":
    """One predict() result as tracker input, boxes in native pixels."""
    r = res[0] if isinstance(res, (list, tuple)) else res
    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return tracker.Detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0))
    xyxy = boxes.xyxy.cpu().numpy() * np.array([sx, sy, sx, sy], dtype=np.float32)
    return tracker.Detections(xyxy, boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy())


def device_string() -> str:
    """'0' for the first CUDA device, else 'cpu'. Reported in the timing log."""
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ---------------------------------------------------------------------------
# the pass
# ---------------------------------------------------------------------------
def run_perception(video_path: str | Path,
                   budget: Budget,
                   cfg: PerceptionConfig | None = None,
                   verbose: bool = True) -> TrackTable:
    """Decode, detect and track. Returns whatever was finished in time.

    Degrades rather than fails: on a budget stop the table is returned with
    complete=False and processed_until_sec set, so Stage 2 can still emit events
    for the part of the video that was seen. A partial answer scores; a timeout
    scores zero for BOTH parts.
    """
    import cv2

    cfg = cfg or CFG.perception
    seed_everything()
    stride = max(1, cfg.frame_stride)

    table = TrackTable.empty(fps=budget.fps, duration=budget.duration,
                             n_frames=budget.n_frames, frame_stride=stride)

    try:
        t_load = time.perf_counter()
        model = load_model(cfg)
        budget.note("model_load", time.perf_counter() - t_load,
                    "one-off; excluded from the per-frame rate")
    except PerceptionUnavailable as e:
        # No detector => no tracks => no events. That is a bad score, but it is
        # a VALID submission, which is the thing we refuse to risk.
        if verbose:
            print(f"[perception] unavailable, returning empty tracks: {e}",
                  file=sys.stderr)
        budget.note("perception", 0.0, f"unavailable: {e}")
        return table

    device = device_string()

    # -- decoder selection ---------------------------------------------------
    # "cv2" (default, KEEP THIS -- see PerceptionConfig.decoder in
    # src/config.py) reproduces CP0 exactly: ONE VideoCapture, opened once,
    # used for both metadata and the grab()/retrieve() loop. "ffmpeg" pipes
    # through src.ffdecode instead, scaling to cfg.decode_width -- it needs
    # native width/height FIRST (to compute the scaled target size), so only
    # that path pays for a throwaway metadata-only capture. Do not restructure
    # this to open a capture unconditionally "for symmetry": a small clip's
    # entire time budget can be under 2 s (see BudgetConfig.part_b_probe_*),
    # and a second VideoCapture open/close was measured to be enough overhead,
    # stacked on a cold model load, to blow that budget on the harness
    # contract tests -- this is why that extra open existed only briefly and
    # was removed.
    #
    # width/height below become the TrackTable coordinate frame -- the SCALED
    # size when using ffmpeg, never the native 4K size. Every zones.json
    # lookup and every rule reasons in that frame, through Zones own
    # rescaling (authored_against vs the video actual size), so this must be
    # the size frames were actually decoded at, not the source file size.
    ff_reader = None
    av_reader = None
    decoder_used = "cv2"
    cap = None
    # Detector-frame -> native-pixel factor. Boxes are stored in NATIVE pixels
    # whatever the decode size, so zones, box-height units and cached tracks
    # all keep one coordinate frame.
    sx = sy = 1.0

    if cfg.decoder == "pyav":
        try:
            av_reader = avdecode.KeyframeReader(video_path, budget.fps,
                                                cfg.skip_frame, cfg.imgsz)
            width, height = av_reader.native_width, av_reader.native_height
            sx, sy = av_reader.scale
            decoder_used = f"pyav/{cfg.skip_frame}"
        except Exception as e:  # noqa: BLE001 - never lose the run to the decoder
            if verbose:
                print(f"[perception] pyav decoder unavailable, falling back to "
                      f"cv2: {e!r}", file=sys.stderr)
            av_reader = None

    if cfg.decoder == "ffmpeg":
        try:
            probe = cv2.VideoCapture(str(video_path))
            if not probe.isOpened():
                budget.note("perception", 0.0, "cannot open video")
                return table
            native_width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
            native_height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
            probe.release()
            width, height = scaled_size(native_width, native_height,
                                        cfg.decode_width)
            reader = FFmpegFrameReader(video_path, width, height)
            reader.__enter__()
            ff_reader = reader
            decoder_used = "ffmpeg"
        except Exception as e:
            # A missing binary or a pipe that fails to start must never break
            # the run: fall back to the cv2 path exactly as CP0 always has.
            if verbose:
                print(f"[perception] ffmpeg decoder unavailable, falling back "
                      f"to cv2: {e!r}", file=sys.stderr)
            ff_reader = None
            decoder_used = "cv2"

    if ff_reader is None and av_reader is None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            budget.note("perception", 0.0, "cannot open video")
            return table
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    track_kwargs = build_track_kwargs(cfg, device)

    rows: list[np.ndarray] = []
    idx = 0
    processed = 0
    last_t = 0.0
    t_start = time.perf_counter()
    # The per-frame loop starts here, AFTER the model load. project_overrun
    # measures the marginal cost from this point, so the one-off startup is not
    # amortised across the first handful of frames.
    loop_t0 = t_start
    expected = max(1, budget.n_frames // stride)
    # Camera-pose alignment (src/align.py) from frames this decode already
    # has, matched as they arrive. Matching inside the loop -- not after it --
    # is what charges its cost to the per-frame projection, so perception can
    # never use up the budget that alignment then needs: an early version that
    # matched at the end found no headroom left on every real clip.
    align_at = align.sample_times(budget.duration)
    align_matches: list = []
    align_state = {"skipped": 0, "sec": 0.0}

    def _keep_for_alignment(frame, t_sec):
        if align_at and t_sec >= align_at[0]:
            while align_at and t_sec >= align_at[0]:
                align_at.pop(0)
            if budget.remaining_to_hard() < align.CFG_ALIGN.min_headroom_sec:
                align_state["skipped"] += 1
                return
            t_al = time.perf_counter()
            try:
                align_matches.append(align.match_frame(align.prepare(frame)))
            except Exception:  # noqa: BLE001 - a bad frame costs alignment, never tracks
                pass
            align_state["sec"] += time.perf_counter() - t_al

    def _project_by_position(frame_idx):
        done = frame_idx + 1
        if budget.project_overrun(done, budget.n_frames, loop_t0):
            budget.force_stop(
                f"projected overrun at frame {done}/{budget.n_frames} after "
                f"{budget.elapsed():.1f}s, limit {budget.part_a_hard:.1f}s")
            return True
        return False

    def _should_stop_after(processed_count):
        """Shared budget check, invoked identically from either decode path."""
        if processed_count % _BUDGET_CHECK_EVERY != 0:
            return False
        if budget.should_stop():
            return True
        if budget.project_overrun(processed_count, expected, loop_t0):
            budget.force_stop(
                f"projected overrun: {processed_count}/{expected} frames in "
                f"{budget.elapsed():.1f}s, limit {budget.part_a_hard:.1f}s"
            )
            return True
        return False

    seen: list[int] = []

    try:
        if av_reader is not None:
            # Only the frames skip_frame lets through are decoded at all. The
            # budget projection goes by POSITION in the video (source frame
            # index), since the number of keyframes is not known up front.
            #
            # Detection is model.predict; tracking is src/tracker.py, ByteTrack
            # with a motion-tolerant association. Stock ByteTrack (model.track)
            # cannot link a walking pedestrian across 0.5 s.
            predict_kwargs = build_predict_kwargs(cfg, device)
            kf_tracker = None
            for idx, frame in av_reader.frames():
                t_sec = idx / budget.fps if budget.fps else 0.0
                _keep_for_alignment(frame, t_sec)
                det = _detections(model.predict(frame, **predict_kwargs), sx, sy)
                if kf_tracker is None:
                    # The real step is known from the second keyframe on.
                    kf_tracker = tracker.make_tracker(
                        cfg, int(round(0.5 * (budget.fps or 25.0))), budget.fps)
                elif len(seen) == 1:
                    tracker.set_step(kf_tracker, cfg, idx - seen[0], budget.fps)
                for r in tracker.update(kf_tracker, det):
                    rows.append(make_row(idx, t_sec, int(r[4]), int(r[6]), float(r[5]),
                                         float(r[0]), float(r[1]), float(r[2]), float(r[3])))
                last_t = t_sec
                seen.append(idx)
                processed += 1
                if processed % _BUDGET_CHECK_EVERY == 0 and (
                        budget.should_stop() or _project_by_position(idx)):
                    break
        elif ff_reader is not None:
            # Every source frame arrives already scaled; unlike cv2 grab-only
            # skip, a frame we do not process still cost a decode+scale+pipe
            # write, since ffmpeg cannot cheaply skip that work mid-pipe the
            # way cv2 two-call grab/retrieve can.
            for idx, frame in ff_reader.frames():
                if idx % stride == 0:
                    t_sec = idx / budget.fps if budget.fps else 0.0
                    _keep_for_alignment(frame, t_sec)
                    res = model.track(frame, **track_kwargs)
                    rows.extend(_rows_from_result(res, idx, t_sec))
                    last_t = t_sec
                    processed += 1
                    if _should_stop_after(processed):
                        break
        else:
            while True:
                # grab() advances without the colour convert + copy that
                # read() does, so skipped frames cost only the decode we
                # cannot avoid.
                if not cap.grab():
                    break
                if idx % stride == 0:
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    t_sec = idx / budget.fps if budget.fps else 0.0
                    _keep_for_alignment(frame, t_sec)
                    res = model.track(frame, **track_kwargs)
                    rows.extend(_rows_from_result(res, idx, t_sec))
                    last_t = t_sec
                    processed += 1
                    if _should_stop_after(processed):
                        break
                idx += 1
    finally:
        if cap is not None:
            cap.release()
        if ff_reader is not None:
            ff_reader.__exit__(None, None, None)
        if av_reader is not None:
            av_reader.close()

    if len(seen) >= 2:
        # The sample step actually achieved (the GOP for NONKEY). Rules use it
        # as "how much time one sample covers".
        stride = max(1, int(np.median(np.diff(seen))))

    data = (np.vstack(rows).astype(np.float32) if rows
            else np.zeros((0, len(COLUMNS)), dtype=np.float32))
    data = compute_kinematics(data, cfg.velocity_window_samples(budget.fps, stride))
    pose = _combine_pose(align_matches, align_state, budget, verbose)

    table = TrackTable(
        data=data, fps=budget.fps, duration=budget.duration,
        n_frames=budget.n_frames, width=width, height=height,
        frame_stride=stride,
        complete=budget.stop_reason is None,
        processed_until_sec=last_t,
        pose=pose.to_dict(),
    )
    elapsed = time.perf_counter() - t_start
    budget.note("perception", elapsed,
                f"{processed} frames, {len(table)} rows, dev={device}, "
                f"decoder={decoder_used}, "
                f"{elapsed / budget.duration:.3f}x realtime"
                if budget.duration else "")
    if verbose:
        print(f"[perception] {table.summary()} in {elapsed:.1f}s on {device} "
              f"(decoder={decoder_used})", file=sys.stderr)
    return table


def _combine_pose(matches: list, state: dict, budget: Budget,
                  verbose: bool) -> "align.Pose":
    """One pose from the per-frame matches made during the loop; IDENTITY when
    none could be afforded or trusted. Combining is cheap: the matching already
    happened, and was charged, inside the loop."""
    from dataclasses import replace

    if not matches:
        reason = ("skipped: no budget headroom" if state["skipped"] else "no frames")
        pose = replace(align.IDENTITY, reason=reason)
    else:
        try:
            pose = align.estimate([], matches=matches)
        except Exception as e:  # noqa: BLE001 - alignment must never cost the video
            pose = replace(align.IDENTITY, reason=f"failed: {e!r}")
    budget.note("align", state["sec"], f"{len(matches)} frames matched in-loop; {pose.reason}")
    if verbose:
        align.log(pose)
    return pose


def _rows_from_result(res, frame_idx: int, t_sec: float,
                      sx: float = 1.0, sy: float = 1.0) -> list[np.ndarray]:
    """Flatten one Ultralytics result into track rows.

    Detections the tracker did not adopt are kept with track_id = -1 rather than
    dropped: a stopped_vehicle or road_obstacle can sit untracked for a long
    time, and throwing the evidence away at Stage 1 would make that class
    undetectable no matter what Stage 2 does.

    (sx, sy) scale detector-frame coordinates back to native pixels.
    """
    out: list[np.ndarray] = []
    if not res:
        return out
    r = res[0] if isinstance(res, (list, tuple)) else res
    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return out
    try:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        ids = (boxes.id.cpu().numpy() if getattr(boxes, "id", None) is not None
               else np.full(len(xyxy), -1.0))
    except Exception:
        return out
    for k in range(len(xyxy)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[k])
        out.append(make_row(frame_idx, t_sec, int(ids[k]), int(cls[k]),
                            float(conf[k]), x1 * sx, y1 * sy, x2 * sx, y2 * sy))
    return out
