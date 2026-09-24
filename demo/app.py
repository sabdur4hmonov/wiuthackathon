"""Live demo: upload a traffic clip, get events, an annotated video and a risk curve.

Runs the submission's own code (src/, the same weights) on CPU. A Hugging Face
Space (Gradio SDK); see demo/README.md for the limits and deploy steps.

  Limits: .mp4 / .mov / .avi / .mkv, at most MAX_SEC seconds and MAX_MB MB.
  Expect roughly 2-3x the clip's length on a free CPU Space.

The four event rules (jaywalking, stopped_vehicle, wrong_way, congestion) are
drawn for ONE camera: config/zones.json, aligned to each upload by
src/align.py. If the upload is a different scene the alignment fails, and the
app says so and shows detections, tracks and the risk curve only -- zones
drawn on the wrong street would produce confident nonsense.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if not (ROOT / "src").exists():          # running from the repo's demo/ folder
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

try:                                     # Hugging Face ZeroGPU: must be imported before CUDA is touched
    import spaces  # type: ignore

    GPU = spaces.GPU
except Exception:  # noqa: BLE001 - local runs / CPU hardware: a no-op decorator
    def GPU(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        return lambda fn: fn

MAX_SEC = 120.0
MAX_MB = 200
OUT_WIDTH = 960
CLIP_PAD_SEC = 1.5
CLASS_COLOURS = {"jaywalking": (0, 140, 255), "stopped_vehicle": (0, 0, 255),
                 "wrong_way": (255, 0, 255), "congestion": (0, 200, 255)}


class DemoError(Exception):
    """A problem with the upload the user can act on."""


# ---------------------------------------------------------------------------
# the analysis (plain functions: testable without Gradio)
# ---------------------------------------------------------------------------
def probe(path: str) -> tuple[float, float, int, int, int]:
    """(duration, fps, n_frames, width, height), exactly as the harness derives
    them, or DemoError with a message for the user."""
    import cv2

    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        raise DemoError("The upload is empty.")
    if p.stat().st_size > MAX_MB * 1024 * 1024:
        raise DemoError(f"The file is larger than {MAX_MB} MB.")
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise DemoError("This file could not be opened as a video.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, _ = cap.read()
    finally:
        cap.release()
    if not ok or w <= 0 or h <= 0:
        raise DemoError("No frame could be decoded from this file.")
    if not 1.0 <= fps <= 240.0:
        fps = 25.0                                  # the harness's own fallback
    if n < 2:
        raise DemoError("The clip is too short (under two frames).")
    dur = n / fps
    if dur > MAX_SEC + 0.5:
        raise DemoError(f"The clip is {dur:.0f} s long; the demo accepts up to {MAX_SEC:.0f} s.")
    return dur, fps, n, w, h


def run_part_a(path: str, log) -> dict:
    """Stage 1 + zones + rules + Stage 3, returning events AND the tracks and
    zones for drawing. Same code as solution.detect_events, unpacked."""
    from src import pipeline
    from src.budget import Budget
    from src.perception import run_perception
    from src.postprocess import finalise
    from src.rules import run_rules

    class DemoBudget(Budget):
        """The demo has no 3x harness budget: never stop Stage 1 early. (The
        submission's long-GOP mode stops at the budget; almost every user
        upload is long-GOP, and a half-analysed clip would be a bad demo.)"""

        def should_stop(self) -> bool:
            return False

        def project_overrun(self, *a, **k) -> bool:
            return False

    budget = DemoBudget.for_video(path)
    log("Detecting and tracking ...")
    tracks = run_perception(path, budget, verbose=False)
    zones = pipeline._get_zones(tracks, verbose=False)
    zones = pipeline._align_zones(path, tracks, zones, budget, verbose=False)
    pose = tracks.pose or {}
    aligned = str(pose.get("reason", "")).startswith("aligned")
    if zones is not None and aligned:
        log("Running the event rules ...")
        events = finalise(run_rules(tracks, zones), tracks.duration)
        note = (f"Scene matched the configured camera (shift {pose.get('tx', 0):+.0f}, "
                f"{pose.get('ty', 0):+.0f} px at 960, rotation {pose.get('angle_deg', 0):+.2f} deg).")
    else:
        events = []
        note = ("This does not look like the configured camera "
                f"({pose.get('reason', 'no pose')}). The event rules need that camera's "
                "zones, so they were NOT run: detections, tracks and the risk curve only.")
    return {"events": [[round(s, 3), round(e, 3), lab] for s, e, lab in events],
            "tracks": tracks, "zones": zones if aligned else None, "note": note}


def run_part_b(path: str, fps: float, n: int, log) -> list[list[float]]:
    """The submission's causal RiskEstimator over every frame, as run_risk
    does. Demo settings: work at 2 Hz and no budget guard (there is no 3x
    budget here, and a user waiting for a curve wants the whole curve)."""
    import cv2

    from src.config import CFG
    from src.risk import RiskEstimator

    cfg = replace(CFG.risk, work_every_n_frames=max(1, int(round(fps / 2))),
                  work_budget_frac=100.0, part_b_wall_limit_x=1e9)
    est = RiskEstimator(cfg)
    est.reset({"video_id": Path(path).name, "fps": fps, "width": 0, "height": 0, "n_frames": n})
    cap = cv2.VideoCapture(path)
    curve, idx = [], 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            curve.append([round(idx / fps, 3), round(est.step(frame, idx / fps), 4)])
            idx += 1
            if idx % max(1, n // 10) == 0:
                log(f"Risk curve {100 * idx // max(1, n)}% ...")
    finally:
        cap.release()
    return curve


@GPU(duration=120)
def detect(path: str, fps: float, n: int):
    """The detector-heavy part -- Stage 1 + rules, and the risk pass -- in one
    call, so a ZeroGPU Space holds the GPU once per upload. Everything it
    returns is plain data (ZeroGPU runs it in a worker process); progress
    messages stay outside, in analyse()."""
    quiet = lambda m: None  # noqa: E731
    return run_part_a(path, quiet), run_part_b(path, fps, n, quiet)


def _draw_frame(img, t, tracks, zones, events, risk_at, scale):
    import cv2

    if zones is not None:
        for poly in zones.crossings:
            pts = (np.asarray(poly.polygon, np.float32) * scale).astype(np.int32)
            cv2.polylines(img, [pts], True, (255, 200, 0), 1, cv2.LINE_AA)
        for q in zones.signal_queue_zones:
            pts = (np.asarray(q.polygon, np.float32) * scale).astype(np.int32)
            cv2.polylines(img, [pts], True, (120, 120, 120), 1, cv2.LINE_AA)
    if tracks is not None and len(tracks):
        from src.tracks import COL

        d = tracks.data
        half = (tracks.frame_stride / tracks.fps) / 2 + 1e-3
        near = np.flatnonzero(np.abs(d[:, COL["t_sec"]] - t) <= half)
        for i in near:
            x1, y1, x2, y2 = (d[i, [COL["x1"], COL["y1"], COL["x2"], COL["y2"]]] * scale).astype(int)
            person = int(d[i, COL["cls"]]) == 0
            cv2.rectangle(img, (x1, y1), (x2, y2), (80, 220, 80) if person else (230, 160, 60), 1)
    active = [lab for s, e, lab in events if s <= t < e]
    y = 30
    cv2.putText(img, f"t = {t:6.1f} s", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, f"t = {t:6.1f} s", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    for lab in active:
        y += 30
        cv2.rectangle(img, (8, y - 22), (230, y + 6), CLASS_COLOURS.get(lab, (0, 0, 255)), -1)
        cv2.putText(img, lab.upper(), (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    r = risk_at(t)
    h, w = img.shape[:2]
    cv2.rectangle(img, (w - 30, 10), (w - 12, 110), (60, 60, 60), 1)
    cv2.rectangle(img, (w - 29, 110 - int(99 * r)), (w - 13, 109),
                  (0, 0, 255) if r >= 0.5 else (0, 200, 255), -1)
    cv2.putText(img, "risk", (w - 46, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def render(path: str, out_dir: Path, part_a: dict, curve: list, fps: float, log) -> tuple[str, list[str]]:
    """Annotated H.264 video (every 2nd frame, real-time playback) and one
    short clip per event cut from it."""
    import av

    ts = np.array([c[0] for c in curve]) if curve else np.zeros(0)
    rs = np.array([c[1] for c in curve]) if curve else np.zeros(0)

    def risk_at(t):
        if ts.size == 0:
            return 0.0
        return float(rs[min(ts.size - 1, int(np.searchsorted(ts, t)))])

    out = out_dir / "annotated.mp4"
    log("Rendering the annotated video ...")
    with av.open(path) as src, av.open(str(out), "w") as dst:
        s = src.streams.video[0]
        s.thread_type = "AUTO"
        src_w, src_h = s.codec_context.width, s.codec_context.height
        w = min(OUT_WIDTH, src_w - src_w % 2)
        h = int(round(src_h * w / src_w / 2)) * 2
        scale = w / src_w
        o = dst.add_stream("libx264", rate=max(1, int(round(fps / 2))))
        o.width, o.height, o.pix_fmt = w, h, "yuv420p"
        o.options = {"crf": "28", "preset": "veryfast"}
        tb, st = float(s.time_base), s.start_time or 0
        for k, f in enumerate(src.decode(s)):
            if k % 2:
                continue
            t = (f.pts - st) * tb if f.pts is not None else k / fps
            img = f.to_ndarray(format="bgr24", width=w, height=h)
            _draw_frame(img, t, part_a["tracks"], part_a["zones"], part_a["events"], risk_at, scale)
            for pkt in o.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
                dst.mux(pkt)
        for pkt in o.encode():
            dst.mux(pkt)

    clips = []
    for i, (s0, e0, lab) in enumerate(part_a["events"]):
        clip = out_dir / f"event_{i + 1:02d}_{lab}_{s0:.1f}s.mp4"
        _cut(out, clip, max(0.0, s0 - CLIP_PAD_SEC), e0 + CLIP_PAD_SEC)
        clips.append(str(clip))
    return str(out), clips


def _cut(src_path: Path, dst_path: Path, t0: float, t1: float) -> None:
    import av

    with av.open(str(src_path)) as src, av.open(str(dst_path), "w") as dst:
        s = src.streams.video[0]
        o = dst.add_stream("libx264", rate=s.average_rate)
        o.width, o.height, o.pix_fmt = s.codec_context.width, s.codec_context.height, "yuv420p"
        o.options = {"crf": "28", "preset": "veryfast"}
        for f in src.decode(s):
            t = float(f.pts * s.time_base) if f.pts is not None else 0.0
            if t < t0:
                continue
            if t > t1:
                break
            for pkt in o.encode(av.VideoFrame.from_ndarray(f.to_ndarray(format="bgr24"), format="bgr24")):
                dst.mux(pkt)
        for pkt in o.encode():
            dst.mux(pkt)


def plot(events: list, curve: list, duration: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1.2, 1]})
    labels = sorted({lab for _, _, lab in events}) or ["(no events)"]
    for s, e, lab in events:
        b, g, r = CLASS_COLOURS.get(lab, (0, 0, 255))
        ax1.barh(labels.index(lab), e - s, left=s, height=0.6, color=(r / 255, g / 255, b / 255))
    ax1.set_yticks(range(len(labels)), labels)
    ax1.set_title("Events")
    if curve:
        ax2.plot([c[0] for c in curve], [c[1] for c in curve], lw=1)
    ax2.axhline(0.5, ls="--", lw=0.8, color="grey")
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, max(duration, 1e-3))
    ax2.set_xlabel("seconds")
    ax2.set_title("Risk (0.5 = alarm threshold)")
    fig.tight_layout()
    return fig


def analyse(path: str, log=print) -> dict:
    """Everything for one upload. Raises DemoError for problems the user can fix."""
    t0 = time.perf_counter()
    dur, fps, n, w, h = probe(path)
    log(f"{dur:.1f} s at {fps:.2f} fps, {w}x{h}.")
    out_dir = Path(tempfile.mkdtemp(prefix="wiut_demo_"))
    log("Detecting, tracking, scoring risk ...")
    part_a, curve = detect(path, fps, n)
    video, clips = render(path, out_dir, part_a, curve, fps, log)
    result = {"video": Path(path).name, "duration_sec": round(dur, 2), "fps": round(fps, 3),
              "events": part_a["events"], "risk": curve, "note": part_a["note"],
              "processing_sec": round(time.perf_counter() - t0, 1)}
    js = out_dir / "prediction.json"
    js.write_text(json.dumps(result, indent=1))
    return {**result, "annotated": video, "clips": clips, "json": str(js),
            "figure": plot(part_a["events"], curve, dur)}


# ---------------------------------------------------------------------------
# the Gradio UI
# ---------------------------------------------------------------------------
def build_ui():
    import gradio as gr

    def handle(upload, progress=gr.Progress()):
        if not upload:
            raise gr.Error("Upload a video first.")
        path = upload if isinstance(upload, str) else getattr(upload, "name", None)
        steps = {"n": 0}

        def log(msg):
            steps["n"] += 1
            progress(min(0.95, steps["n"] / 16), desc=msg)

        try:
            r = analyse(path, log)
        except DemoError as e:
            raise gr.Error(str(e))
        except Exception:
            traceback.print_exc()
            raise gr.Error("Processing failed on this file. Try another clip (MP4, H.264, under "
                           f"{MAX_SEC:.0f} s).")
        table = [[f"{s:.1f}", f"{e:.1f}", lab] for s, e, lab in r["events"]]
        summary = (f"**{len(r['events'])} event(s)** in {r['duration_sec']} s of video "
                   f"(processed in {r['processing_sec']} s).  \n{r['note']}")
        return (summary, r["annotated"], r["figure"], table, r["clips"] or None, r["json"])

    with gr.Blocks(title="WIUT traffic event detection") as ui:
        gr.Markdown(
            "# Traffic event detection -- live demo\n"
            f"Upload a road-camera clip (**up to {MAX_SEC:.0f} s, {MAX_MB} MB**; MP4 works best). "
            "It runs the submitted pipeline on CPU: detection and tracking, the four event rules "
            "(jaywalking, stopped vehicle, wrong way, congestion) and the causal risk estimator. "
            "Expect about 2-3x the clip's length. The event rules are drawn for one intersection; "
            "on other scenes you get detections and the risk curve only.")
        with gr.Row():
            inp = gr.File(label="Video", file_types=["video"], type="filepath")
            btn = gr.Button("Analyse", variant="primary")
        summary = gr.Markdown()
        video = gr.Video(label="Annotated playback")
        fig = gr.Plot(label="Event timeline and risk curve")
        table = gr.Dataframe(headers=["start (s)", "end (s)", "event"], label="Events")
        clips = gr.Files(label="One clip per event")
        js = gr.File(label="prediction.json (harness format)")
        btn.click(handle, inputs=inp, outputs=[summary, video, fig, table, clips, js],
                  concurrency_limit=1)
    return ui


if __name__ == "__main__":
    build_ui().queue(max_size=8).launch(max_file_size=f"{MAX_MB}mb")
