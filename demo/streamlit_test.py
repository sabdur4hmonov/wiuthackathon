"""Streamlit Community Cloud front end for the live demo.

Same analysis and outputs as the Gradio app (demo/app.py, kept as is): it calls
app.analyse() unchanged -- the submission's Stage 1 + rules + Stage 3 and the
causal risk estimator -- and shows the summary, annotated playback, event
timeline + risk curve, the event table, one clip per event, and the
prediction in the harness's JSON format.

Deployed on Streamlit Community Cloud with this file as the entrypoint and
demo/requirements.txt as its requirements.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import app  # noqa: E402  (demo/app.py: analyse(), DemoError, limits)

st.set_page_config(page_title="WIUT traffic events", page_icon="🚦", layout="wide")
st.title("Traffic event detection — live demo")
st.markdown(
    f"Upload a road-camera clip (**up to {app.MAX_SEC:.0f} s, {app.MAX_MB} MB**; MP4 works best). "
    "It runs the submitted pipeline on CPU: detection and tracking, the four event rules "
    "(jaywalking, stopped vehicle, wrong way, congestion) and the causal risk estimator. "
    "Expect about 2-4x the clip's length. The event rules are drawn for one intersection; "
    "on other scenes you get detections and the risk curve only.")

upload = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv", "m4v"])


def _check(label: str, fn) -> bool:
    """One self-test step: its result, or its full traceback."""
    try:
        st.write(f"✅ **{label}**: {fn()}")
        return True
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        print(f"[self-test] {label} FAILED\n{tb}", file=sys.stderr, flush=True)
        st.write(f"❌ **{label}**")
        st.code(tb)
        return False


def self_test() -> None:
    """Environment checks, then the real analyse() on a clip made here."""
    import importlib.metadata as md
    import platform

    def pkgs():
        out = {}
        for p in ("torch", "ultralytics", "av", "lap", "streamlit", "gradio", "spaces",
                  "opencv-python", "opencv-python-headless", "numpy"):
            try:
                out[p] = md.version(p)
            except md.PackageNotFoundError:
                out[p] = "-"
        return f"python {platform.python_version()}; {out}"

    def cv2_check():
        import cv2

        return f"cv2 {cv2.__version__} from {Path(cv2.__file__).parent}"

    def gl_check():
        import ctypes.util

        return f"libGL: {ctypes.util.find_library('GL')}, libglib: {ctypes.util.find_library('glib-2.0')}"

    def av_check():
        import av

        dec, enc = av.codec.Codec("h264", "r").name, av.codec.Codec("libx264", "w").name
        return f"PyAV {av.__version__}: h264 decoder {dec}, encoder {enc}"

    def files_check():
        w = app.ROOT / "weights" / "yolo11s.pt"
        z = app.ROOT / "config" / "zones.json"
        return f"weights {w.stat().st_size} B, zones.json {z.stat().st_size} B, root {app.ROOT}"

    def model_check():
        from src.perception import load_model

        load_model()
        return "YOLO11s loaded"

    state = {}

    def make_clip():
        import av
        import numpy as np

        p = Path(tempfile.mkdtemp()) / "selftest.mp4"
        with av.open(str(p), "w") as out:
            s = out.add_stream("libx264", rate=25)
            s.width, s.height, s.pix_fmt = 320, 240, "yuv420p"
            rng = np.random.default_rng(0)
            for _ in range(50):
                for pkt in s.encode(av.VideoFrame.from_ndarray(
                        rng.integers(0, 255, (240, 320, 3), dtype=np.uint8), format="bgr24")):
                    out.mux(pkt)
            for pkt in s.encode():
                out.mux(pkt)
        state["clip"] = str(p)
        return f"{p.stat().st_size} B H.264 clip"

    def analyse_check():
        r = app.analyse(state["clip"], log=lambda m: None)
        return (f"{len(r['events'])} events, {len(r['risk'])} risk samples, annotated "
                f"{Path(r['annotated']).stat().st_size} B, {r['processing_sec']} s")

    for label, fn in (("packages", pkgs), ("OpenCV import", cv2_check), ("system libs", gl_check),
                      ("PyAV codecs", av_check), ("repo files", files_check),
                      ("detector", model_check), ("make a clip", make_clip)):
        _check(label, fn)
    if "clip" in state:
        _check("analyse() end to end", analyse_check)


with st.sidebar:
    st.caption("Diagnostics")
    if st.button("Run self-test"):
        self_test()


def run(data: bytes, name: str) -> dict:
    suffix = Path(name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        path = f.name
    bar = st.progress(0.0, text="Starting ...")
    steps = {"n": 0}

    def log(msg: str) -> None:
        steps["n"] += 1
        bar.progress(min(0.95, steps["n"] / 6), text=msg)

    try:
        with st.spinner("Analysing -- detection and tracking take the longest ..."):
            result = app.analyse(path, log)
    finally:
        Path(path).unlink(missing_ok=True)
    bar.progress(1.0, text=f"Done in {result['processing_sec']} s")
    return result


if upload is not None:
    if upload.size > app.MAX_MB * 1024 * 1024:
        st.error(f"The file is larger than {app.MAX_MB} MB.")
        st.stop()
    key = (upload.name, upload.size)
    if st.session_state.get("key") != key:        # analyse once per upload, not per rerun
        st.session_state.pop("result", None)
        if st.button("Analyse", type="primary"):
            try:
                st.session_state["result"] = run(upload.getvalue(), upload.name)
                st.session_state["key"] = key
            except app.DemoError as e:
                st.error(str(e))
            except Exception:  # noqa: BLE001
                # Full traceback to the server log (Streamlit Cloud: "Manage app"
                # -> logs) AND behind an expander, so a failure on Cloud can be
                # diagnosed without a debugger.
                tb = traceback.format_exc()
                print(tb, file=sys.stderr, flush=True)
                st.error(f"Processing failed on this file. Try another clip (MP4, H.264, under "
                         f"{app.MAX_SEC:.0f} s).")
                with st.expander("Technical details"):
                    st.code(tb)

r = st.session_state.get("result")
if r and upload is not None:
    st.success(f"**{len(r['events'])} event(s)** in {r['duration_sec']} s of video "
               f"(processed in {r['processing_sec']} s).")
    st.info(r["note"])
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Annotated playback")
        st.video(r["annotated"])
    with right:
        st.subheader("Events")
        if r["events"]:
            st.dataframe([{"start (s)": round(s, 1), "end (s)": round(e, 1), "event": lab}
                          for s, e, lab in r["events"]], use_container_width=True, hide_index=True)
        else:
            st.write("No events detected.")
        st.download_button("prediction.json (harness format)", Path(r["json"]).read_bytes(),
                           file_name="prediction.json", mime="application/json")
    st.subheader("Event timeline and risk curve")
    st.pyplot(r["figure"])
    if r["clips"]:
        st.subheader("One clip per event")
        cols = st.columns(min(3, len(r["clips"])))
        for i, clip in enumerate(r["clips"]):
            with cols[i % len(cols)]:
                st.caption(Path(clip).stem.replace("_", " "))
                st.video(clip)
