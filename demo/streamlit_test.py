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
            except Exception:  # noqa: BLE001 - never a stack trace for a user
                st.error(f"Processing failed on this file. Try another clip (MP4, H.264, under "
                         f"{app.MAX_SEC:.0f} s).")

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
