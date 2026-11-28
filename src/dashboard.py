"""
dashboard.py – Real-time Cognitive Load Estimator dashboard (Streamlit).

Run:  streamlit run src/dashboard.py

Layout
------
Left column  : Live webcam feed with landmark overlay + current load badge
Right column : Time-series graphs (EAR, blink rate, PERCLOS, load score)
Sidebar      : Controls – start/stop recording, export CSV, alert threshold
Bottom       : Session stats table
"""

from __future__ import annotations

import base64
import sys
import os
import time
import threading
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# -- ensure src/ is importable when launched from project root --------
sys.path.insert(0, str(Path(__file__).parent))

from capture import WebcamCapture
from face_mesh import FaceMeshExtractor
from features import FeatureExtractor, FrameFeatures
from window_buffer import SlidingWindowBuffer, WindowFeatures, FEATURE_NAMES
from model import CognitiveLoadModel, LABEL_NAMES, LABEL_COLORS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH   = Path(__file__).parent.parent / "data" / "trained_model.pkl"
SESSION_DIR  = Path(__file__).parent.parent / "data" / "sessions"
ALERT_SECS   = 60          # seconds of High load before alert fires
WINDOW_SECS  = 30
HISTORY_ROWS = 300         # rows kept in live graph

LOAD_BADGE = {
    "Low":    ("🟢 Low",    "#1a9850"),
    "Medium": ("🟡 Medium", "#fdae61"),
    "High":   ("🔴 High",   "#d73027"),
}


# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Cognitive Load Estimator",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Real-Time Cognitive Load Estimator")
st.caption("Powered by MediaPipe FaceMesh + Random Forest")


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    run_camera  = st.toggle("▶ Start Camera", value=False)
    record_data = st.toggle("⏺ Record Session", value=False)
    alert_on    = st.toggle("🔔 High-Load Alert (>60 s)", value=True)
    st.divider()
    st.subheader("Export")
    export_btn  = st.button("💾 Export Session CSV")
    st.divider()
    st.subheader("Model")
    model_info  = st.empty()


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
defaults = {
    "history":      [],         # list of dicts – one per window update
    "frame_buf":    [],         # raw FrameFeatures list for recording
    "high_since":   None,       # timestamp when last High load started
    "alert_shown":  False,
    "recording":    False,
    "session_name": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# Load model (graceful fallback if not yet trained)
# ---------------------------------------------------------------------------
@st.cache_resource
def _load_model():
    if MODEL_PATH.exists():
        try:
            return CognitiveLoadModel.load(MODEL_PATH)
        except Exception as e:
            st.warning(f"Could not load model: {e}. Using rule-based fallback.")
    return None

model = _load_model()
model_info.write("✅ Trained model loaded" if model else "⚠️ No model – rule-based fallback")


# ---------------------------------------------------------------------------
# Rule-based fallback when model is not yet trained
# ---------------------------------------------------------------------------
def _rule_based_load(wf: WindowFeatures) -> tuple[int, str]:
    score = 0
    if wf.blink_rate < 10 or wf.blink_rate > 25:
        score += 1
    if wf.perclos > 0.15:
        score += 1
    if wf.ear_mean_avg < 0.22:
        score += 1
    if wf.au4_mean < 0.6:    # brows closer = furrow
        score += 1
    if score >= 3:
        return 2, "High"
    if score >= 1:
        return 1, "Medium"
    return 0, "Low"


# ---------------------------------------------------------------------------
# Main camera loop (runs in Streamlit fragment to avoid full rerenders)
# ---------------------------------------------------------------------------
col_cam, col_charts = st.columns([1, 1])

frame_placeholder = col_cam.empty()
badge_placeholder  = col_cam.empty()
alert_placeholder  = col_cam.empty()

chart_ear    = col_charts.empty()
chart_blink  = col_charts.empty()
chart_perclos = col_charts.empty()
chart_load   = col_charts.empty()

stats_table  = st.empty()

# ---------------------------------------------------------------------------
# Export handler
# ---------------------------------------------------------------------------
if export_btn and st.session_state["history"]:
    df = pd.DataFrame(st.session_state["history"])
    sname = st.session_state["session_name"] or "session"
    fname = SESSION_DIR / f"{sname}.csv"
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(fname, index=False)
    st.sidebar.success(f"Saved → {fname.name}")


# ---------------------------------------------------------------------------
# Camera loop
# ---------------------------------------------------------------------------
if run_camera:
    if st.session_state["session_name"] is None:
        st.session_state["session_name"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    cam = WebcamCapture(source=0)
    cam.open()
    fme = FaceMeshExtractor()
    extractor = FeatureExtractor()
    buf = SlidingWindowBuffer(window_seconds=WINDOW_SECS)
    last_display = 0.0

    try:
        for bgr, rgb, ts in cam.frames():
            frame_started = time.perf_counter()
            result  = fme.process(rgb)
            ff      = extractor.extract(result)
            ff.timestamp = ts
            buf.push(ff)

            # -- overlay ---------------------------------------------------
            annotated = fme.draw_landmarks(bgr, result) if result.face_found else bgr
            if frame_started - last_display >= 0.2:
                encoded_ok, encoded_frame = cv2.imencode(
                    ".jpg",
                    annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 80],
                )
                if not encoded_ok:
                    raise RuntimeError("OpenCV failed to encode the webcam frame as JPEG")
                image_data = base64.b64encode(encoded_frame).decode("ascii")
                frame_placeholder.markdown(
                    f"<img src='data:image/jpeg;base64,{image_data}' "
                    "style='width:100%;height:auto' alt='Live webcam frame'>",
                    unsafe_allow_html=True,
                )
                last_display = frame_started

            # -- aggregate & predict every ~1 s when buffer is ready -------
            wf = buf.aggregate()
            if wf is not None:
                if model:
                    vec   = wf.to_vector()
                    label, proba = model.predict(vec)
                    load_name = LABEL_NAMES[label]
                else:
                    label, load_name = _rule_based_load(wf)

                entry = wf.to_dict()
                entry.update({"timestamp": ts, "load_label": label, "load_name": load_name})
                st.session_state["history"].append(entry)

                # Keep chart history bounded
                if len(st.session_state["history"]) > HISTORY_ROWS:
                    st.session_state["history"] = st.session_state["history"][-HISTORY_ROWS:]

                # -- badge -------------------------------------------------
                emoji_label, color = LOAD_BADGE[load_name]
                badge_placeholder.markdown(
                    f"<div style='background:{color};padding:10px;border-radius:8px;"
                    f"text-align:center;font-size:1.4em;color:white'>{emoji_label}</div>",
                    unsafe_allow_html=True,
                )

                # -- High-load alert ---------------------------------------
                if alert_on:
                    if load_name == "High":
                        if st.session_state["high_since"] is None:
                            st.session_state["high_since"] = ts
                        elif ts - st.session_state["high_since"] >= ALERT_SECS:
                            alert_placeholder.warning(
                                "⚠️ **High cognitive load detected for >60 s. "
                                "Consider taking a short break!**"
                            )
                            st.session_state["alert_shown"] = True
                    else:
                        st.session_state["high_since"] = None
                        if st.session_state["alert_shown"]:
                            alert_placeholder.empty()
                            st.session_state["alert_shown"] = False

                # -- charts ------------------------------------------------
                hist_df = pd.DataFrame(st.session_state["history"])
                chart_ear.line_chart(hist_df[["ear_mean_avg"]].rename(columns={"ear_mean_avg": "EAR"}))
                chart_blink.line_chart(hist_df[["blink_rate"]].rename(columns={"blink_rate": "Blink Rate (bpm)"}))
                chart_perclos.line_chart(hist_df[["perclos"]].rename(columns={"perclos": "PERCLOS"}))
                chart_load.line_chart(hist_df[["load_label"]].rename(columns={"load_label": "Load (0=Low 2=High)"}))

            # -- Streamlit: check for stop signal --------------------------
            if not run_camera:
                break

            frame_time = time.perf_counter() - frame_started
            time.sleep(max(0.0, (1.0 / 30.0) - frame_time))

    finally:
        cam.release()
        fme.close()

else:
    frame_placeholder.info("👆 Toggle **Start Camera** in the sidebar to begin.")
