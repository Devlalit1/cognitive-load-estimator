"""Live cognitive-load dashboard with session recording and brief reports."""

from __future__ import annotations

import base64
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from capture import WebcamCapture
from face_mesh import FaceMeshExtractor
from features import FeatureExtractor
from model import CognitiveLoadModel, LABEL_NAMES
from session_report import build_session_report, format_duration
from window_buffer import SlidingWindowBuffer, WindowFeatures

MODEL_PATH = Path(__file__).parent.parent / "data" / "trained_model.pkl"
SESSION_DIR = Path(__file__).parent.parent / "data" / "sessions"
WINDOW_SECS = 30
SAMPLE_INTERVAL_SECS = 1.0
DISPLAY_INTERVAL_SECS = 0.2
HISTORY_ROWS = 300

LOAD_STYLE = {
    "Low": ("🟢", "#1a9850"),
    "Medium": ("🟡", "#fdae61"),
    "High": ("🔴", "#d73027"),
    "Unknown": ("⚪", "#657080"),
}

st.set_page_config(
    page_title="Cognitive Load Estimator",
    page_icon="🧠",
    layout="wide",
)

DEFAULTS = {
    "history": [],
    "session_history": [],
    "session_name": None,
    "session_started_at": None,
    "camera": None,
    "face_mesh": None,
    "feature_extractor": None,
    "window_buffer": None,
    "last_sample_ts": 0.0,
    "last_display_ts": 0.0,
    "camera_error": None,
    "high_since": None,
    "alert_shown": False,
    "run_camera": False,
    "record_data": False,
    "alert_on": True,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


@st.cache_resource
def get_model() -> CognitiveLoadModel | None:
    if not MODEL_PATH.exists():
        return None
    try:
        return CognitiveLoadModel.load(MODEL_PATH)
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f"Trained model could not be loaded: {exc}")
        return None


def rule_based_load(features: WindowFeatures) -> tuple[int, str]:
    score = 0
    if features.blink_rate < 10 or features.blink_rate > 25:
        score += 1
    if features.perclos > 0.15:
        score += 1
    if features.ear_mean_avg < 0.22:
        score += 1
    if features.au4_mean < 0.6:
        score += 1
    if score >= 3:
        return 2, "High"
    if score >= 1:
        return 1, "Medium"
    return 0, "Low"


def close_camera_resources() -> None:
    camera = st.session_state.get("camera")
    if camera is not None:
        camera.release()
        st.session_state["camera"] = None

    face_mesh = st.session_state.get("face_mesh")
    if face_mesh is not None:
        face_mesh.close()
        st.session_state["face_mesh"] = None

    st.session_state["feature_extractor"] = None
    st.session_state["window_buffer"] = None
    st.session_state["last_sample_ts"] = 0.0
    st.session_state["last_display_ts"] = 0.0


model = get_model()

st.title("🧠 Cognitive Load Monitor")
st.caption(
    "Facial-behavior estimates for reflection and research—not a clinical or "
    "diagnostic assessment."
)

with st.sidebar:
    st.header("Session controls")
    st.toggle("Record this session", key="record_data")
    st.toggle("Start camera", key="run_camera")
    st.toggle("High-load alert", key="alert_on")
    st.caption("Turn on recording before or during a live session.")
    if st.session_state["session_name"]:
        st.caption(f"Session: `{st.session_state['session_name']}`")
    if model is None:
        st.info("Rule-based estimates · no trained model found")
    else:
        st.success("Trained model loaded")


live_tab, report_tab, guidance_tab = st.tabs(
    ["Live monitor", "Session brief", "How to interpret"]
)

@st.fragment(run_every=0.1)
def render_live_monitor() -> None:
    camera_column, metrics_column = st.columns([1.25, 1])
    camera_status = camera_column.empty()
    camera_frame = camera_column.empty()
    live_load = metrics_column.empty()
    metric_row = metrics_column.container()
    trend_container = st.container()
    alert_status = st.empty()

    if not st.session_state["run_camera"]:
        close_camera_resources()
        if st.session_state["camera_error"]:
            camera_status.error(st.session_state["camera_error"])
            st.session_state["camera_error"] = None
        else:
            camera_status.info("Turn on **Start camera** to begin.")
        if st.session_state["history"]:
            render_live_metrics(live_load, metric_row)
            render_live_trends(trend_container)
        return

    if st.session_state["camera_error"]:
        camera_status.error(st.session_state["camera_error"])
        return

    try:
        if st.session_state["camera"] is None:
            camera = WebcamCapture(source=0)
            camera.open()
            st.session_state["camera"] = camera
        if st.session_state["face_mesh"] is None:
            st.session_state["face_mesh"] = FaceMeshExtractor()
            st.session_state["feature_extractor"] = FeatureExtractor()
            st.session_state["window_buffer"] = SlidingWindowBuffer(
                window_seconds=WINDOW_SECS
            )
    except Exception as exc:
        close_camera_resources()
        st.session_state["camera_error"] = (
            f"Camera or face detector could not start: {type(exc).__name__}: {exc}. "
            "Check camera access and the Face Landmarker model."
        )
        camera_status.error(st.session_state["camera_error"])
        return

    camera = st.session_state["camera"]
    face_mesh = st.session_state["face_mesh"]
    ok, bgr_frame, rgb_frame = camera.read_frame()
    if not ok or bgr_frame is None or rgb_frame is None:
        camera_status.error("Webcam frame capture failed. Check the camera connection.")
        close_camera_resources()
        return

    now = time.time()
    result = face_mesh.process(rgb_frame)
    frame_features = st.session_state["feature_extractor"].extract(
        result, timestamp=now
    )
    buffer = st.session_state["window_buffer"]
    buffer.push(frame_features)

    annotated = face_mesh.draw_landmarks(bgr_frame, result)
    if now - st.session_state["last_display_ts"] >= DISPLAY_INTERVAL_SECS:
        encoded_ok, encoded_frame = cv2.imencode(
            ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 78]
        )
        if not encoded_ok:
            raise RuntimeError("OpenCV failed to encode the webcam frame as JPEG")
        image_data = base64.b64encode(encoded_frame).decode("ascii")
        camera_frame.markdown(
            f"<img src='data:image/jpeg;base64,{image_data}' "
            "style='width:100%;height:auto;border-radius:10px' "
            "alt='Live webcam frame'>",
            unsafe_allow_html=True,
        )
        st.session_state["last_display_ts"] = now
    camera_status.caption(
        "Face detected" if result.face_found else "No face detected — adjust lighting or position"
    )

    if now - st.session_state["last_sample_ts"] >= SAMPLE_INTERVAL_SECS:
        window_features = buffer.aggregate()
        st.session_state["last_sample_ts"] = now
        if window_features is not None:
            record = make_estimate(window_features, now, buffer.buffer_size)
            history = st.session_state["history"]
            history.append(record)
            st.session_state["history"] = history[-HISTORY_ROWS:]

            if st.session_state["record_data"]:
                if st.session_state["session_name"] is None:
                    st.session_state["session_name"] = datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )
                    st.session_state["session_started_at"] = now
                st.session_state["session_history"].append(record)

            render_live_metrics(live_load, metric_row)
            render_live_trends(trend_container)
            update_load_alert(record, alert_status)
        else:
            camera_status.caption(
                "Collecting valid face frames; estimates appear after enough landmarks "
                "are available."
            )


def make_estimate(
    features: WindowFeatures, timestamp: float, buffered_frames: int
) -> dict[str, float | int | str | None]:
    confidence: float | None = None
    if model is not None:
        label, probabilities = model.predict(features.to_vector())
        load_name = LABEL_NAMES[label]
        confidence = max(probabilities.values(), default=0.0) * 100.0
        prediction_mode = "Trained model"
    else:
        label, load_name = rule_based_load(features)
        prediction_mode = "Rule-based fallback"

    record = features.to_dict()
    record.update(
        {
            "timestamp": timestamp,
            "load_label": label,
            "load_name": load_name,
            "confidence": confidence,
            "prediction_mode": prediction_mode,
            "face_quality_pct": (
                features.n_frames / max(buffered_frames, 1) * 100.0
            ),
        }
    )
    return record


def render_live_metrics(live_load: Any, metric_row: Any) -> None:
    history = st.session_state["history"]
    if not history:
        return
    latest = history[-1]
    name = str(latest["load_name"])
    emoji, color = LOAD_STYLE.get(name, LOAD_STYLE["Unknown"])
    confidence = latest.get("confidence")
    confidence_text = (
        f"{float(confidence):.0f}% model confidence"
        if confidence is not None
        else "Heuristic estimate"
    )
    live_load.markdown(
        f"<div style='background:{color};padding:14px 18px;border-radius:10px;"
        f"color:white;font-size:1.5rem;font-weight:650'>"
        f"{emoji} Current estimate: {name}"
        f"<div style='font-size:.85rem;font-weight:400'>{confidence_text}</div></div>",
        unsafe_allow_html=True,
    )
    with metric_row:
        columns = st.columns(4)
        columns[0].metric("Blink rate", f"{float(latest['blink_rate']):.1f}/min")
        columns[1].metric("PERCLOS", f"{float(latest['perclos']) * 100:.1f}%")
        columns[2].metric("Mean EAR", f"{float(latest['ear_mean_avg']):.3f}")
        columns[3].metric("Face coverage", f"{float(latest['face_quality_pct']):.0f}%")


def render_live_trends(trend_container: Any) -> None:
    if not st.session_state["history"]:
        return
    data = pd.DataFrame(st.session_state["history"])
    data["time"] = pd.to_datetime(data["timestamp"], unit="s")
    data = data.set_index("time")
    with trend_container:
        st.subheader("Recent trends")
        chart_columns = st.columns(2)
        chart_columns[0].line_chart(data[["ear_mean_avg"]].rename(columns={"ear_mean_avg": "EAR"}))
        chart_columns[1].line_chart(data[["blink_rate"]].rename(columns={"blink_rate": "Blinks/min"}))
        chart_columns[0].line_chart(data[["perclos"]].rename(columns={"perclos": "PERCLOS"}))
        chart_columns[1].line_chart(
            data[["load_label"]].rename(columns={"load_label": "Load class · 0 Low / 2 High"})
        )


def update_load_alert(
    record: dict[str, float | int | str | None], alert_status: Any
) -> None:
    if not st.session_state["alert_on"]:
        alert_status.empty()
        st.session_state["high_since"] = None
        return

    timestamp = float(record["timestamp"])
    if record["load_name"] == "High":
        if st.session_state["high_since"] is None:
            st.session_state["high_since"] = timestamp
        elif timestamp - st.session_state["high_since"] >= 60:
            alert_status.warning(
                "High-load estimates have persisted for about one minute. "
                "Consider a pause and check how you feel."
            )
    else:
        st.session_state["high_since"] = None
        alert_status.empty()


@st.fragment(run_every=1.0)
def render_session_brief() -> None:
    records = st.session_state["session_history"]
    if not records:
        st.info(
            "Enable **Record this session** and start the camera. "
            "A brief report becomes available after the first valid estimate."
        )
        return

    session_name = st.session_state["session_name"] or "cognitive_load_session"
    metrics, report = build_session_report(records, session_name)
    st.subheader("Session at a glance")
    summary_columns = st.columns(4)
    summary_columns[0].metric("Recorded estimates", metrics["sample_count"])
    summary_columns[1].metric("Dominant estimate", metrics["dominant_load"])
    summary_columns[2].metric("High-load share", f"{metrics['high_percent']:.1f}%")
    summary_columns[3].metric(
        "Observed span", format_duration(metrics["duration_seconds"])
    )

    distribution = pd.DataFrame.from_dict(
        metrics["distribution"], orient="index", columns=["Share (%)"]
    )
    report_columns = st.columns([1, 1.5])
    with report_columns[0]:
        st.caption("Distribution of recorded load estimates")
        st.bar_chart(distribution, horizontal=True)
    with report_columns[1]:
        st.markdown(report)

    csv_data = pd.DataFrame(records).to_csv(index=False)
    st.download_button(
        "Download session data (CSV)",
        data=csv_data,
        file_name=f"{session_name}.csv",
        mime="text/csv",
        key="download_session_csv",
    )
    if st.button("Save CSV in project data/sessions", key="save_session_csv"):
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = SESSION_DIR / f"{session_name}.csv"
        csv_path.write_text(csv_data, encoding="utf-8")
        st.success(f"Saved session data to {csv_path}")
    st.download_button(
        "Download brief report (Markdown)",
        data=report,
        file_name=f"{session_name}_brief.md",
        mime="text/markdown",
        key="download_session_brief",
    )


with live_tab:
    render_live_monitor()

with report_tab:
    st.header("Session brief")
    st.caption(
        "A concise summary of estimates recorded while **Record this session** was enabled."
    )
    render_session_brief()

with guidance_tab:
    st.header("Reading the estimates")
    st.markdown(
        """
        - **EAR** summarizes eye openness; lower values can indicate eye closure.
        - **Blink rate** and **PERCLOS** summarize ocular behavior over the rolling window.
        - **Head pose, brow proxy, and gaze stability** are approximate visual features.
        - **Low / Medium / High** are model estimates, not direct measurements of mental state.
        - Compare results with the task context and self-reported NASA-TLX ratings.

        For a useful report, record a complete task and label it separately with
        its NASA-TLX score. Lighting, camera position, glasses, and face visibility
        can affect the measurements.
        """
    )
