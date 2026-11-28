"""Live cognitive-load dashboard with session recording and brief reports."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

sys.path.insert(0, str(Path(__file__).parent))

from browser_stream_state import BrowserStreamState
from browser_video import BrowserFaceVideoProcessor
from model import CognitiveLoadModel, LABEL_NAMES
from session_report import build_session_report, format_duration
from window_buffer import WindowFeatures

MODEL_PATH = Path(__file__).parent.parent / "data" / "trained_model.pkl"
SESSION_DIR = Path(__file__).parent.parent / "data" / "sessions"
HISTORY_ROWS = 300
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
}

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
    "browser_stream_state": None,
    "last_processed_sample_id": 0,
    "high_since": None,
    "alert_shown": False,
    "record_data": False,
    "alert_on": True,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

if st.session_state["browser_stream_state"] is None:
    st.session_state["browser_stream_state"] = BrowserStreamState()
browser_stream_state: BrowserStreamState = st.session_state["browser_stream_state"]


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


model = get_model()

st.title("🧠 Cognitive Load Monitor")
st.caption(
    "Facial-behavior estimates for reflection and research—not a clinical or "
    "diagnostic assessment."
)

with st.sidebar:
    st.header("Session controls")
    st.toggle("Record this session", key="record_data")
    st.toggle("High-load alert", key="alert_on")
    st.caption(
        "Start the browser camera below; recording is optional and stores estimates only."
    )
    if st.session_state["session_name"]:
        st.caption(f"Session: `{st.session_state['session_name']}`")
    if model is None:
        st.info("Rule-based estimates · no trained model found")
    else:
        st.success("Trained model loaded")


live_tab, report_tab, guidance_tab = st.tabs(
    ["Live monitor", "Session brief", "How to interpret"]
)

@st.fragment(run_every=1.0)
def render_live_monitor() -> None:
    stream_state: BrowserStreamState = st.session_state["browser_stream_state"]
    snapshot = stream_state.snapshot()
    alert_status = st.empty()

    if snapshot.error:
        st.error(snapshot.error)
    elif snapshot.running and snapshot.frames_received:
        if snapshot.face_found:
            st.success("Face detected")
        else:
            st.warning("No face detected — adjust lighting or position.")
    elif snapshot.running:
        st.info("Waiting for video frames from your browser.")
    else:
        st.info("Use **Start** in the video panel to grant browser camera access.")

    if (
        snapshot.latest_features is not None
        and snapshot.sample_timestamp is not None
        and snapshot.sample_id > st.session_state["last_processed_sample_id"]
    ):
        record = make_estimate(
            snapshot.latest_features,
            snapshot.sample_timestamp,
            snapshot.buffered_frames,
        )
        history = st.session_state["history"]
        history.append(record)
        st.session_state["history"] = history[-HISTORY_ROWS:]
        st.session_state["last_processed_sample_id"] = snapshot.sample_id

        if st.session_state["record_data"]:
            if st.session_state["session_name"] is None:
                st.session_state["session_name"] = datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )
                st.session_state["session_started_at"] = snapshot.sample_timestamp
            st.session_state["session_history"].append(record)

        update_load_alert(record, alert_status)

    if st.session_state["history"]:
        render_live_metrics(st.empty(), st.container())


@st.fragment(run_every=1.0)
def render_live_trends_fragment() -> None:
    if st.session_state["history"]:
        render_live_trends(st.container())


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
    camera_column, metrics_column = st.columns([1.25, 1])
    with camera_column:
        st.subheader("Browser camera")
        webrtc_streamer(
            key="cognitive-load-camera",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 15, "max": 20},
                },
                "audio": False,
            },
            video_processor_factory=lambda: BrowserFaceVideoProcessor(
                browser_stream_state
            ),
            async_processing=True,
            video_receiver_size=1,
        )
        st.caption(
            "Video is streamed to the app server for analysis. Raw frames are not "
            "saved; recorded sessions contain estimates only."
        )
    with metrics_column:
        render_live_monitor()
    render_live_trends_fragment()

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
