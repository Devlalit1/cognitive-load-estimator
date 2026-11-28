"""
nasa_tlx.py – Minimal NASA-TLX survey (Streamlit page).

Run:  streamlit run experiments/nasa_tlx.py

Saves JSON + CSV to data/labels/<session_id>_tlx.csv
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

LABELS_DIR = Path(__file__).parent.parent / "data" / "labels"

DIMENSIONS = [
    ("mental_demand",   "Mental Demand",   "How much mental and perceptual activity was required?"),
    ("physical_demand", "Physical Demand", "How much physical activity was required?"),
    ("temporal_demand", "Temporal Demand", "How much time pressure did you feel?"),
    ("performance",     "Performance",     "How successful were you in accomplishing the task? (lower = better)"),
    ("effort",          "Effort",          "How hard did you have to work to attain your performance level?"),
    ("frustration",     "Frustration",     "How insecure, discouraged, irritated were you?"),
]

st.set_page_config(page_title="NASA-TLX Survey", page_icon="📋")
st.title("📋 NASA Task Load Index (TLX) Survey")
st.markdown(
    "Rate each dimension on a scale of **0–100** (0 = very low, 100 = very high). "
    "Take this survey immediately **after each task**."
)
st.divider()

session_id = st.text_input("Session ID (matches your feature CSV)", value=datetime.now().strftime("%Y%m%d_%H%M%S"))
task_label = st.selectbox("Task just completed", ["Low Load (relaxing video)", "Medium Load (article + MCQ)", "High Load (N-back / Stroop)"])

scores: dict[str, int] = {}
for key, label, description in DIMENSIONS:
    scores[key] = st.slider(f"**{label}**  \n*{description}*", 0, 100, 50, step=5)

st.divider()

# Weighted NASA-TLX (equal weights → simple mean)
raw_tlx = sum(scores.values()) / len(scores)
st.metric("Raw NASA-TLX Score", f"{raw_tlx:.1f} / 100")

col1, col2 = st.columns(2)
if col1.button("✅ Submit & Save"):
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "session_id": session_id,
        "task_label": task_label,
        "timestamp": datetime.now().isoformat(),
        "raw_tlx": raw_tlx,
        **scores,
    }
    # CSV
    csv_path = LABELS_DIR / f"{session_id}_tlx.csv"
    df = pd.DataFrame([record])
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    st.success(f"Saved → {csv_path}")
    st.json(record)

if col2.button("🔄 Reset"):
    st.rerun()
