"""Build concise summaries from recorded cognitive-load estimates."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from statistics import mean, median
from typing import Any


def format_duration(seconds: float) -> str:
    """Format short spans in seconds and longer spans in minutes."""
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    return f"{seconds / 60.0:.1f} minutes"


def build_session_report(
    records: list[dict[str, Any]], session_name: str
) -> tuple[dict[str, Any], str]:
    """Return summary metrics and a Markdown brief for recorded samples."""
    if not records:
        raise ValueError("Cannot build a report without recorded session samples")

    ordered = sorted(records, key=lambda row: float(row["timestamp"]))
    timestamps = [float(row["timestamp"]) for row in ordered]
    intervals = [
        min(5.0, max(0.0, later - earlier))
        for earlier, later in zip(timestamps, timestamps[1:])
    ]
    typical_interval = median(intervals) if intervals else 0.0
    intervals.append(typical_interval)
    covered_seconds = sum(intervals)

    labels = [str(row.get("load_name", "Unknown")) for row in ordered]
    counts = Counter(labels)
    prediction_methods = Counter(
        str(row.get("prediction_mode", "Unknown")) for row in ordered
    )
    total = len(ordered)
    distribution = {
        label: counts.get(label, 0) / total * 100.0
        for label in ("Low", "Medium", "High")
    }

    high_seconds = sum(
        interval
        for label, interval in zip(labels, intervals)
        if label == "High"
    )

    def average(key: str) -> float | None:
        values = [
            float(row[key])
            for row in ordered
            if row.get(key) is not None
        ]
        return mean(values) if values else None

    started = datetime.fromtimestamp(timestamps[0]).astimezone()
    ended = datetime.fromtimestamp(timestamps[-1]).astimezone()
    most_common = counts.most_common(1)[0][0]
    metrics: dict[str, Any] = {
        "sample_count": total,
        "started": started,
        "ended": ended,
        "duration_seconds": max(0.0, timestamps[-1] - timestamps[0]),
        "covered_seconds": covered_seconds,
        "dominant_load": most_common,
        "distribution": distribution,
        "high_seconds": high_seconds,
        "high_percent": (
            high_seconds / covered_seconds * 100.0 if covered_seconds else 0.0
        ),
        "mean_blink_rate": average("blink_rate"),
        "mean_perclos": average("perclos"),
        "mean_ear": average("ear_mean_avg"),
        "mean_head_stability": average("head_stability"),
        "mean_face_quality_pct": average("face_quality_pct"),
        "mean_confidence": average("confidence"),
        "prediction_methods": dict(prediction_methods),
    }

    def display(value: float | None, suffix: str = "", precision: int = 1) -> str:
        return "Not available" if value is None else f"{value:.{precision}f}{suffix}"

    recommendations = {
        "High": "High-load estimates were most frequent. Review the task conditions and consider adding a short recovery break.",
        "Medium": "Medium-load estimates were most frequent. Review the task and compare this session with its NASA-TLX rating.",
        "Low": "Low-load estimates were most frequent. Compare this session with its NASA-TLX rating to validate the estimate.",
        "Unknown": "There were no classifiable load estimates in this session.",
    }
    report = "\n".join(
        [
            f"# Session Brief — {session_name}",
            "",
            f"- **Captured:** {started:%Y-%m-%d %H:%M:%S %Z} to {ended:%H:%M:%S %Z}",
            f"- **Observed span:** {format_duration(metrics['duration_seconds'])}",
            f"- **Recorded estimates:** {total}",
            f"- **Most frequent estimate:** {most_common}",
            "- **Prediction method(s):** "
            + ", ".join(
                f"{method} ({count})"
                for method, count in sorted(prediction_methods.items())
            ),
            "",
            "## Load distribution",
            *[
                f"- **{label}:** {distribution[label]:.1f}% ({counts.get(label, 0)} samples)"
                for label in ("Low", "Medium", "High")
            ],
            "",
            "## Average indicators",
            f"- **Blink rate:** {display(metrics['mean_blink_rate'], ' blinks/min')}",
            f"- **PERCLOS:** {display((metrics['mean_perclos'] or 0.0) * 100.0, '%')}",
            f"- **Eye aspect ratio:** {display(metrics['mean_ear'], precision=3)}",
            f"- **Face detection coverage:** {display(metrics['mean_face_quality_pct'], '%')}",
            f"- **Average model confidence:** {display(metrics['mean_confidence'], '%')}",
            "",
            "## Brief interpretation",
            recommendations.get(most_common, recommendations["Unknown"]),
            "",
            "> Estimates are behavioral signals, not a medical assessment. Interpret them alongside the task and self-reported NASA-TLX results.",
            "",
        ]
    )
    return metrics, report
