"""
window_buffer.py – 30-second sliding-window aggregation.

Aggregates per-frame FrameFeatures into a single feature vector
suitable for passing to the ML model (src/model.py).

Aggregated features (20 total)
--------------------------------
ear_mean_avg, ear_mean_std
ear_left_avg, ear_right_avg
blink_rate        – blinks per minute over the window
perclos           – fraction of frames with eye_closed == True
pitch_mean, pitch_std
yaw_mean, yaw_std
roll_mean, roll_std
head_stability    – 1 / (std of combined head motion + ε)
au4_mean, au4_std – brow-furrow activity
gaze_x_std, gaze_y_std  – fixation stability
n_frames          – usable frames in window (quality metric)
window_duration   – actual seconds covered
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Deque, Dict, Optional

import numpy as np

try:
    from .features import FrameFeatures
except ImportError:
    from features import FrameFeatures

# ---------------------------------------------------------------------------
# Aggregated vector container
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "ear_mean_avg", "ear_mean_std",
    "ear_left_avg", "ear_right_avg",
    "blink_rate",
    "perclos",
    "pitch_mean", "pitch_std",
    "yaw_mean",   "yaw_std",
    "roll_mean",  "roll_std",
    "head_stability",
    "au4_mean", "au4_std",
    "gaze_x_std", "gaze_y_std",
    "n_frames",
    "window_duration",
]


@dataclass
class WindowFeatures:
    ear_mean_avg:    float = 0.0
    ear_mean_std:    float = 0.0
    ear_left_avg:    float = 0.0
    ear_right_avg:   float = 0.0
    blink_rate:      float = 0.0   # blinks / minute
    perclos:         float = 0.0   # 0-1
    pitch_mean:      float = 0.0
    pitch_std:       float = 0.0
    yaw_mean:        float = 0.0
    yaw_std:         float = 0.0
    roll_mean:       float = 0.0
    roll_std:        float = 0.0
    head_stability:  float = 0.0   # higher = more stable
    au4_mean:        float = 0.0
    au4_std:         float = 0.0
    gaze_x_std:      float = 0.0
    gaze_y_std:      float = 0.0
    n_frames:        int   = 0
    window_duration: float = 0.0   # seconds

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.ear_mean_avg, self.ear_mean_std,
            self.ear_left_avg, self.ear_right_avg,
            self.blink_rate,
            self.perclos,
            self.pitch_mean, self.pitch_std,
            self.yaw_mean,   self.yaw_std,
            self.roll_mean,  self.roll_std,
            self.head_stability,
            self.au4_mean, self.au4_std,
            self.gaze_x_std, self.gaze_y_std,
            float(self.n_frames),
            self.window_duration,
        ], dtype=np.float32)

    def to_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


# ---------------------------------------------------------------------------
# Sliding-window buffer
# ---------------------------------------------------------------------------
class SlidingWindowBuffer:
    """
    Maintains a deque of FrameFeatures timestamped at capture time.
    Call push() after every frame; call aggregate() to get the latest
    WindowFeatures (over the past `window_seconds` seconds).

    Parameters
    ----------
    window_seconds : float
        Length of the rolling window (default: 30 s).
    min_frames : int
        Minimum valid frames required before returning a result.
    """

    def __init__(self, window_seconds: float = 30.0, min_frames: int = 15):
        self.window_seconds = window_seconds
        self.min_frames = min_frames
        self._buffer: Deque[FrameFeatures] = deque()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def push(self, ff: FrameFeatures) -> None:
        """Add a new FrameFeatures to the buffer and evict old entries."""
        self._buffer.append(ff)
        self._evict()

    def aggregate(self) -> Optional[WindowFeatures]:
        """
        Compute aggregated WindowFeatures over the current window.
        Returns None if fewer than min_frames valid frames are present.
        """
        valid = [f for f in self._buffer if f.face_found]
        if len(valid) < self.min_frames:
            return None

        # Arrays for vectorised stats
        ear_m  = np.array([f.ear_mean  for f in valid])
        ear_l  = np.array([f.ear_left  for f in valid])
        ear_r  = np.array([f.ear_right for f in valid])
        pitch  = np.array([f.pitch     for f in valid])
        yaw    = np.array([f.yaw       for f in valid])
        roll   = np.array([f.roll      for f in valid])
        au4    = np.array([f.au4_brow  for f in valid])
        gaze_x = np.array([f.gaze_x    for f in valid])
        gaze_y = np.array([f.gaze_y    for f in valid])
        closed = np.array([f.eye_closed for f in valid], dtype=float)

        # Blink count: rising-edge detection on blink_flag
        blink_flags = [f.blink_flag for f in valid]
        blinks = sum(
            1 for i in range(1, len(blink_flags))
            if blink_flags[i] and not blink_flags[i - 1]
        )

        t0 = valid[0].timestamp
        t1 = valid[-1].timestamp
        duration = max(t1 - t0, 1e-3)  # seconds
        blink_rate = blinks / duration * 60.0  # per minute

        # Head stability: inverse of combined angular variance
        head_var = float(np.var(pitch) + np.var(yaw) + np.var(roll))
        head_stability = 1.0 / (head_var + 1e-3)

        return WindowFeatures(
            ear_mean_avg   = float(np.mean(ear_m)),
            ear_mean_std   = float(np.std(ear_m)),
            ear_left_avg   = float(np.mean(ear_l)),
            ear_right_avg  = float(np.mean(ear_r)),
            blink_rate     = float(blink_rate),
            perclos        = float(np.mean(closed)),
            pitch_mean     = float(np.mean(pitch)),
            pitch_std      = float(np.std(pitch)),
            yaw_mean       = float(np.mean(yaw)),
            yaw_std        = float(np.std(yaw)),
            roll_mean      = float(np.mean(roll)),
            roll_std       = float(np.std(roll)),
            head_stability = head_stability,
            au4_mean       = float(np.mean(au4)),
            au4_std        = float(np.std(au4)),
            gaze_x_std     = float(np.std(gaze_x)),
            gaze_y_std     = float(np.std(gaze_y)),
            n_frames       = len(valid),
            window_duration= float(duration),
        )

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------
    def _evict(self) -> None:
        cutoff = time.time() - self.window_seconds
        while self._buffer and self._buffer[0].timestamp < cutoff:
            self._buffer.popleft()
