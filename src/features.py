"""Facial and ocular feature extraction for cognitive-load estimation."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
EAR_BLINK_THRESHOLD = 0.21
BLINK_CONSEC_FRAMES = 2


def compute_ear(landmarks: np.ndarray, eye_indices: Sequence[int]) -> float:
    """Compute EAR using the standard six points ``p1..p6``."""
    if max(eye_indices) >= len(landmarks):
        raise ValueError("landmarks does not contain all requested eye indices")
    p1, p2, p3, p4, p5, p6 = (landmarks[i] for i in eye_indices)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal <= np.finfo(float).eps:
        return 0.0
    return float(
        (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5))
        / (2.0 * horizontal)
    )


@dataclass
class FrameFeatures:
    """Features computed for one captured frame."""

    timestamp: float = 0.0
    face_found: bool = False
    ear_mean: float = 0.0
    ear_left: float = 0.0
    ear_right: float = 0.0
    eye_closed: bool = False
    blink_flag: bool = False
    blink_rate: float = 0.0
    perclos: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    au4_brow: float = 0.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "timestamp": self.timestamp,
            "face_found": self.face_found,
            "ear": self.ear_mean,
            "left_ear": self.ear_left,
            "right_ear": self.ear_right,
            "eye_closed": self.eye_closed,
            "blink_flag": self.blink_flag,
            "blink_rate_per_min": self.blink_rate,
            "perclos": self.perclos,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "roll": self.roll,
            "brow_furrow": self.au4_brow,
            "gaze_x": self.gaze_x,
            "gaze_y": self.gaze_y,
        }


class OcularFeatureExtractor:
    """Track EAR, debounced blinks, blink rate, and PERCLOS."""

    def __init__(
        self,
        ear_threshold: float = EAR_BLINK_THRESHOLD,
        consec_frames: int = BLINK_CONSEC_FRAMES,
        window_seconds: float = 30.0,
    ) -> None:
        if ear_threshold <= 0 or consec_frames < 1 or window_seconds <= 0:
            raise ValueError("ocular thresholds and window_seconds must be positive")
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.window_seconds = window_seconds
        self.closed_frame_counter = 0
        self.total_blinks = 0
        self.eye_state_history: deque[tuple[float, bool]] = deque()
        self.blink_timestamps: deque[float] = deque()

    def update(
        self, landmarks: np.ndarray, timestamp: Optional[float] = None
    ) -> dict[str, float | int | bool]:
        """Update state and return current ocular metrics."""
        now = time.time() if timestamp is None else timestamp
        left_ear = compute_ear(landmarks, LEFT_EYE_IDX)
        right_ear = compute_ear(landmarks, RIGHT_EYE_IDX)
        avg_ear = (left_ear + right_ear) / 2.0
        is_closed = avg_ear < self.ear_threshold
        self.eye_state_history.append((now, is_closed))

        blink_detected = False
        if is_closed:
            self.closed_frame_counter += 1
        else:
            if self.closed_frame_counter >= self.consec_frames:
                self.total_blinks += 1
                self.blink_timestamps.append(now)
                blink_detected = True
            self.closed_frame_counter = 0

        cutoff = now - self.window_seconds
        while self.eye_state_history and self.eye_state_history[0][0] < cutoff:
            self.eye_state_history.popleft()
        while self.blink_timestamps and self.blink_timestamps[0] < cutoff:
            self.blink_timestamps.popleft()

        elapsed = max(1.0, now - self.eye_state_history[0][0])
        perclos = (
            sum(closed for _, closed in self.eye_state_history)
            / len(self.eye_state_history)
            if self.eye_state_history
            else 0.0
        )
        return {
            "ear": round(avg_ear, 3),
            "left_ear": round(left_ear, 3),
            "right_ear": round(right_ear, 3),
            "eye_closed": is_closed,
            "blink_flag": blink_detected,
            "blink_rate": round(len(self.blink_timestamps) / elapsed * 60.0, 1),
            "perclos": round(perclos, 3),
            "total_blinks": self.total_blinks,
        }


class FeatureExtractor:
    """Compute the multi-feature frame vector consumed by the buffer."""

    _POSE_INDICES = [1, 152, 33, 263, 61, 291]
    _POSE_MODEL = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0),
        ],
        dtype=np.float64,
    )

    def __init__(self) -> None:
        self.ocular = OcularFeatureExtractor()

    def extract(
        self,
        result: object,
        timestamp: Optional[float] = None,
    ) -> FrameFeatures:
        """Extract features from a ``FaceMeshResult``-like object."""
        landmarks = getattr(result, "landmarks", result)
        now = time.time() if timestamp is None else timestamp
        if landmarks is None:
            return FrameFeatures(timestamp=now)

        landmarks = np.asarray(landmarks, dtype=np.float32)
        if len(landmarks) <= max(self._POSE_INDICES + LEFT_EYE_IDX + RIGHT_EYE_IDX):
            raise ValueError("Face landmarks array is missing required points")

        ocular = self.ocular.update(landmarks, timestamp=now)
        pitch, yaw, roll = self._head_pose(landmarks)
        brow = self._brow_furrow(landmarks)
        gaze_x, gaze_y = self._gaze(landmarks)
        return FrameFeatures(
            timestamp=now,
            face_found=True,
            ear_mean=float(ocular["ear"]),
            ear_left=float(ocular["left_ear"]),
            ear_right=float(ocular["right_ear"]),
            eye_closed=bool(ocular["eye_closed"]),
            blink_flag=bool(ocular["blink_flag"]),
            blink_rate=float(ocular["blink_rate"]),
            perclos=float(ocular["perclos"]),
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            au4_brow=brow,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
        )

    def _head_pose(self, landmarks: np.ndarray) -> tuple[float, float, float]:
        image_points = landmarks[self._POSE_INDICES].astype(np.float64)
        width = max(float(np.ptp(landmarks[:, 0])), 1.0)
        height = max(float(np.ptp(landmarks[:, 1])), 1.0)
        focal = max(width, height)
        center = np.mean(landmarks, axis=0)
        camera_matrix = np.array(
            [[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]],
            dtype=np.float64,
        )
        ok, rotation, _ = cv2.solvePnP(
            self._POSE_MODEL,
            image_points,
            camera_matrix,
            np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return 0.0, 0.0, 0.0
        matrix, _ = cv2.Rodrigues(rotation)
        sy = float(np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2))
        if sy < 1e-6:
            pitch = np.degrees(np.arctan2(-matrix[2, 1], matrix[1, 1]))
            yaw = np.degrees(np.arctan2(-matrix[1, 0], sy))
            roll = 0.0
        else:
            pitch = np.degrees(np.arctan2(matrix[2, 1], matrix[2, 2]))
            yaw = np.degrees(np.arctan2(-matrix[2, 0], sy))
            roll = np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))
        return float(pitch), float(yaw), float(roll)

    @staticmethod
    def _brow_furrow(landmarks: np.ndarray) -> float:
        brow = np.mean(landmarks[[70, 63, 105, 107]], axis=0)
        eye = np.mean(landmarks[[33, 133, 362, 263]], axis=0)
        scale = max(np.linalg.norm(landmarks[33] - landmarks[263]), 1.0)
        return float(np.clip(np.linalg.norm(brow - eye) / scale, 0.0, 2.0))

    @staticmethod
    def _gaze(landmarks: np.ndarray) -> tuple[float, float]:
        if len(landmarks) < 478:
            return 0.0, 0.0
        left = np.mean(landmarks[[468, 469, 470, 471]], axis=0)
        right = np.mean(landmarks[[473, 474, 475, 476]], axis=0)
        center = np.mean(landmarks[[362, 263, 33, 133]], axis=0)
        scale = max(np.linalg.norm(landmarks[33] - landmarks[263]), 1.0)
        gaze = (np.mean((left, right), axis=0) - center) / scale
        return float(gaze[0]), float(gaze[1])
