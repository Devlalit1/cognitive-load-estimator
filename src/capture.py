"""OpenCV webcam capture helpers."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Optional

import cv2
import numpy as np


class WebcamCapture:
    """Capture BGR frames and their RGB conversion from an OpenCV camera."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        *,
        source: Optional[int] = None,
    ) -> None:
        self.camera_index = camera_index if source is None else source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Open the configured camera, raising a useful error on failure."""
        if self.cap is not None and self.cap.isOpened():
            return

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Could not open webcam index {self.camera_index}. "
                "Check camera permissions, index, and that no other app is using it."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap = cap

    def read_frame(self) -> tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Return ``(success, bgr_frame, rgb_frame)`` for one frame."""
        self.open()
        assert self.cap is not None
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None, None
        return True, frame, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def frames(self) -> Iterator[tuple[np.ndarray, np.ndarray, float]]:
        """Yield ``(bgr, rgb, timestamp)`` until a frame cannot be read."""
        while True:
            ok, bgr, rgb = self.read_frame()
            if not ok or bgr is None or rgb is None:
                return
            yield bgr, rgb, time.time()

    def release(self) -> None:
        """Release the camera and destroy OpenCV windows safely."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def __enter__(self) -> "WebcamCapture":
        self.open()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.release()
