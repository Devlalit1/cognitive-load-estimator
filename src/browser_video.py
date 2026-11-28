"""Browser webcam frame processing for Streamlit WebRTC."""

from __future__ import annotations

import logging
import time

import cv2
from av import VideoFrame
from streamlit_webrtc import VideoProcessorBase

if __package__:
    from .browser_stream_state import BrowserStreamState
    from .face_mesh import FaceMeshDetector
    from .features import FeatureExtractor
    from .window_buffer import SlidingWindowBuffer
else:
    from browser_stream_state import BrowserStreamState
    from face_mesh import FaceMeshDetector
    from features import FeatureExtractor
    from window_buffer import SlidingWindowBuffer

logger = logging.getLogger(__name__)

WINDOW_SECS = 30
SAMPLE_INTERVAL_SECS = 1.0


class BrowserFaceVideoProcessor(VideoProcessorBase):
    """Receive browser frames, extract face features, and return annotations."""

    def __init__(self, shared_state: BrowserStreamState) -> None:
        self._shared_state = shared_state
        self._shared_state.begin()
        self._face_mesh: FaceMeshDetector | None = None
        self._feature_extractor = FeatureExtractor()
        self._window_buffer = SlidingWindowBuffer(window_seconds=WINDOW_SECS)
        self._last_sample_ts = 0.0
        try:
            self._face_mesh = FaceMeshDetector()
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            self._shared_state.set_error(
                f"Face detector could not start: {type(exc).__name__}: {exc}"
            )
            raise

    def recv(self, frame: VideoFrame) -> VideoFrame:
        if self._face_mesh is None:
            raise RuntimeError("The face detector is not initialized")

        try:
            bgr_frame = frame.to_ndarray(format="bgr24")
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            now = time.time()
            result = self._face_mesh.process(rgb_frame)
            frame_features = self._feature_extractor.extract(result, timestamp=now)
            self._window_buffer.push(frame_features)
            self._shared_state.publish_frame(
                result.face_found, self._window_buffer.buffer_size
            )

            if now - self._last_sample_ts >= SAMPLE_INTERVAL_SECS:
                self._last_sample_ts = now
                window_features = self._window_buffer.aggregate()
                if window_features is not None:
                    self._shared_state.publish_features(
                        window_features,
                        now,
                        self._window_buffer.buffer_size,
                    )

            annotated = self._face_mesh.draw_landmarks(bgr_frame, result)
            output_frame = VideoFrame.from_ndarray(annotated, format="bgr24")
            output_frame.pts = frame.pts
            output_frame.time_base = frame.time_base
            return output_frame
        except (OSError, RuntimeError, ValueError, cv2.error) as exc:
            self._shared_state.set_error(
                f"Video processing failed: {type(exc).__name__}: {exc}"
            )
            logger.exception("Browser webcam frame processing failed")
            raise

    def on_ended(self) -> None:
        try:
            if self._face_mesh is not None:
                self._face_mesh.close()
                self._face_mesh = None
        finally:
            self._shared_state.stop()
