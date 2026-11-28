"""Thread-safe state shared between WebRTC video processing and Streamlit UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from threading import Lock

try:
    from .window_buffer import WindowFeatures
except ImportError:
    from window_buffer import WindowFeatures


@dataclass(frozen=True)
class BrowserStreamSnapshot:
    running: bool = False
    frames_received: int = 0
    face_found: bool = False
    buffered_frames: int = 0
    sample_id: int = 0
    latest_features: WindowFeatures | None = None
    sample_timestamp: float | None = None
    error: str | None = None


class BrowserStreamState:
    """Publish immutable snapshots across the video worker and UI thread."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._snapshot = BrowserStreamSnapshot()

    def begin(self) -> None:
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                running=True,
                frames_received=0,
                face_found=False,
                buffered_frames=0,
                latest_features=None,
                sample_timestamp=None,
                error=None,
            )

    def publish_frame(self, face_found: bool, buffered_frames: int) -> None:
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                frames_received=self._snapshot.frames_received + 1,
                face_found=face_found,
                buffered_frames=buffered_frames,
            )

    def publish_features(
        self,
        features: WindowFeatures,
        timestamp: float,
        buffered_frames: int,
    ) -> None:
        copied_features = WindowFeatures(**asdict(features))
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                sample_id=self._snapshot.sample_id + 1,
                latest_features=copied_features,
                sample_timestamp=timestamp,
                buffered_frames=buffered_frames,
            )

    def set_error(self, message: str) -> None:
        with self._lock:
            self._snapshot = replace(self._snapshot, error=message)

    def stop(self) -> None:
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                running=False,
                face_found=False,
            )

    def snapshot(self) -> BrowserStreamSnapshot:
        with self._lock:
            snapshot = self._snapshot
        if snapshot.latest_features is not None:
            snapshot = replace(
                snapshot,
                latest_features=WindowFeatures(**asdict(snapshot.latest_features)),
            )
        return snapshot
