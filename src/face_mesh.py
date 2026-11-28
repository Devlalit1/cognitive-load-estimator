"""MediaPipe face-landmark extraction with legacy and Tasks API support."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceMeshResult:
    """Landmarks and optional MediaPipe outputs for one RGB frame."""

    landmarks: Optional[np.ndarray]
    raw: Any = None
    blendshapes: Any = None
    transformation_matrix: Any = None

    @property
    def face_found(self) -> bool:
        return self.landmarks is not None


class FaceMeshDetector:
    """Extract 2D pixel coordinates for one face.

    MediaPipe versions before 1.0 expose ``mp.solutions.face_mesh``. Newer
    versions expose the Tasks ``FaceLandmarker`` API and require a model asset.
    Set ``FACE_LANDMARKER_MODEL`` or pass ``model_path`` for the newer API.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: Optional[str | os.PathLike[str]] = None,
    ) -> None:
        if not 0.0 <= min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be between 0 and 1")
        if not 0.0 <= min_tracking_confidence <= 1.0:
            raise ValueError("min_tracking_confidence must be between 0 and 1")

        self._legacy = None
        self._landmarker = None
        self._model_path: Optional[Path] = None
        self._closed = False

        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe is required for face landmark extraction") from exc

        solutions = getattr(mp, "solutions", None)
        if solutions is not None and hasattr(solutions, "face_mesh"):
            self._legacy = solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            return

        default_model = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
        configured_model = model_path or os.environ.get("FACE_LANDMARKER_MODEL")
        path = Path(configured_model) if configured_model else default_model
        if not path.is_file():
            self._model_path = None
            return
        self._model_path = path
        self._create_task_landmarker(
            min_detection_confidence, min_tracking_confidence
        )

    def _create_task_landmarker(
        self, min_detection_confidence: float, min_tracking_confidence: float
    ) -> None:
        assert self._model_path is not None
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    @staticmethod
    def _to_pixels(landmark_list: Any, width: int, height: int) -> np.ndarray:
        return np.asarray(
            [(lm.x * width, lm.y * height) for lm in landmark_list],
            dtype=np.float32,
        )

    def process(self, rgb_frame: np.ndarray) -> FaceMeshResult:
        """Process an RGB frame and return a normalized result object."""
        if self._closed:
            raise RuntimeError("FaceMeshDetector has already been closed")
        if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            raise ValueError("rgb_frame must have shape (height, width, 3)")

        height, width = rgb_frame.shape[:2]
        if self._legacy is not None:
            result = self._legacy.process(rgb_frame)
            faces = result.multi_face_landmarks
            landmarks = self._to_pixels(faces[0].landmark, width, height) if faces else None
            return FaceMeshResult(landmarks=landmarks, raw=result)

        if self._landmarker is None:
            raise RuntimeError(
                "FaceLandmarker model is not configured for this MediaPipe version. "
                "Set FACE_LANDMARKER_MODEL to a face_landmarker.task file."
            )
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))
        result = self._landmarker.detect(image)
        faces = result.face_landmarks
        landmarks = self._to_pixels(faces[0], width, height) if faces else None
        blendshapes = result.face_blendshapes[0] if result.face_blendshapes else None
        matrix = (
            result.facial_transformation_matrixes[0]
            if result.facial_transformation_matrixes
            else None
        )
        return FaceMeshResult(
            landmarks=landmarks,
            raw=result,
            blendshapes=blendshapes,
            transformation_matrix=matrix,
        )

    def get_landmarks(self, rgb_frame: np.ndarray) -> Optional[np.ndarray]:
        """Return pixel landmarks, or ``None`` when no face is detected."""
        return self.process(rgb_frame).landmarks

    def draw_landmarks(self, bgr_frame: np.ndarray, result: FaceMeshResult) -> np.ndarray:
        """Draw detected landmark points onto a BGR frame."""
        if not result.face_found:
            return bgr_frame
        annotated = bgr_frame.copy()
        for x, y in result.landmarks:
            cv2.circle(annotated, (round(float(x)), round(float(y))), 1, (0, 255, 0), -1)
        return annotated

    def close(self) -> None:
        """Release the underlying MediaPipe graph/task."""
        if self._closed:
            return
        if self._legacy is not None:
            self._legacy.close()
        if self._landmarker is not None:
            self._landmarker.close()
        self._closed = True

    def __enter__(self) -> "FaceMeshDetector":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()


FaceMeshExtractor = FaceMeshDetector
