"""Thin wrapper around MediaPipe's FaceLandmarker (Tasks API, VIDEO mode)."""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np

from faceheatmap.layout import BBox

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


@dataclass
class FaceObservation:
    bbox: BBox
    landmarks_px: np.ndarray  # (478, 3), x/y in pixel coords, z in the same relative scale
    transform_matrix: np.ndarray  # (4, 4)


def ensure_model(model_path: str, download_if_missing: bool = True, timeout_s: float = 60.0) -> str:
    if os.path.exists(model_path):
        return model_path
    if not download_if_missing:
        raise FileNotFoundError(
            f"FaceLandmarker model not found at {model_path!r}. "
            f"Run scripts/download_models.sh or pass download_if_missing=True."
        )
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    with urllib.request.urlopen(_MODEL_URL, timeout=timeout_s) as resp, open(model_path, "wb") as f:
        f.write(resp.read())
    return model_path


class FaceLandmarkerWrapper:
    """Runs MediaPipe FaceLandmarker over a video frame-by-frame.

    Must be driven with strictly increasing timestamps (VIDEO running mode
    requirement) -- use frame index * (1000 / fps) in milliseconds.
    """

    def __init__(
        self,
        model_path: str,
        num_faces: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        download_if_missing: bool = True,
    ):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        resolved_path = ensure_model(model_path, download_if_missing=download_if_missing)

        base_options = mp_python.BaseOptions(model_asset_path=resolved_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_facial_transformation_matrixes=True,
        )
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[FaceObservation]:
        import cv2

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        observations = []
        for face_landmarks, matrix in zip(result.face_landmarks, result.facial_transformation_matrixes):
            pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in face_landmarks], dtype=np.float64)
            x0, y0 = pts[:, :2].min(axis=0)
            x1, y1 = pts[:, :2].max(axis=0)
            observations.append(
                FaceObservation(
                    bbox=BBox(float(x0), float(y0), float(x1), float(y1)),
                    landmarks_px=pts,
                    transform_matrix=np.array(matrix, dtype=np.float64),
                )
            )
        return observations

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceLandmarkerWrapper":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
