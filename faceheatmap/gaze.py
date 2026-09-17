"""Calibration-free gaze-direction and screen-point estimation.

There is no way to recover a true gaze-to-screen calibration from a
recording alone (we don't know the viewer's screen size, distance, or
camera offset), so this module implements a documented heuristic instead:

1. Head pose (yaw/pitch) comes from MediaPipe's facial transformation
   matrix, decomposed via `faceheatmap.rotation`.
2. A within-eye iris-offset correction is added on top, so gaze doesn't
   collapse to "wherever the head points" when someone holds their head
   still and just moves their eyes.
3. The resulting gaze angle is projected onto an assumed field of view
   (`GazeConfig.fov_h_deg` / `fov_v_deg`) to get a normalized point in
   [-1, 1] x [-1, 1], which is then mapped onto the recording frame --
   i.e. we assume the recorded frame is a reasonable proxy for what's on
   the person's own screen.

This is a best-effort approximation suitable for aggregate attention
patterns (e.g. "did they spend more time looking at the other person's
face"), not a precise gaze-tracking replacement.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from faceheatmap.config import GazeConfig
from faceheatmap.landmark_indices import (
    LEFT_IRIS_CENTER,
    RIGHT_IRIS_CENTER,
    get_landmark_indices,
)
from faceheatmap.rotation import matrix_to_euler


@dataclass
class GazeResult:
    yaw_deg: float
    pitch_deg: float
    norm_x: float  # normalized gaze point, -1 (left/up) .. 1 (right/down)
    norm_y: float
    frame_x: float  # gaze point mapped into full-frame pixel coordinates
    frame_y: float
    valid: bool  # False when yaw/pitch exceed the configured max (near-profile face)


def _eye_offset_ratio(landmarks_px: np.ndarray, eye_indices: list[int], iris_center_idx: int) -> tuple[float, float]:
    eye_pts = landmarks_px[eye_indices, :2]
    x0, y0 = eye_pts.min(axis=0)
    x1, y1 = eye_pts.max(axis=0)
    eye_cx, eye_cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w, half_h = max((x1 - x0) / 2, 1e-6), max((y1 - y0) / 2, 1e-6)

    iris = landmarks_px[iris_center_idx, :2]
    h_ratio = np.clip((iris[0] - eye_cx) / half_w, -1.0, 1.0)
    v_ratio = np.clip((iris[1] - eye_cy) / half_h, -1.0, 1.0)
    return float(h_ratio), float(v_ratio)


def estimate_gaze(
    landmarks_px: np.ndarray,
    transform_matrix: np.ndarray,
    frame_w: int,
    frame_h: int,
    config: GazeConfig,
) -> GazeResult:
    """Estimates where on the recording frame a face is looking.

    `landmarks_px` must be the 478 FaceLandmarker landmarks in pixel
    coordinates (x, y[, z]) for the full frame. `transform_matrix` is the
    4x4 facial transformation matrix for the same face.
    """
    indices = get_landmark_indices()
    angles = matrix_to_euler(transform_matrix)

    left_h, left_v = _eye_offset_ratio(landmarks_px, indices["left_eye"], LEFT_IRIS_CENTER)
    right_h, right_v = _eye_offset_ratio(landmarks_px, indices["right_eye"], RIGHT_IRIS_CENTER)
    eye_h = (left_h + right_h) / 2
    eye_v = (left_v + right_v) / 2

    yaw = angles.yaw_deg + config.eye_gain_h_deg * eye_h
    pitch = angles.pitch_deg + config.eye_gain_v_deg * eye_v

    if config.flip_yaw:
        yaw = -yaw
    if config.flip_pitch:
        pitch = -pitch

    valid = abs(yaw) <= config.max_valid_yaw_deg and abs(pitch) <= config.max_valid_pitch_deg

    norm_x = float(np.clip(yaw / (config.fov_h_deg / 2), -1.0, 1.0))
    norm_y = float(np.clip(pitch / (config.fov_v_deg / 2), -1.0, 1.0))

    frame_x = (norm_x + 1.0) / 2.0 * frame_w
    frame_y = (norm_y + 1.0) / 2.0 * frame_h

    return GazeResult(
        yaw_deg=yaw,
        pitch_deg=pitch,
        norm_x=norm_x,
        norm_y=norm_y,
        frame_x=frame_x,
        frame_y=frame_y,
        valid=valid,
    )


class SmoothedGazeEstimator:
    """Stateful wrapper applying exponential smoothing to the mapped gaze point.

    Reduces frame-to-frame jitter from landmark noise before it gets baked
    into the heatmap.
    """

    def __init__(self, config: GazeConfig):
        self._config = config
        self._prev_point: tuple[float, float] | None = None

    def update(self, landmarks_px: np.ndarray, transform_matrix: np.ndarray, frame_w: int, frame_h: int) -> GazeResult:
        result = estimate_gaze(landmarks_px, transform_matrix, frame_w, frame_h, self._config)

        alpha = self._config.smoothing_alpha
        if self._prev_point is None or alpha <= 0:
            smoothed = (result.frame_x, result.frame_y)
        else:
            px, py = self._prev_point
            smoothed = (
                alpha * result.frame_x + (1 - alpha) * px,
                alpha * result.frame_y + (1 - alpha) * py,
            )
        self._prev_point = smoothed
        result.frame_x, result.frame_y = smoothed
        return result

    def reset(self) -> None:
        self._prev_point = None
