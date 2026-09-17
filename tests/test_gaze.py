import numpy as np
import pytest

from faceheatmap.config import GazeConfig
from faceheatmap.gaze import SmoothedGazeEstimator, estimate_gaze
from faceheatmap.landmark_indices import LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER, NUM_LANDMARKS, get_landmark_indices
from faceheatmap.rotation import euler_to_matrix

FRAME_W, FRAME_H = 1000, 800


def make_landmarks(eye_h_offset: float = 0.0, eye_v_offset: float = 0.0) -> np.ndarray:
    """Builds a synthetic 478-point landmark array with both eyes centered
    at (500, 400) spanning +/-20px, and irises offset by the given ratios.
    """
    indices = get_landmark_indices()
    landmarks = np.zeros((NUM_LANDMARKS, 3))

    for eye_key, iris_idx in [("left_eye", LEFT_IRIS_CENTER), ("right_eye", RIGHT_IRIS_CENTER)]:
        eye_cx, eye_cy, half_w, half_h = 500.0, 400.0, 20.0, 10.0
        pts = indices[eye_key]
        # Spread eye boundary points evenly around the box so min/max recovers the box.
        for i, idx in enumerate(pts):
            angle = 2 * np.pi * i / len(pts)
            landmarks[idx, 0] = eye_cx + half_w * np.cos(angle)
            landmarks[idx, 1] = eye_cy + half_h * np.sin(angle)
        landmarks[iris_idx, 0] = eye_cx + eye_h_offset * half_w
        landmarks[iris_idx, 1] = eye_cy + eye_v_offset * half_h

    return landmarks


def identity_transform() -> np.ndarray:
    return np.eye(4)


def test_centered_iris_and_frontal_head_looks_at_frame_center():
    config = GazeConfig(fov_h_deg=40, fov_v_deg=24)
    landmarks = make_landmarks(0.0, 0.0)
    result = estimate_gaze(landmarks, identity_transform(), FRAME_W, FRAME_H, config)
    assert result.yaw_deg == pytest.approx(0.0, abs=1e-6)
    assert result.pitch_deg == pytest.approx(0.0, abs=1e-6)
    assert result.frame_x == pytest.approx(FRAME_W / 2)
    assert result.frame_y == pytest.approx(FRAME_H / 2)
    assert result.valid


def test_iris_shifted_right_moves_gaze_point_right():
    config = GazeConfig(fov_h_deg=40, fov_v_deg=24, eye_gain_h_deg=22)
    centered = estimate_gaze(make_landmarks(0.0, 0.0), identity_transform(), FRAME_W, FRAME_H, config)
    shifted = estimate_gaze(make_landmarks(0.8, 0.0), identity_transform(), FRAME_W, FRAME_H, config)
    assert shifted.yaw_deg > centered.yaw_deg
    assert shifted.frame_x > centered.frame_x


def test_iris_shifted_down_moves_gaze_point_down():
    config = GazeConfig(fov_h_deg=40, fov_v_deg=24, eye_gain_v_deg=18)
    centered = estimate_gaze(make_landmarks(0.0, 0.0), identity_transform(), FRAME_W, FRAME_H, config)
    shifted = estimate_gaze(make_landmarks(0.0, 0.8), identity_transform(), FRAME_W, FRAME_H, config)
    assert shifted.pitch_deg > centered.pitch_deg
    assert shifted.frame_y > centered.frame_y


def test_head_yaw_shifts_gaze_consistently_with_eye_offset():
    config = GazeConfig(fov_h_deg=40, fov_v_deg=24)
    landmarks = make_landmarks(0.0, 0.0)
    turned = euler_to_matrix(yaw_deg=10, pitch_deg=0, roll_deg=0)
    m4 = np.eye(4)
    m4[:3, :3] = turned
    result = estimate_gaze(landmarks, m4, FRAME_W, FRAME_H, config)
    assert result.yaw_deg == pytest.approx(10.0, abs=1e-6)
    assert result.frame_x > FRAME_W / 2


def test_flip_yaw_inverts_direction():
    config = GazeConfig(fov_h_deg=40, fov_v_deg=24, flip_yaw=True)
    landmarks = make_landmarks(0.8, 0.0)
    result = estimate_gaze(landmarks, identity_transform(), FRAME_W, FRAME_H, config)
    assert result.yaw_deg < 0
    assert result.frame_x < FRAME_W / 2


def test_large_angle_is_flagged_invalid():
    config = GazeConfig(fov_h_deg=40, fov_v_deg=24, eye_gain_h_deg=22, max_valid_yaw_deg=15)
    landmarks = make_landmarks(1.0, 0.0)
    turned = euler_to_matrix(yaw_deg=60, pitch_deg=0, roll_deg=0)
    m4 = np.eye(4)
    m4[:3, :3] = turned
    result = estimate_gaze(landmarks, m4, FRAME_W, FRAME_H, config)
    assert not result.valid


def test_norm_coords_clipped_to_unit_range():
    config = GazeConfig(fov_h_deg=5, fov_v_deg=5, eye_gain_h_deg=22, eye_gain_v_deg=18, max_valid_yaw_deg=999, max_valid_pitch_deg=999)
    landmarks = make_landmarks(1.0, 1.0)
    result = estimate_gaze(landmarks, identity_transform(), FRAME_W, FRAME_H, config)
    assert result.norm_x == pytest.approx(1.0)
    assert result.norm_y == pytest.approx(1.0)
    assert result.frame_x == pytest.approx(FRAME_W)
    assert result.frame_y == pytest.approx(FRAME_H)


def test_smoothed_estimator_moves_gradually_toward_new_point():
    config = GazeConfig(fov_h_deg=40, fov_v_deg=24, smoothing_alpha=0.5)
    estimator = SmoothedGazeEstimator(config)

    first = estimator.update(make_landmarks(0.0, 0.0), identity_transform(), FRAME_W, FRAME_H)
    assert first.frame_x == pytest.approx(FRAME_W / 2)

    second = estimator.update(make_landmarks(0.8, 0.0), identity_transform(), FRAME_W, FRAME_H)
    raw = estimate_gaze(make_landmarks(0.8, 0.0), identity_transform(), FRAME_W, FRAME_H, config)
    assert FRAME_W / 2 < second.frame_x < raw.frame_x


def test_smoothing_alpha_zero_disables_smoothing():
    config = GazeConfig(fov_h_deg=40, fov_v_deg=24, smoothing_alpha=0.0)
    estimator = SmoothedGazeEstimator(config)
    estimator.update(make_landmarks(0.0, 0.0), identity_transform(), FRAME_W, FRAME_H)
    second = estimator.update(make_landmarks(0.8, 0.0), identity_transform(), FRAME_W, FRAME_H)
    raw = estimate_gaze(make_landmarks(0.8, 0.0), identity_transform(), FRAME_W, FRAME_H, config)
    assert second.frame_x == pytest.approx(raw.frame_x)
