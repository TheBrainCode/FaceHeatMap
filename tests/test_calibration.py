import json
import os

import cv2
import numpy as np
import pytest

from faceheatmap.calibration import (
    CALIBRATION_POINTS,
    CalibrationConfig,
    CalibrationSample,
    PersonCalibration,
    collect_calibration_samples,
    fit_axis,
    fit_person_calibration,
    load_calibration,
    run_calibration,
    save_calibration,
)
from faceheatmap.config import LayoutMode, PersonId
from faceheatmap.gaze import RawGazeSignal
from faceheatmap.landmarker import FaceObservation
from faceheatmap.layout import BBox


def test_fit_axis_recovers_exact_linear_relationship():
    # target = 2*a - 0.5*b + 3, exactly, no noise
    a = np.array([0.0, 1.0, 2.0, -1.0, 3.0])
    b = np.array([1.0, 0.0, -2.0, 4.0, 1.0])
    target = 2 * a - 0.5 * b + 3
    w0, w1, w2 = fit_axis(a, b, target)
    assert w0 == pytest.approx(2.0, abs=1e-8)
    assert w1 == pytest.approx(-0.5, abs=1e-8)
    assert w2 == pytest.approx(3.0, abs=1e-8)


def _samples_for_known_mapping(h_weights, v_weights) -> list[CalibrationSample]:
    """Builds one calibration sample per CALIBRATION_POINTS entry, with raw
    signal chosen so the exact given mapping is recoverable by least squares.
    Uses head_yaw/head_pitch as the only signal (eye_h/eye_v fixed at 0), so
    the inverse is trivial: head_yaw = (target_x - w2) / w0.
    """
    wh0, wh1, wh2 = h_weights
    wv0, wv1, wv2 = v_weights
    samples = []
    for name, tx, ty in CALIBRATION_POINTS:
        head_yaw = (tx - wh2) / wh0
        head_pitch = (ty - wv2) / wv0
        raw = RawGazeSignal(head_yaw_deg=head_yaw, head_pitch_deg=head_pitch, eye_h=0.0, eye_v=0.0)
        samples.append(CalibrationSample(name, tx, ty, raw))
    return samples


def test_fit_person_calibration_recovers_known_mapping():
    h_weights = (0.05, 0.3, 0.1)
    v_weights = (0.08, 0.2, -0.05)
    samples = _samples_for_known_mapping(h_weights, v_weights)

    cal = fit_person_calibration(samples)
    assert cal.h_weights[0] == pytest.approx(h_weights[0], abs=1e-6)
    assert cal.h_weights[2] == pytest.approx(h_weights[2], abs=1e-6)
    assert cal.v_weights[0] == pytest.approx(v_weights[0], abs=1e-6)
    assert cal.v_weights[2] == pytest.approx(v_weights[2], abs=1e-6)


def test_fit_person_calibration_requires_at_least_three_samples():
    samples = _samples_for_known_mapping((0.05, 0.3, 0.1), (0.08, 0.2, -0.05))[:2]
    with pytest.raises(ValueError, match="at least 3"):
        fit_person_calibration(samples)


def test_person_calibration_apply():
    cal = PersonCalibration(h_weights=(2.0, 1.0, 0.5), v_weights=(1.0, -1.0, 0.0))
    raw = RawGazeSignal(head_yaw_deg=1.0, head_pitch_deg=2.0, eye_h=0.5, eye_v=0.25)
    norm_x, norm_y = cal.apply(raw)
    assert norm_x == pytest.approx(2.0 * 1.0 + 1.0 * 0.5 + 0.5)
    assert norm_y == pytest.approx(1.0 * 2.0 - 1.0 * 0.25 + 0.0)


def test_person_calibration_round_trip_dict():
    cal = PersonCalibration(h_weights=(0.1, 0.2, 0.3), v_weights=(0.4, 0.5, 0.6))
    restored = PersonCalibration.from_dict(cal.to_dict())
    assert restored.h_weights == cal.h_weights
    assert restored.v_weights == cal.v_weights


def test_save_and_load_calibration_round_trip(tmp_path):
    calibration = {
        PersonId.PERSON_1: PersonCalibration(h_weights=(0.1, 0.2, 0.3), v_weights=(0.4, 0.5, 0.6)),
        PersonId.PERSON_2: PersonCalibration(h_weights=(0.7, 0.8, 0.9), v_weights=(1.0, 1.1, 1.2)),
    }
    path = str(tmp_path / "calibration.json")
    save_calibration(calibration, path)

    with open(path) as f:
        raw_json = json.load(f)
    assert set(raw_json.keys()) == {"person1", "person2"}

    loaded = load_calibration(path)
    assert loaded[PersonId.PERSON_1].h_weights == (0.1, 0.2, 0.3)
    assert loaded[PersonId.PERSON_2].v_weights == (1.0, 1.1, 1.2)


# --- End-to-end sample collection against a scripted (fake) landmarker ---

FRAME_W, FRAME_H = 800, 400


def _write_blank_video(path: str, num_frames: int, fps: float = 10.0, size=(FRAME_W, FRAME_H)) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    for _ in range(num_frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()


class ScriptedCalibrationLandmarker:
    """Both people always at fixed panel positions; person1's iris offset
    encodes which calibration point index is 'currently' being displayed,
    driven directly off the call count / known fps so segment boundaries
    line up exactly with CalibrationConfig's timing.
    """

    def __init__(self, fps: float, seconds_per_point: float):
        self.fps = fps
        self.seconds_per_point = seconds_per_point
        self.calls = 0

    def _landmarks_for(self, cx: float, cy: float, h_offset: float, v_offset: float) -> np.ndarray:
        from faceheatmap.landmark_indices import LEFT_IRIS_CENTER, NUM_LANDMARKS, RIGHT_IRIS_CENTER, get_landmark_indices

        indices = get_landmark_indices()
        landmarks = np.zeros((NUM_LANDMARKS, 3))
        oval = indices["face_oval"]
        for i, idx in enumerate(oval):
            angle = 2 * np.pi * i / len(oval)
            landmarks[idx, 0] = cx + 60 * np.cos(angle)
            landmarks[idx, 1] = cy + 60 * np.sin(angle)
        for eye_key, iris_idx, x_off in [("left_eye", LEFT_IRIS_CENTER, 15.0), ("right_eye", RIGHT_IRIS_CENTER, -15.0)]:
            eye_cx, eye_cy = cx + x_off, cy - 10.0
            half_w, half_h = 8.0, 4.0
            for i, idx in enumerate(indices[eye_key]):
                angle = 2 * np.pi * i / len(indices[eye_key])
                landmarks[idx, 0] = eye_cx + half_w * np.cos(angle)
                landmarks[idx, 1] = eye_cy + half_h * np.sin(angle)
            landmarks[iris_idx, 0] = eye_cx + h_offset * half_w
            landmarks[iris_idx, 1] = eye_cy + v_offset * half_h
        return landmarks

    def detect(self, frame_bgr, timestamp_ms):
        t_s = timestamp_ms / 1000.0
        point_index = min(int(t_s // self.seconds_per_point), len(CALIBRATION_POINTS) - 1)
        _, target_x, target_y = CALIBRATION_POINTS[point_index]

        left_landmarks = self._landmarks_for(200, 200, target_x, target_y)
        right_landmarks = self._landmarks_for(600, 200, 0.0, 0.0)

        left_bbox = BBox(*left_landmarks[:, :2].min(axis=0), *left_landmarks[:, :2].max(axis=0))
        right_bbox = BBox(*right_landmarks[:, :2].min(axis=0), *right_landmarks[:, :2].max(axis=0))

        self.calls += 1
        return [
            FaceObservation(bbox=left_bbox, landmarks_px=left_landmarks, transform_matrix=np.eye(4)),
            FaceObservation(bbox=right_bbox, landmarks_px=right_landmarks, transform_matrix=np.eye(4)),
        ]


def test_collect_calibration_samples_buckets_by_time(tmp_path):
    fps = 10.0
    seconds_per_point = 1.0  # short, for a fast test
    num_frames = int(len(CALIBRATION_POINTS) * seconds_per_point * fps)
    video_path = str(tmp_path / "calib.mp4")
    _write_blank_video(video_path, num_frames, fps=fps)

    landmarker = ScriptedCalibrationLandmarker(fps, seconds_per_point)
    config = CalibrationConfig(seconds_per_point=seconds_per_point, trim_start_frac=0.2, trim_end_frac=0.1)
    samples = collect_calibration_samples(video_path, landmarker, config, layout=LayoutMode.SIDE_BY_SIDE)

    assert len(samples[PersonId.PERSON_1]) == len(CALIBRATION_POINTS)
    assert len(samples[PersonId.PERSON_2]) == len(CALIBRATION_POINTS)

    by_name = {s.point_name: s for s in samples[PersonId.PERSON_1]}
    assert by_name["top-left"].raw.eye_h == pytest.approx(-1.0, abs=0.05)
    assert by_name["top-left"].raw.eye_v == pytest.approx(-1.0, abs=0.05)
    assert by_name["right"].raw.eye_h == pytest.approx(1.0, abs=0.05)
    assert by_name["right"].raw.eye_v == pytest.approx(0.0, abs=0.05)

    # Person 2 always looked "straight ahead" (0, 0) regardless of point.
    for s in samples[PersonId.PERSON_2]:
        assert s.raw.eye_h == pytest.approx(0.0, abs=0.05)
        assert s.raw.eye_v == pytest.approx(0.0, abs=0.05)


def test_run_calibration_end_to_end_writes_usable_calibration(tmp_path):
    fps = 10.0
    seconds_per_point = 1.0
    num_frames = int(len(CALIBRATION_POINTS) * seconds_per_point * fps)
    video_path = str(tmp_path / "calib.mp4")
    _write_blank_video(video_path, num_frames, fps=fps)

    landmarker = ScriptedCalibrationLandmarker(fps, seconds_per_point)
    config = CalibrationConfig(seconds_per_point=seconds_per_point, trim_start_frac=0.2, trim_end_frac=0.1)
    calibration = run_calibration(video_path, landmarker, config, layout=LayoutMode.SIDE_BY_SIDE)

    assert set(calibration.keys()) == {PersonId.PERSON_1, PersonId.PERSON_2}

    out_path = str(tmp_path / "calibration.json")
    save_calibration(calibration, out_path)
    assert os.path.exists(out_path)

    # Person 1's fitted mapping should map "looking right" (eye_h=1) toward +x.
    raw_right = RawGazeSignal(head_yaw_deg=0.0, head_pitch_deg=0.0, eye_h=1.0, eye_v=0.0)
    norm_x, _ = calibration[PersonId.PERSON_1].apply(raw_right)
    assert norm_x > 0


class NeverTwoFacesLandmarker:
    def detect(self, frame_bgr, timestamp_ms):
        return []


def test_collect_calibration_samples_raises_clear_error_when_layout_never_established(tmp_path):
    video_path = str(tmp_path / "calib.mp4")
    _write_blank_video(video_path, num_frames=10)

    with pytest.raises(RuntimeError, match="Never detected both faces together"):
        collect_calibration_samples(video_path, NeverTwoFacesLandmarker())


def test_run_calibration_reports_missing_points_by_name(tmp_path):
    # Video only covers the first 2 of 9 points (1s each at 10fps = 20 frames).
    fps = 10.0
    seconds_per_point = 1.0
    video_path = str(tmp_path / "calib.mp4")
    _write_blank_video(video_path, num_frames=20, fps=fps)

    landmarker = ScriptedCalibrationLandmarker(fps, seconds_per_point)
    config = CalibrationConfig(seconds_per_point=seconds_per_point, trim_start_frac=0.2, trim_end_frac=0.1)

    with pytest.raises(RuntimeError) as excinfo:
        run_calibration(video_path, landmarker, config, layout=LayoutMode.SIDE_BY_SIDE)

    message = str(excinfo.value)
    assert "top-right" in message  # one of the never-reached points
    assert "center" not in message.split("Missing:")[1]  # center *was* captured, shouldn't be listed as missing


def test_collect_calibration_samples_writes_debug_video_even_on_failure(tmp_path):
    video_path = str(tmp_path / "calib.mp4")
    _write_blank_video(video_path, num_frames=10)
    debug_path = str(tmp_path / "debug.mp4")

    with pytest.raises(RuntimeError):
        collect_calibration_samples(video_path, NeverTwoFacesLandmarker(), debug_video_path=debug_path)

    assert os.path.exists(debug_path)
    assert os.path.getsize(debug_path) > 0


def test_collect_calibration_samples_debug_video_on_success(tmp_path):
    fps = 10.0
    seconds_per_point = 1.0
    num_frames = int(len(CALIBRATION_POINTS) * seconds_per_point * fps)
    video_path = str(tmp_path / "calib.mp4")
    _write_blank_video(video_path, num_frames, fps=fps)
    debug_path = str(tmp_path / "debug.mp4")

    landmarker = ScriptedCalibrationLandmarker(fps, seconds_per_point)
    config = CalibrationConfig(seconds_per_point=seconds_per_point, trim_start_frac=0.2, trim_end_frac=0.1)
    collect_calibration_samples(video_path, landmarker, config, layout=LayoutMode.SIDE_BY_SIDE, debug_video_path=debug_path)

    assert os.path.exists(debug_path)


class ScriptedPanelLandmarker:
    """Split-panel counterpart to ScriptedCalibrationLandmarker: only ever
    handed one person's own panel crop, reports a single face at fixed
    local coordinates with an iris offset driven by which calibration
    point is active at that timestamp (or a fixed straight-ahead offset).
    """

    def __init__(self, seconds_per_point: float, moving: bool):
        self.seconds_per_point = seconds_per_point
        self.moving = moving
        self.calls = 0

    def detect(self, frame_bgr, timestamp_ms):
        from faceheatmap.landmark_indices import LEFT_IRIS_CENTER, NUM_LANDMARKS, RIGHT_IRIS_CENTER, get_landmark_indices

        self.calls += 1
        if self.moving:
            t_s = timestamp_ms / 1000.0
            point_index = min(int(t_s // self.seconds_per_point), len(CALIBRATION_POINTS) - 1)
            _, h_offset, v_offset = CALIBRATION_POINTS[point_index]
        else:
            h_offset, v_offset = 0.0, 0.0

        indices = get_landmark_indices()
        landmarks = np.zeros((NUM_LANDMARKS, 3))
        cx, cy = 100, 100
        oval = indices["face_oval"]
        for i, idx in enumerate(oval):
            angle = 2 * np.pi * i / len(oval)
            landmarks[idx, 0] = cx + 60 * np.cos(angle)
            landmarks[idx, 1] = cy + 60 * np.sin(angle)
        for eye_key, iris_idx, x_off in [("left_eye", LEFT_IRIS_CENTER, 15.0), ("right_eye", RIGHT_IRIS_CENTER, -15.0)]:
            eye_cx, eye_cy = cx + x_off, cy - 10.0
            half_w, half_h = 8.0, 4.0
            for i, idx in enumerate(indices[eye_key]):
                angle = 2 * np.pi * i / len(indices[eye_key])
                landmarks[idx, 0] = eye_cx + half_w * np.cos(angle)
                landmarks[idx, 1] = eye_cy + half_h * np.sin(angle)
            landmarks[iris_idx, 0] = eye_cx + h_offset * half_w
            landmarks[iris_idx, 1] = eye_cy + v_offset * half_h

        bbox = BBox(*landmarks[:, :2].min(axis=0), *landmarks[:, :2].max(axis=0))
        return [FaceObservation(bbox=bbox, landmarks_px=landmarks, transform_matrix=np.eye(4))]


def test_collect_calibration_samples_split_panel_mode(tmp_path):
    fps = 10.0
    seconds_per_point = 1.0
    num_frames = int(len(CALIBRATION_POINTS) * seconds_per_point * fps)
    video_path = str(tmp_path / "calib.mp4")
    _write_blank_video(video_path, num_frames, fps=fps)

    landmarker1 = ScriptedPanelLandmarker(seconds_per_point, moving=True)
    landmarker2 = ScriptedPanelLandmarker(seconds_per_point, moving=False)
    config = CalibrationConfig(seconds_per_point=seconds_per_point, trim_start_frac=0.2, trim_end_frac=0.1)

    samples = collect_calibration_samples(
        video_path, calib_config=config, layout=LayoutMode.SIDE_BY_SIDE,
        landmarker_person1=landmarker1, landmarker_person2=landmarker2,
    )

    assert len(samples[PersonId.PERSON_1]) == len(CALIBRATION_POINTS)
    assert len(samples[PersonId.PERSON_2]) == len(CALIBRATION_POINTS)
    assert landmarker1.calls == num_frames
    assert landmarker2.calls == num_frames

    by_name = {s.point_name: s for s in samples[PersonId.PERSON_1]}
    assert by_name["top-left"].raw.eye_h == pytest.approx(-1.0, abs=0.05)
    assert by_name["right"].raw.eye_h == pytest.approx(1.0, abs=0.05)


def test_run_calibration_split_panel_mode_fits_usable_calibration(tmp_path):
    fps = 10.0
    seconds_per_point = 1.0
    num_frames = int(len(CALIBRATION_POINTS) * seconds_per_point * fps)
    video_path = str(tmp_path / "calib.mp4")
    _write_blank_video(video_path, num_frames, fps=fps)

    landmarker1 = ScriptedPanelLandmarker(seconds_per_point, moving=True)
    landmarker2 = ScriptedPanelLandmarker(seconds_per_point, moving=False)
    config = CalibrationConfig(seconds_per_point=seconds_per_point, trim_start_frac=0.2, trim_end_frac=0.1)

    calibration = run_calibration(
        video_path, calib_config=config, layout=LayoutMode.SIDE_BY_SIDE,
        landmarker_person1=landmarker1, landmarker_person2=landmarker2,
    )
    assert set(calibration.keys()) == {PersonId.PERSON_1, PersonId.PERSON_2}

    raw_right = RawGazeSignal(head_yaw_deg=0.0, head_pitch_deg=0.0, eye_h=1.0, eye_v=0.0)
    norm_x, _ = calibration[PersonId.PERSON_1].apply(raw_right)
    assert norm_x > 0


def test_collect_calibration_samples_requires_landmarker_or_both_panel_landmarkers(tmp_path):
    video_path = str(tmp_path / "calib.mp4")
    _write_blank_video(video_path, num_frames=5)

    with pytest.raises(ValueError, match="Pass either landmarker"):
        collect_calibration_samples(video_path)

    with pytest.raises(ValueError, match="must be given together"):
        collect_calibration_samples(video_path, landmarker_person1=ScriptedPanelLandmarker(1.0, True))
