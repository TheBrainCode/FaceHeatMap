"""Verifies a fitted calibration actually changes the pipeline's gaze mapping,
compared to the default FOV heuristic, using a fake landmarker end-to-end.
"""
import csv

import cv2
import numpy as np

from faceheatmap.calibration import PersonCalibration
from faceheatmap.config import GazeConfig, PersonId, PipelineConfig
from faceheatmap.gaze import RawGazeSignal
from faceheatmap.pipeline import FaceHeatmapPipeline
from tests.test_pipeline_integration import FakeLandmarker

FRAME_W, FRAME_H = 800, 400


def _write_blank_video(path: str, num_frames: int = 10, size=(FRAME_W, FRAME_H)) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 10.0, size)
    for _ in range(num_frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()


def test_calibration_changes_mapped_gaze_point(tmp_path):
    video_path = str(tmp_path / "video.mp4")
    _write_blank_video(video_path)

    config = PipelineConfig(gaze=GazeConfig(smoothing_alpha=1.0))

    default_out = str(tmp_path / "default_out")
    default_result = FaceHeatmapPipeline(video_path, default_out, config, landmarker=FakeLandmarker()).run()

    # A calibration that maps everything to the far-left, far-top corner
    # regardless of raw signal (bias-only, zero weights on the signal terms).
    calibration = {
        PersonId.PERSON_1: PersonCalibration(h_weights=(0.0, 0.0, -1.0), v_weights=(0.0, 0.0, -1.0)),
        PersonId.PERSON_2: PersonCalibration(h_weights=(0.0, 0.0, -1.0), v_weights=(0.0, 0.0, -1.0)),
    }
    calibrated_out = str(tmp_path / "calibrated_out")
    calibrated_result = FaceHeatmapPipeline(
        video_path, calibrated_out, config, landmarker=FakeLandmarker(), calibration=calibration
    ).run()

    # Sanity: both runs succeeded and processed the same number of frames.
    assert default_result.frames_processed == calibrated_result.frames_processed == 10

    def read_person1_frame_xy(csv_path):
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        return [(float(r["person1_frame_x"]), float(r["person1_frame_y"])) for r in rows]

    default_points = read_person1_frame_xy(default_result.csv_path)
    calibrated_points = read_person1_frame_xy(calibrated_result.csv_path)

    # The calibration forces every point to the far top-left corner
    # (norm_x=norm_y=-1 -> frame_x=frame_y=0), which the default FOV
    # heuristic would not produce given the fake landmarker's raw signal.
    assert calibrated_points != default_points
    assert all(x == 0.0 and y == 0.0 for x, y in calibrated_points)
    assert not all(x == 0.0 and y == 0.0 for x, y in default_points)


def test_calibration_apply_matches_pipeline_frame_mapping():
    # Direct check that PersonCalibration.apply's contract (norm_x/norm_y in
    # [-1, 1], mapped linearly to frame_x/frame_y) matches what estimate_gaze
    # actually does when given a calibration override.
    from faceheatmap.gaze import estimate_gaze

    cal = PersonCalibration(h_weights=(0.0, 0.0, 0.5), v_weights=(0.0, 0.0, -0.5))
    landmarks = np.zeros((478, 3))
    landmarks[:, :2] = 100.0  # degenerate but estimate_gaze only needs eye/iris indices populated meaningfully for eye_h/eye_v; bias-only calibration ignores raw signal anyway
    from faceheatmap.landmark_indices import get_landmark_indices, LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER

    indices = get_landmark_indices()
    for key, iris_idx in [("left_eye", LEFT_IRIS_CENTER), ("right_eye", RIGHT_IRIS_CENTER)]:
        for idx in indices[key]:
            landmarks[idx, :2] = [100.0, 100.0]
        landmarks[iris_idx, :2] = [100.0, 100.0]

    result = estimate_gaze(landmarks, np.eye(4), 1000, 500, GazeConfig(), calibration=cal)
    assert result.norm_x == 0.5
    assert result.norm_y == -0.5
    assert result.frame_x == 0.75 * 1000
    assert result.frame_y == 0.25 * 500
