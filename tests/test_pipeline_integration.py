"""Integration tests for the pipeline, using a fake landmarker so they don't
depend on the real MediaPipe model or actual face content in the video.
"""
import json

import cv2
import numpy as np
import pytest

from faceheatmap.config import GazeConfig, LayoutMode, PipelineConfig
from faceheatmap.landmark_indices import (
    LEFT_IRIS_CENTER,
    NUM_LANDMARKS,
    RIGHT_IRIS_CENTER,
    get_landmark_indices,
)
from faceheatmap.landmarker import FaceObservation
from faceheatmap.layout import BBox
from faceheatmap.pipeline import FaceHeatmapPipeline

FRAME_W, FRAME_H = 800, 400


def _make_face_landmarks(center_x: float, center_y: float, face_half: float = 60.0, iris_h_offset: float = 0.0) -> np.ndarray:
    """Builds synthetic landmarks: a face_oval circle plus centered eyes,
    with the iris optionally offset horizontally within the eye box.
    """
    indices = get_landmark_indices()
    landmarks = np.zeros((NUM_LANDMARKS, 3))

    oval = indices["face_oval"]
    for i, idx in enumerate(oval):
        angle = 2 * np.pi * i / len(oval)
        landmarks[idx, 0] = center_x + face_half * np.cos(angle)
        landmarks[idx, 1] = center_y + face_half * np.sin(angle)

    for eye_key, iris_idx, x_off in [
        ("left_eye", LEFT_IRIS_CENTER, 15.0),
        ("right_eye", RIGHT_IRIS_CENTER, -15.0),
    ]:
        eye_cx, eye_cy = center_x + x_off, center_y - 10.0
        half_w, half_h = 8.0, 4.0
        for i, idx in enumerate(indices[eye_key]):
            angle = 2 * np.pi * i / len(indices[eye_key])
            landmarks[idx, 0] = eye_cx + half_w * np.cos(angle)
            landmarks[idx, 1] = eye_cy + half_h * np.sin(angle)
        landmarks[iris_idx, 0] = eye_cx + iris_h_offset * half_w
        landmarks[iris_idx, 1] = eye_cy

    return landmarks


class FakeLandmarker:
    """Two faces, fixed positions (left panel / right panel). Person 1's iris
    is offset toward the panel center line every frame, so its gaze should
    consistently land on person 2's side -- deterministic enough to assert on.
    """

    def __init__(self):
        self.calls = 0

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[FaceObservation]:
        self.calls += 1
        identity = np.eye(4)

        left_landmarks = _make_face_landmarks(200, 200, iris_h_offset=1.0)  # looking toward image-right (person 2)
        right_landmarks = _make_face_landmarks(600, 200, iris_h_offset=0.0)  # looking straight ahead

        left_bbox = BBox(*left_landmarks[:, :2].min(axis=0), *left_landmarks[:, :2].max(axis=0))
        right_bbox = BBox(*right_landmarks[:, :2].min(axis=0), *right_landmarks[:, :2].max(axis=0))

        return [
            FaceObservation(bbox=left_bbox, landmarks_px=left_landmarks, transform_matrix=identity),
            FaceObservation(bbox=right_bbox, landmarks_px=right_landmarks, transform_matrix=identity),
        ]


def _write_blank_video(path: str, num_frames: int = 20, size=(FRAME_W, FRAME_H)) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 10.0, size)
    for _ in range(num_frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()


def test_pipeline_end_to_end_with_fake_landmarker(tmp_path):
    video_path = str(tmp_path / "video.mp4")
    _write_blank_video(video_path, num_frames=20)

    out_dir = str(tmp_path / "out")
    config = PipelineConfig(
        gaze=GazeConfig(fov_h_deg=60, fov_v_deg=60, eye_gain_h_deg=30, eye_gain_v_deg=20, smoothing_alpha=1.0),
    )
    landmarker = FakeLandmarker()
    pipeline = FaceHeatmapPipeline(video_path, out_dir, config, landmarker=landmarker)
    result = pipeline.run()

    assert result.frames_processed == 20
    assert result.frames_with_both_faces == 20
    assert result.layout_used is LayoutMode.SIDE_BY_SIDE
    assert landmarker.calls == 20

    # Person 1's gaze was steered toward person 2's side every frame.
    p1_stats = result.boundary_stats["person1"]
    assert p1_stats["total_frames"] == 20

    for person, paths in result.heatmap_paths.items():
        for path in paths.values():
            img = cv2.imread(path)
            assert img is not None
            assert img.size > 0

    with open(result.summary_json_path) as f:
        summary = json.load(f)
    assert summary["frames_processed"] == 20
    assert "person1" in summary["boundary_stats"]

    assert result.csv_path is not None
    with open(result.csv_path) as f:
        lines = f.readlines()
    assert len(lines) == 21  # header + 20 frames


def test_pipeline_respects_frame_stride_and_max_frames(tmp_path):
    video_path = str(tmp_path / "video.mp4")
    _write_blank_video(video_path, num_frames=30)

    out_dir = str(tmp_path / "out")
    config = PipelineConfig(frame_stride=2, max_frames=5)
    landmarker = FakeLandmarker()
    pipeline = FaceHeatmapPipeline(video_path, out_dir, config, landmarker=landmarker)
    result = pipeline.run()

    assert result.frames_processed == 5
    assert landmarker.calls == 5


def test_pipeline_no_csv_when_disabled(tmp_path):
    video_path = str(tmp_path / "video.mp4")
    _write_blank_video(video_path, num_frames=5)

    out_dir = str(tmp_path / "out")
    config = PipelineConfig(write_csv_log=False)
    pipeline = FaceHeatmapPipeline(video_path, out_dir, config, landmarker=FakeLandmarker())
    result = pipeline.run()
    assert result.csv_path is None


class NeverTwoFacesLandmarker:
    def detect(self, frame_bgr, timestamp_ms):
        return []


def test_pipeline_raises_when_two_faces_never_detected(tmp_path):
    video_path = str(tmp_path / "video.mp4")
    _write_blank_video(video_path, num_frames=5)

    out_dir = str(tmp_path / "out")
    pipeline = FaceHeatmapPipeline(video_path, out_dir, PipelineConfig(), landmarker=NeverTwoFacesLandmarker())
    with pytest.raises(RuntimeError, match="Never detected two faces"):
        pipeline.run()


def test_pipeline_missing_video_raises(tmp_path):
    out_dir = str(tmp_path / "out")
    pipeline = FaceHeatmapPipeline(str(tmp_path / "does_not_exist.mp4"), out_dir, PipelineConfig(), landmarker=FakeLandmarker())
    with pytest.raises(FileNotFoundError):
        pipeline.run()
