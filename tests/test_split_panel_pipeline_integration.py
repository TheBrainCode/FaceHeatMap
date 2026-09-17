"""Regression test for the split-panel detection mode, modeling the exact
real-world failure it fixes: MediaPipe's FaceLandmarker returning only one
face when asked for two on a combined frame where the faces differ a lot in
scale, even though each face is trivially detectable on its own.
"""
import cv2
import numpy as np
import pytest

from faceheatmap.config import GazeConfig, LayoutMode, PersonId, PipelineConfig
from faceheatmap.landmark_indices import NUM_LANDMARKS, get_landmark_indices
from faceheatmap.landmarker import FaceObservation
from faceheatmap.layout import BBox
from faceheatmap.pipeline import FaceHeatmapPipeline

FRAME_W, FRAME_H = 800, 400


def _write_blank_video(path: str, num_frames: int = 10, size=(FRAME_W, FRAME_H)) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 10.0, size)
    for _ in range(num_frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()


def _face_landmarks(cx: float, cy: float, face_half: float) -> np.ndarray:
    indices = get_landmark_indices()
    landmarks = np.zeros((NUM_LANDMARKS, 3))
    oval = indices["face_oval"]
    for i, idx in enumerate(oval):
        angle = 2 * np.pi * i / len(oval)
        landmarks[idx, 0] = cx + face_half * np.cos(angle)
        landmarks[idx, 1] = cy + face_half * np.sin(angle)
    for eye_key in ("left_eye", "right_eye"):
        eye_cx, eye_cy = cx, cy - face_half * 0.2
        half_w, half_h = face_half * 0.15, face_half * 0.08
        for i, idx in enumerate(indices[eye_key]):
            angle = 2 * np.pi * i / len(indices[eye_key])
            landmarks[idx, 0] = eye_cx + half_w * np.cos(angle)
            landmarks[idx, 1] = eye_cy + half_h * np.sin(angle)
    from faceheatmap.landmark_indices import LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER

    landmarks[LEFT_IRIS_CENTER] = [cx, cy - face_half * 0.2, 0]
    landmarks[RIGHT_IRIS_CENTER] = [cx, cy - face_half * 0.2, 0]
    return landmarks


class OnlyLargestFaceWholeFrameLandmarker:
    """Models MediaPipe's real observed behavior on the user's footage:
    given the whole frame, it only ever returns the larger/closer face
    (person1's, positioned in the left panel), never both, regardless of
    num_faces requested.
    """

    def detect(self, frame_bgr, timestamp_ms):
        landmarks = _face_landmarks(cx=150, cy=200, face_half=80)  # large, left panel
        bbox = BBox(*landmarks[:, :2].min(axis=0), *landmarks[:, :2].max(axis=0))
        return [FaceObservation(bbox=bbox, landmarks_px=landmarks, transform_matrix=np.eye(4))]


class PanelLocalLandmarker:
    """Detects a single face in whatever crop it's given, assuming the face
    sits near the given local coordinates (i.e. this landmarker is only
    ever handed one person's panel crop, as SplitPanelDetectionLoop does).
    """

    def __init__(self, local_cx: float, local_cy: float, face_half: float):
        self.local_cx = local_cx
        self.local_cy = local_cy
        self.face_half = face_half
        self.calls = 0

    def detect(self, frame_bgr, timestamp_ms):
        self.calls += 1
        landmarks = _face_landmarks(self.local_cx, self.local_cy, self.face_half)
        bbox = BBox(*landmarks[:, :2].min(axis=0), *landmarks[:, :2].max(axis=0))
        return [FaceObservation(bbox=bbox, landmarks_px=landmarks, transform_matrix=np.eye(4))]


def test_whole_frame_mode_fails_on_the_asymmetric_scale_scenario(tmp_path):
    """Sanity check that this test actually models the real bug: the
    default (non-split) pipeline should fail exactly as it did in practice.
    """
    video_path = str(tmp_path / "video.mp4")
    _write_blank_video(video_path)

    config = PipelineConfig()  # split_panel_detection defaults to False
    pipeline = FaceHeatmapPipeline(video_path, str(tmp_path / "out"), config, landmarker=OnlyLargestFaceWholeFrameLandmarker())
    with pytest.raises(RuntimeError, match="Never detected two faces"):
        pipeline.run()


def test_split_panel_mode_succeeds_where_whole_frame_mode_fails(tmp_path):
    video_path = str(tmp_path / "video.mp4")
    _write_blank_video(video_path)

    config = PipelineConfig(
        layout=LayoutMode.SIDE_BY_SIDE,
        split_panel_detection=True,
        gaze=GazeConfig(smoothing_alpha=1.0),
    )
    # Person 1's face is large/close (left panel, local coords near panel center);
    # person 2's is small/far (right panel) -- exactly the scale mismatch that
    # broke whole-frame detection, but each is trivially found in its own crop.
    landmarker1 = PanelLocalLandmarker(local_cx=200, local_cy=200, face_half=80)
    landmarker2 = PanelLocalLandmarker(local_cx=200, local_cy=200, face_half=25)

    pipeline = FaceHeatmapPipeline(
        video_path,
        str(tmp_path / "out"),
        config,
        landmarker_person1=landmarker1,
        landmarker_person2=landmarker2,
    )
    result = pipeline.run()

    assert result.frames_processed == 10
    assert result.frames_with_both_faces == 10
    assert result.layout_used is LayoutMode.SIDE_BY_SIDE
    assert landmarker1.calls == 10
    assert landmarker2.calls == 10


def test_split_panel_mode_rejects_auto_layout(tmp_path):
    video_path = str(tmp_path / "video.mp4")
    _write_blank_video(video_path)

    config = PipelineConfig(layout=LayoutMode.AUTO, split_panel_detection=True)
    pipeline = FaceHeatmapPipeline(
        video_path,
        str(tmp_path / "out"),
        config,
        landmarker_person1=PanelLocalLandmarker(200, 200, 80),
        landmarker_person2=PanelLocalLandmarker(200, 200, 25),
    )
    with pytest.raises(ValueError, match="explicit --layout"):
        pipeline.run()


def test_split_panel_mode_internal_landmarker_uses_num_faces_one(tmp_path):
    """When neither external landmarker is given, the pipeline should build
    its own per-panel landmarkers configured for a single face each, not
    the whole-frame default of 2 -- verified via the injected factory args.
    """
    from faceheatmap import pipeline as pipeline_module

    video_path = str(tmp_path / "video.mp4")
    _write_blank_video(video_path, num_frames=1)

    captured_num_faces = []
    real_make_landmarker = pipeline_module._make_landmarker

    def spy_make_landmarker(config, num_faces=None):
        captured_num_faces.append(num_faces)
        return PanelLocalLandmarker(200, 200, 40)

    pipeline_module._make_landmarker = spy_make_landmarker
    try:
        config = PipelineConfig(layout=LayoutMode.SIDE_BY_SIDE, split_panel_detection=True)
        FaceHeatmapPipeline(video_path, str(tmp_path / "out"), config).run()
    finally:
        pipeline_module._make_landmarker = real_make_landmarker

    assert captured_num_faces == [1, 1]
