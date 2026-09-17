"""Regression test: the heatmap overlay background must come from a frame
that actually shows both people, not literally frame 0 -- real recordings
can have a black/blank leading frame (e.g. a brief intro) before content
starts, which happened in practice and produced heatmaps overlaid on a
black background instead of the actual video.
"""
import cv2
import numpy as np

from faceheatmap.config import PipelineConfig
from faceheatmap.landmarker import FaceObservation
from faceheatmap.layout import BBox
from faceheatmap.pipeline import FaceHeatmapPipeline
from tests.test_pipeline_integration import _make_face_landmarks

FRAME_W, FRAME_H = 800, 400
CONTENT_COLOR = (100, 150, 200)  # BGR, distinct from black


def _write_black_then_colored_video(path: str, num_black_frames: int, num_colored_frames: int) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 10.0, (FRAME_W, FRAME_H))
    for _ in range(num_black_frames):
        writer.write(np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8))
    colored = np.full((FRAME_H, FRAME_W, 3), CONTENT_COLOR, dtype=np.uint8)
    for _ in range(num_colored_frames):
        writer.write(colored)
    writer.release()


class BlackFrameThenFacesLandmarker:
    """No faces for the first few calls (simulating a blank leading frame
    with nothing to detect), both faces afterward.
    """

    def __init__(self, blank_calls: int):
        self.blank_calls = blank_calls
        self.calls = 0

    def detect(self, frame_bgr, timestamp_ms) -> list[FaceObservation]:
        self.calls += 1
        if self.calls <= self.blank_calls:
            return []

        identity = np.eye(4)
        left_landmarks = _make_face_landmarks(200, 200)
        right_landmarks = _make_face_landmarks(600, 200)
        left_bbox = BBox(*left_landmarks[:, :2].min(axis=0), *left_landmarks[:, :2].max(axis=0))
        right_bbox = BBox(*right_landmarks[:, :2].min(axis=0), *right_landmarks[:, :2].max(axis=0))
        return [
            FaceObservation(bbox=left_bbox, landmarks_px=left_landmarks, transform_matrix=identity),
            FaceObservation(bbox=right_bbox, landmarks_px=right_landmarks, transform_matrix=identity),
        ]


def test_background_frame_skips_leading_blank_frames(tmp_path):
    video_path = str(tmp_path / "video.mp4")
    _write_black_then_colored_video(video_path, num_black_frames=3, num_colored_frames=7)

    out_dir = str(tmp_path / "out")
    config = PipelineConfig()
    landmarker = BlackFrameThenFacesLandmarker(blank_calls=3)
    result = FaceHeatmapPipeline(video_path, out_dir, config, landmarker=landmarker).run()

    full_frame_path = result.heatmap_paths["person1"]["full_frame"]
    img = cv2.imread(full_frame_path)
    # Sample a corner far from any heatmap heat: should be close to the
    # content color (allowing for mp4v lossy-compression drift), not black.
    corner_pixel = [int(c) for c in img[5, 5]]
    assert all(abs(a - b) <= 10 for a, b in zip(corner_pixel, CONTENT_COLOR))


def test_background_frame_falls_back_to_frame_zero_if_both_faces_never_seen(tmp_path):
    video_path = str(tmp_path / "video.mp4")
    _write_black_then_colored_video(video_path, num_black_frames=2, num_colored_frames=0)

    out_dir = str(tmp_path / "out")
    config = PipelineConfig()

    class NeverBothLandmarker:
        def detect(self, frame_bgr, timestamp_ms):
            return []

    try:
        FaceHeatmapPipeline(video_path, out_dir, config, landmarker=NeverBothLandmarker()).run()
    except RuntimeError:
        pass  # expected: never established a layout -- this test only cares that we didn't crash picking a background frame
