import cv2
import numpy as np
import pytest

from faceheatmap.config import LayoutMode, PersonId
from faceheatmap.detection_loop import TwoPersonDetectionLoop
from faceheatmap.landmarker import FaceObservation
from faceheatmap.layout import BBox

FRAME_W, FRAME_H = 800, 400


def _write_blank_video(path: str, num_frames: int, size=(FRAME_W, FRAME_H)) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 10.0, size)
    for _ in range(num_frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()


def _obs(cx: float, cy: float) -> FaceObservation:
    bbox = BBox(cx - 30, cy - 30, cx + 30, cy + 30)
    landmarks = np.zeros((478, 3))
    return FaceObservation(bbox=bbox, landmarks_px=landmarks, transform_matrix=np.eye(4))


class ScriptedLandmarker:
    """Returns a pre-scripted list of observations per call, in order."""

    def __init__(self, script: list[list[FaceObservation]]):
        self.script = script
        self.calls = 0

    def detect(self, frame_bgr, timestamp_ms):
        result = self.script[self.calls]
        self.calls += 1
        return result


def test_yields_one_frame_per_processed_frame(tmp_path):
    video_path = str(tmp_path / "v.mp4")
    _write_blank_video(video_path, num_frames=5)
    script = [[_obs(200, 200), _obs(600, 200)] for _ in range(5)]
    loop = TwoPersonDetectionLoop(video_path, ScriptedLandmarker(script))

    detected = list(loop)
    assert len(detected) == 5
    assert all(len(d.person_obs) == 2 for d in detected)
    assert loop.layout_used is LayoutMode.SIDE_BY_SIDE


def test_frames_before_two_faces_seen_have_empty_person_obs(tmp_path):
    video_path = str(tmp_path / "v.mp4")
    _write_blank_video(video_path, num_frames=4)
    script = [
        [_obs(200, 200)],  # only 1 face
        [],  # no faces
        [_obs(200, 200), _obs(600, 200)],  # layout established here
        [_obs(200, 200), _obs(600, 200)],
    ]
    loop = TwoPersonDetectionLoop(video_path, ScriptedLandmarker(script))
    detected = list(loop)

    assert len(detected) == 4  # every processed frame yields, even pre-layout ones
    assert detected[0].person_obs == {}
    assert detected[1].person_obs == {}
    assert len(detected[2].person_obs) == 2
    assert len(detected[3].person_obs) == 2


def test_probe_populates_dimensions_before_iteration(tmp_path):
    video_path = str(tmp_path / "v.mp4")
    _write_blank_video(video_path, num_frames=2)
    loop = TwoPersonDetectionLoop(video_path, ScriptedLandmarker([[], []]))
    assert loop.frame_w == FRAME_W
    assert loop.frame_h == FRAME_H
    assert loop.panels is None  # not established until iteration happens


def test_missing_video_raises_immediately(tmp_path):
    with pytest.raises(FileNotFoundError):
        TwoPersonDetectionLoop(str(tmp_path / "nope.mp4"), ScriptedLandmarker([]))


def test_respects_frame_stride_and_max_frames(tmp_path):
    video_path = str(tmp_path / "v.mp4")
    _write_blank_video(video_path, num_frames=20)
    script = [[_obs(200, 200), _obs(600, 200)] for _ in range(20)]
    loop = TwoPersonDetectionLoop(video_path, ScriptedLandmarker(script), frame_stride=2, max_frames=4)
    detected = list(loop)
    assert len(detected) == 4
    assert [d.frame_index for d in detected] == [0, 2, 4, 6]


def test_identity_persists_across_frames_via_tracker(tmp_path):
    video_path = str(tmp_path / "v.mp4")
    _write_blank_video(video_path, num_frames=2)
    script = [
        [_obs(200, 200), _obs(600, 200)],
        [_obs(600, 200), _obs(200, 200)],  # order swapped this frame
    ]
    loop = TwoPersonDetectionLoop(video_path, ScriptedLandmarker(script))
    detected = list(loop)
    assert detected[0].person_obs[PersonId.PERSON_1].bbox.x0 == pytest.approx(170)
    assert detected[1].person_obs[PersonId.PERSON_1].bbox.x0 == pytest.approx(170)
