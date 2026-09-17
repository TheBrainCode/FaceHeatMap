import cv2
import numpy as np
import pytest

from faceheatmap.config import LayoutMode, PersonId
from faceheatmap.detection_loop import SplitPanelDetectionLoop, TwoPersonDetectionLoop, _offset_observation
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


# --- SplitPanelDetectionLoop ---


def test_offset_observation_shifts_bbox_and_landmarks():
    landmarks = np.zeros((478, 3))
    landmarks[0] = [5.0, 10.0, 1.0]
    obs = FaceObservation(bbox=BBox(0, 0, 20, 30), landmarks_px=landmarks, transform_matrix=np.eye(4) * 2)

    shifted = _offset_observation(obs, dx=100, dy=50)

    assert shifted.bbox == BBox(100, 50, 120, 80)
    assert shifted.landmarks_px[0].tolist() == [105.0, 60.0, 1.0]
    assert np.array_equal(shifted.transform_matrix, obs.transform_matrix)  # unaffected
    # Original observation must not be mutated.
    assert obs.landmarks_px[0].tolist() == [5.0, 10.0, 1.0]


def test_split_panel_loop_rejects_auto_layout(tmp_path):
    video_path = str(tmp_path / "v.mp4")
    _write_blank_video(video_path, num_frames=1)
    with pytest.raises(ValueError, match="explicit layout"):
        SplitPanelDetectionLoop(video_path, ScriptedLandmarker([]), ScriptedLandmarker([]), layout=LayoutMode.AUTO)


def test_split_panel_loop_establishes_panels_immediately(tmp_path):
    video_path = str(tmp_path / "v.mp4")
    _write_blank_video(video_path, num_frames=1)
    loop = SplitPanelDetectionLoop(video_path, ScriptedLandmarker([[]]), ScriptedLandmarker([[]]), layout=LayoutMode.SIDE_BY_SIDE)
    assert loop.layout_used is LayoutMode.SIDE_BY_SIDE
    assert loop.panels[PersonId.PERSON_1] == BBox(0, 0, FRAME_W / 2, FRAME_H)
    assert loop.panels[PersonId.PERSON_2] == BBox(FRAME_W / 2, 0, FRAME_W, FRAME_H)


def test_split_panel_loop_offsets_detections_into_full_frame_coords(tmp_path):
    video_path = str(tmp_path / "v.mp4")
    _write_blank_video(video_path, num_frames=1)

    # Each landmarker sees only its own panel crop, so it reports a
    # panel-local bbox (e.g. near the crop's own origin).
    local_obs = _obs(50, 50)
    landmarker1 = ScriptedLandmarker([[local_obs]])
    landmarker2 = ScriptedLandmarker([[local_obs]])

    loop = SplitPanelDetectionLoop(video_path, landmarker1, landmarker2, layout=LayoutMode.SIDE_BY_SIDE)
    detected = list(loop)
    assert len(detected) == 1

    p1_bbox = detected[0].person_obs[PersonId.PERSON_1].bbox
    p2_bbox = detected[0].person_obs[PersonId.PERSON_2].bbox
    # Person 1's panel starts at x=0, so local and full-frame coords match.
    assert p1_bbox == local_obs.bbox
    # Person 2's panel starts at x=FRAME_W/2, so that offset must be added.
    assert p2_bbox.x0 == pytest.approx(local_obs.bbox.x0 + FRAME_W / 2)
    assert p2_bbox.y0 == pytest.approx(local_obs.bbox.y0)


def test_split_panel_loop_handles_one_panel_missing_a_detection(tmp_path):
    video_path = str(tmp_path / "v.mp4")
    _write_blank_video(video_path, num_frames=1)

    landmarker1 = ScriptedLandmarker([[_obs(50, 50)]])
    landmarker2 = ScriptedLandmarker([[]])  # no face found this frame

    loop = SplitPanelDetectionLoop(video_path, landmarker1, landmarker2, layout=LayoutMode.SIDE_BY_SIDE)
    detected = list(loop)
    assert set(detected[0].person_obs.keys()) == {PersonId.PERSON_1}


def test_split_panel_loop_no_identity_ambiguity_by_construction(tmp_path):
    # Each landmarker is permanently bound to its own panel, so there's
    # nothing to track/swap -- unlike TwoPersonDetectionLoop's IoU tracker.
    video_path = str(tmp_path / "v.mp4")
    _write_blank_video(video_path, num_frames=2)

    landmarker1 = ScriptedLandmarker([[_obs(50, 50)], [_obs(55, 52)]])
    landmarker2 = ScriptedLandmarker([[_obs(45, 48)], [_obs(48, 50)]])

    loop = SplitPanelDetectionLoop(video_path, landmarker1, landmarker2, layout=LayoutMode.SIDE_BY_SIDE)
    detected = list(loop)
    for d in detected:
        assert PersonId.PERSON_1 in d.person_obs
        assert PersonId.PERSON_2 in d.person_obs


def test_split_panel_loop_respects_frame_stride_and_max_frames(tmp_path):
    video_path = str(tmp_path / "v.mp4")
    _write_blank_video(video_path, num_frames=20)
    obs_list = [[_obs(50, 50)] for _ in range(20)]
    landmarker1 = ScriptedLandmarker(obs_list)
    landmarker2 = ScriptedLandmarker(obs_list)

    loop = SplitPanelDetectionLoop(video_path, landmarker1, landmarker2, layout=LayoutMode.SIDE_BY_SIDE, frame_stride=2, max_frames=4)
    detected = list(loop)
    assert len(detected) == 4
    assert [d.frame_index for d in detected] == [0, 2, 4, 6]


def test_split_panel_loop_missing_video_raises_immediately(tmp_path):
    with pytest.raises(FileNotFoundError):
        SplitPanelDetectionLoop(str(tmp_path / "nope.mp4"), ScriptedLandmarker([]), ScriptedLandmarker([]), layout=LayoutMode.SIDE_BY_SIDE)
