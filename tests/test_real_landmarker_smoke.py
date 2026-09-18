"""Smoke test against the real MediaPipe model, using a synthetic two-panel
frame built from a real face photo (tests/fixtures/portrait.jpg, a public
MediaPipe sample asset). Skipped if the model bundle hasn't been downloaded.
"""
import os

import cv2
import numpy as np
import pytest

from faceheatmap.calibration import CALIBRATION_POINTS, CalibrationConfig, run_calibration
from faceheatmap.cli import main as cli_main
from faceheatmap.config import LayoutMode, PersonId, PipelineConfig
from faceheatmap.pipeline import FaceHeatmapPipeline

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "face_landmarker.task")
FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "portrait.jpg")

requires_model = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="face_landmarker.task not present; run scripts/download_models.sh first",
)


def _build_two_panel_video(path: str, num_frames: int = 5) -> None:
    face = cv2.imread(FIXTURE_PATH)
    assert face is not None, f"missing test fixture at {FIXTURE_PATH}"
    face = cv2.resize(face, (400, 500))

    frame = np.zeros((500, 800, 3), dtype=np.uint8)
    frame[:, :400] = face
    frame[:, 400:] = face

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 10.0, (800, 500))
    for _ in range(num_frames):
        writer.write(frame)
    writer.release()


@requires_model
def test_real_model_detects_both_faces_and_produces_outputs(tmp_path):
    video_path = str(tmp_path / "two_panel.mp4")
    _build_two_panel_video(video_path)

    out_dir = str(tmp_path / "out")
    config = PipelineConfig(model_path=MODEL_PATH)
    result = FaceHeatmapPipeline(video_path, out_dir, config).run()

    assert result.frames_processed == 5
    assert result.frames_with_both_faces == 5

    for person, paths in result.heatmap_paths.items():
        assert "full_frame" in paths
        img = cv2.imread(paths["full_frame"])
        assert img is not None and img.shape == (500, 800, 3)


@requires_model
def test_real_model_calibration_end_to_end(tmp_path):
    from faceheatmap.landmarker import FaceLandmarkerWrapper

    fps = 10.0
    seconds_per_point = 1.0
    num_frames = int(len(CALIBRATION_POINTS) * seconds_per_point * fps)

    video_path = str(tmp_path / "calib.mp4")
    _build_two_panel_video(video_path, num_frames=num_frames)

    with FaceLandmarkerWrapper(model_path=MODEL_PATH, num_faces=2) as landmarker:
        calibration = run_calibration(
            video_path,
            landmarker,
            CalibrationConfig(seconds_per_point=seconds_per_point, trim_start_frac=0.2, trim_end_frac=0.1),
            layout=LayoutMode.SIDE_BY_SIDE,
        )

    assert set(calibration.keys()) == {PersonId.PERSON_1, PersonId.PERSON_2}
    for cal in calibration.values():
        assert len(cal.h_weights) == 3
        assert len(cal.v_weights) == 3


@requires_model
def test_cli_reports_debug_video_path_when_requested(tmp_path, capsys):
    video_path = str(tmp_path / "two_panel.mp4")
    _build_two_panel_video(video_path)
    out_dir = str(tmp_path / "out")

    exit_code = cli_main([video_path, "-o", out_dir, "--model-path", MODEL_PATH, "--debug-video"])
    assert exit_code == 0

    captured = capsys.readouterr()
    debug_path = os.path.join(out_dir, "debug_annotated.mp4")
    assert f"Debug video: {debug_path}" in captured.out
    assert os.path.exists(debug_path)


@requires_model
def test_cli_omits_debug_video_line_when_not_requested(tmp_path, capsys):
    video_path = str(tmp_path / "two_panel.mp4")
    _build_two_panel_video(video_path)
    out_dir = str(tmp_path / "out")

    cli_main([video_path, "-o", out_dir, "--model-path", MODEL_PATH])

    captured = capsys.readouterr()
    assert "Debug video:" not in captured.out
