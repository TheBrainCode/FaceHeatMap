import pytest

from faceheatmap.cli import build_arg_parser, config_from_args, main
from faceheatmap.config import LayoutMode


def test_default_args_build_expected_config():
    parser = build_arg_parser()
    args = parser.parse_args(["video.mp4"])
    config = config_from_args(args)
    assert config.layout is LayoutMode.AUTO
    assert config.gaze.fov_h_deg == 40.0
    assert config.heatmap.sigma_px == 35.0
    assert config.write_csv_log is True
    assert config.write_debug_video is False
    assert config.split_panel_detection is False


def test_flags_are_wired_through():
    parser = build_arg_parser()
    args = parser.parse_args([
        "video.mp4",
        "--layout", "top_bottom",
        "--flip-yaw",
        "--flip-pitch",
        "--debug-video",
        "--no-csv",
        "--fov-h-deg", "50",
        "--sigma-px", "10",
        "--split-panel-detection",
    ])
    config = config_from_args(args)
    assert config.layout is LayoutMode.TOP_BOTTOM
    assert config.gaze.flip_yaw is True
    assert config.gaze.flip_pitch is True
    assert config.write_debug_video is True
    assert config.write_csv_log is False
    assert config.gaze.fov_h_deg == 50.0
    assert config.heatmap.sigma_px == 10.0
    assert config.split_panel_detection is True


def test_split_panel_detection_with_auto_layout_errors_out():
    with pytest.raises(SystemExit):
        main(["video.mp4", "--split-panel-detection"])  # layout defaults to auto
