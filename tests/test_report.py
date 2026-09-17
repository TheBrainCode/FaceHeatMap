import csv
import json
import os

import cv2
import numpy as np

from faceheatmap.report import build_html_report, read_gaze_timeseries, read_summary


def _write_fixture_output_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    summary = {
        "video_path": "example.mp4",
        "layout_used": "side_by_side",
        "frames_processed": 3,
        "frames_with_both_faces": 3,
        "boundary_stats": {
            "person1": {"total_frames": 3, "frames_inside_face": 1, "frames_outside_face": 2, "percent_inside_face": 33.33, "percent_outside_face": 66.67},
            "person2": {"total_frames": 3, "frames_inside_face": 0, "frames_outside_face": 3, "percent_inside_face": 0.0, "percent_outside_face": 100.0},
        },
        "heatmap_paths": {
            "person1": {"full_frame": os.path.join(out_dir, "person1_gaze_full_frame.png"), "gaze_on_person2_panel": os.path.join(out_dir, "person1_gaze_on_person2_panel.png")},
            "person2": {"full_frame": os.path.join(out_dir, "person2_gaze_full_frame.png"), "gaze_on_person1_panel": os.path.join(out_dir, "person2_gaze_on_person1_panel.png")},
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f)

    for paths in summary["heatmap_paths"].values():
        for path in paths.values():
            cv2.imwrite(path, np.zeros((20, 20, 3), dtype=np.uint8))

    fieldnames = [
        "frame_index", "timestamp_ms",
        "person1_yaw_deg", "person1_pitch_deg", "person1_frame_x", "person1_frame_y", "person1_valid",
        "person2_yaw_deg", "person2_pitch_deg", "person2_frame_x", "person2_frame_y", "person2_valid",
    ]
    rows = [
        {"frame_index": 0, "timestamp_ms": 0, "person1_yaw_deg": 1.0, "person1_pitch_deg": 0.5, "person1_frame_x": 400, "person1_frame_y": 200, "person1_valid": True,
         "person2_yaw_deg": -2.0, "person2_pitch_deg": 0.0, "person2_frame_x": 600, "person2_frame_y": 210, "person2_valid": True},
        {"frame_index": 1, "timestamp_ms": 100, "person1_yaw_deg": "", "person1_pitch_deg": "", "person1_frame_x": "", "person1_frame_y": "", "person1_valid": "",
         "person2_yaw_deg": -1.0, "person2_pitch_deg": 0.2, "person2_frame_x": 610, "person2_frame_y": 205, "person2_valid": True},
        {"frame_index": 2, "timestamp_ms": 200, "person1_yaw_deg": 30.0, "person1_pitch_deg": 0.1, "person1_frame_x": 900, "person1_frame_y": 220, "person1_valid": False,
         "person2_yaw_deg": -1.5, "person2_pitch_deg": 0.1, "person2_frame_x": 605, "person2_frame_y": 208, "person2_valid": True},
    ]
    with open(os.path.join(out_dir, "gaze_log.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_read_summary(tmp_path):
    out_dir = str(tmp_path / "out")
    _write_fixture_output_dir(out_dir)
    summary = read_summary(out_dir)
    assert summary["layout_used"] == "side_by_side"
    assert summary["boundary_stats"]["person1"]["percent_inside_face"] == 33.33


def test_read_gaze_timeseries_skips_invalid_and_missing_rows(tmp_path):
    out_dir = str(tmp_path / "out")
    _write_fixture_output_dir(out_dir)
    series = read_gaze_timeseries(out_dir)

    assert series["person1"]["t"] == [0.0, 0.1, 0.2]
    assert series["person1"]["frame_x"][0] == 400.0
    assert series["person1"]["frame_x"][1] != series["person1"]["frame_x"][1]  # NaN (missing row)
    assert series["person1"]["frame_x"][2] != series["person1"]["frame_x"][2]  # NaN (invalid)

    assert series["person2"]["frame_x"] == [600.0, 610.0, 605.0]


def test_build_html_report_creates_report_and_timeline_chart(tmp_path):
    out_dir = str(tmp_path / "out")
    _write_fixture_output_dir(out_dir)

    report_path = build_html_report(out_dir)
    assert os.path.exists(report_path)

    timeline_path = os.path.join(out_dir, "gaze_timeline.png")
    assert os.path.exists(timeline_path)
    img = cv2.imread(timeline_path)
    assert img is not None and img.size > 0

    with open(report_path) as f:
        content = f.read()
    assert "person1" in content
    assert "33.33%" in content
    assert "gaze_timeline.png" in content
    assert "person1_gaze_full_frame.png" in content


def test_read_gaze_timeseries_missing_csv_returns_empty_series(tmp_path):
    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir)
    series = read_gaze_timeseries(out_dir)
    assert series["person1"]["t"] == []
