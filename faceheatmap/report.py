"""Builds a single-page HTML report from a pipeline output directory.

Combines the heatmap PNGs the pipeline already writes with a gaze-over-time
chart derived from gaze_log.csv, so results can be skimmed as one page
instead of opening four separate images plus a JSON file.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_summary(output_dir: str) -> dict:
    path = os.path.join(output_dir, "summary.json")
    with open(path) as f:
        return json.load(f)


def read_gaze_timeseries(output_dir: str) -> dict[str, dict[str, list[float]]]:
    """Returns {"person1": {"t": [...], "frame_x": [...], "frame_y": [...]}, "person2": {...}}.

    Only rows where the person's gaze was valid (i.e. counted toward the
    heatmap) are included; invalid/missing frames show up as a gap.
    """
    path = os.path.join(output_dir, "gaze_log.csv")
    series: dict[str, dict[str, list[float]]] = {
        "person1": {"t": [], "frame_x": [], "frame_y": []},
        "person2": {"t": [], "frame_x": [], "frame_y": []},
    }
    if not os.path.exists(path):
        return series

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            t = float(row["timestamp_ms"]) / 1000.0
            for person in ("person1", "person2"):
                valid = row.get(f"{person}_valid") == "True"
                fx = row.get(f"{person}_frame_x", "")
                fy = row.get(f"{person}_frame_y", "")
                series[person]["t"].append(t)
                series[person]["frame_x"].append(float(fx) if valid and fx != "" else math.nan)
                series[person]["frame_y"].append(float(fy) if valid and fy != "" else math.nan)
    return series


def plot_gaze_timeline(series: dict[str, dict[str, list[float]]], out_path: str) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, person in zip(axes, ("person1", "person2")):
        data = series[person]
        ax.plot(data["t"], data["frame_x"], label="horizontal (frame_x)", linewidth=0.8)
        ax.plot(data["t"], data["frame_y"], label="vertical (frame_y)", linewidth=0.8)
        ax.set_title(f"{person} mapped gaze point over time")
        ax.set_ylabel("pixels on frame")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _stats_table_rows(boundary_stats: dict) -> str:
    rows = []
    for person, stats in boundary_stats.items():
        rows.append(
            f"<tr><td>{person}</td><td>{stats['total_frames']}</td>"
            f"<td>{stats['percent_inside_face']}%</td><td>{stats['percent_outside_face']}%</td></tr>"
        )
    return "\n".join(rows)


def _heatmap_gallery(output_dir: str, heatmap_paths: dict) -> str:
    figures = []
    for person, paths in heatmap_paths.items():
        for kind, path in paths.items():
            basename = os.path.basename(path)
            if not os.path.exists(os.path.join(output_dir, basename)):
                continue
            label = f"{person} &mdash; {kind.replace('_', ' ')}"
            figures.append(f'<figure><img src="{basename}" alt="{label}"><figcaption>{label}</figcaption></figure>')
    return "\n".join(figures)


def build_html_report(output_dir: str) -> str:
    summary = read_summary(output_dir)
    series = read_gaze_timeseries(output_dir)
    timeline_path = plot_gaze_timeline(series, os.path.join(output_dir, "gaze_timeline.png"))
    timeline_basename = os.path.basename(timeline_path)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>faceheatmap report</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
  h1 {{ margin-bottom: 0.2rem; }}
  .meta {{ color: #666; margin-bottom: 1.5rem; }}
  table {{ border-collapse: collapse; margin-bottom: 1.5rem; }}
  th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.8rem; text-align: left; }}
  th {{ background: #f4f4f4; }}
  .gallery {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
  figure {{ margin: 0; width: 45%; min-width: 320px; }}
  figure img {{ width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
  figcaption {{ font-size: 0.85rem; color: #555; margin-top: 0.3rem; }}
  .note {{ background: #fff8e1; border: 1px solid #ffe082; border-radius: 4px; padding: 0.75rem 1rem; font-size: 0.9rem; margin-top: 1.5rem; }}
</style>
</head>
<body>
  <h1>faceheatmap report</h1>
  <div class="meta">
    Video: {summary.get('video_path', 'n/a')}<br>
    Layout: {summary.get('layout_used', 'n/a')}<br>
    Frames processed: {summary.get('frames_processed', 'n/a')}
    (both faces detected in {summary.get('frames_with_both_faces', 'n/a')})
  </div>

  <h2>Face-boundary attention</h2>
  <table>
    <tr><th>Person</th><th>Frames counted</th><th>% inside counterpart's face</th><th>% outside</th></tr>
    {_stats_table_rows(summary.get('boundary_stats', {}))}
  </table>

  <h2>Gaze heatmaps</h2>
  <div class="gallery">
    {_heatmap_gallery(output_dir, summary.get('heatmap_paths', {}))}
  </div>

  <h2>Gaze over time</h2>
  <img src="{timeline_basename}" alt="Gaze position over time" style="width:100%;border:1px solid #ddd;border-radius:4px;">

  <div class="note">
    Gaze direction is estimated heuristically (head pose + iris offset,
    no true screen calibration) &mdash; see the project README's
    "Important limitation" section before drawing firm conclusions from
    these numbers.
  </div>
</body>
</html>
"""
    path = os.path.join(output_dir, "report.html")
    with open(path, "w") as f:
        f.write(html)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="faceheatmap-report", description="Build an HTML report from a faceheatmap output directory")
    parser.add_argument("output_dir", help="Output directory produced by `faceheatmap` (must contain summary.json and gaze_log.csv)")
    args = parser.parse_args(argv)

    report_path = build_html_report(args.output_dir)
    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
