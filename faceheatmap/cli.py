"""Command-line entry point for faceheatmap."""
from __future__ import annotations

import argparse
import sys

from faceheatmap.config import GazeConfig, HeatmapConfig, LayoutMode, PipelineConfig
from faceheatmap.pipeline import FaceHeatmapPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="faceheatmap",
        description=(
            "Estimate gaze direction for two people in a Zoom-style recording and "
            "produce per-person gaze heatmaps plus face-boundary attention stats."
        ),
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("-o", "--output-dir", default="output", help="Directory to write results into")
    parser.add_argument(
        "--layout",
        choices=[m.value for m in LayoutMode],
        default=LayoutMode.AUTO.value,
        help="Panel arrangement of the two participants (default: auto-detect side-by-side vs top-bottom)",
    )
    parser.add_argument("--model-path", default="models/face_landmarker.task", help="Path to the MediaPipe FaceLandmarker .task model")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame (default: 1, every frame)")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after processing this many frames (default: whole video)")

    parser.add_argument("--fov-h-deg", type=float, default=40.0, help="Assumed horizontal field of view of the person's screen, in degrees")
    parser.add_argument("--fov-v-deg", type=float, default=24.0, help="Assumed vertical field of view of the person's screen, in degrees")
    parser.add_argument("--eye-gain-h-deg", type=float, default=22.0, help="Max horizontal gaze deflection attributed to eye movement alone, in degrees")
    parser.add_argument("--eye-gain-v-deg", type=float, default=18.0, help="Max vertical gaze deflection attributed to eye movement alone, in degrees")
    parser.add_argument("--smoothing-alpha", type=float, default=0.4, help="Exponential smoothing factor for the gaze point, 0=no smoothing, 1=no history")
    parser.add_argument("--flip-yaw", action="store_true", help="Invert the left/right gaze direction (use if heatmaps look mirrored)")
    parser.add_argument("--flip-pitch", action="store_true", help="Invert the up/down gaze direction (use if heatmaps look mirrored)")

    parser.add_argument("--sigma-px", type=float, default=35.0, help="Gaussian blur radius (pixels) for the heatmap")
    parser.add_argument("--overlay-alpha", type=float, default=0.55, help="Max opacity of the heatmap overlay at peak intensity")
    parser.add_argument("--colormap", default="jet", help="Matplotlib colormap name for the heatmap")

    parser.add_argument("--debug-video", action="store_true", help="Also write an annotated debug video with face boundaries and gaze markers")
    parser.add_argument("--no-csv", action="store_true", help="Skip writing the per-frame gaze_log.csv")

    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        layout=LayoutMode(args.layout),
        model_path=args.model_path,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        gaze=GazeConfig(
            fov_h_deg=args.fov_h_deg,
            fov_v_deg=args.fov_v_deg,
            eye_gain_h_deg=args.eye_gain_h_deg,
            eye_gain_v_deg=args.eye_gain_v_deg,
            smoothing_alpha=args.smoothing_alpha,
            flip_yaw=args.flip_yaw,
            flip_pitch=args.flip_pitch,
        ),
        heatmap=HeatmapConfig(
            sigma_px=args.sigma_px,
            overlay_alpha=args.overlay_alpha,
            colormap=args.colormap,
        ),
        write_debug_video=args.debug_video,
        write_csv_log=not args.no_csv,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)

    pipeline = FaceHeatmapPipeline(args.video, args.output_dir, config)
    result = pipeline.run()

    print(f"Layout used: {result.layout_used.value}")
    print(f"Frames processed: {result.frames_processed} (both faces detected in {result.frames_with_both_faces})")
    for person, stats in result.boundary_stats.items():
        print(f"{person}: {stats['percent_inside_face']}% of gaze inside counterpart's face boundary, {stats['percent_outside_face']}% outside")
    print(f"Summary written to {result.summary_json_path}")
    for person, paths in result.heatmap_paths.items():
        for kind, path in paths.items():
            print(f"  {person} [{kind}]: {path}")
    if result.csv_path:
        print(f"Per-frame log: {result.csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
