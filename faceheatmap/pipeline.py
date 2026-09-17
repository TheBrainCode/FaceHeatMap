"""Orchestrates the end-to-end gaze-heatmap pipeline over a two-person recording."""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

from faceheatmap.config import PersonId, PipelineConfig, LayoutMode
from faceheatmap.debug_draw import draw_debug_overlay
from faceheatmap.detection_loop import Landmarker, SplitPanelDetectionLoop, TwoPersonDetectionLoop
from faceheatmap.face_boundary import BoundaryStats, build_face_polygon, point_in_polygon
from faceheatmap.gaze import GazeCalibration, GazeResult, SmoothedGazeEstimator
from faceheatmap.heatmap import HeatmapAccumulator, overlay_on_image
from faceheatmap.layout import BBox


@dataclass
class PipelineResult:
    output_dir: str
    frames_processed: int
    frames_with_both_faces: int
    layout_used: LayoutMode
    boundary_stats: dict[str, dict]
    heatmap_paths: dict[str, dict[str, str]]
    summary_json_path: str
    csv_path: str | None = None


def _make_landmarker(config: PipelineConfig, num_faces: int | None = None) -> Landmarker:
    from faceheatmap.landmarker import FaceLandmarkerWrapper

    return FaceLandmarkerWrapper(
        model_path=config.model_path,
        num_faces=num_faces if num_faces is not None else config.num_faces,
        min_detection_confidence=config.min_detection_confidence,
        min_presence_confidence=config.min_presence_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )


class FaceHeatmapPipeline:
    def __init__(
        self,
        video_path: str,
        output_dir: str,
        config: PipelineConfig | None = None,
        landmarker: Landmarker | None = None,
        landmarker_person1: Landmarker | None = None,
        landmarker_person2: Landmarker | None = None,
        calibration: dict[PersonId, GazeCalibration] | None = None,
    ):
        self.video_path = video_path
        self.output_dir = output_dir
        self.config = config or PipelineConfig()
        self._external_landmarker = landmarker
        self._external_landmarker_person1 = landmarker_person1
        self._external_landmarker_person2 = landmarker_person2
        self._calibration = calibration or {}
        self._panels: dict[PersonId, BBox] | None = None
        self._layout_used = self.config.layout

    def run(self) -> PipelineResult:
        os.makedirs(self.output_dir, exist_ok=True)

        if not cv2.VideoCapture(self.video_path).isOpened():
            raise FileNotFoundError(f"Could not open video: {self.video_path}")

        owned_landmarkers: list[Landmarker] = []

        if self.config.split_panel_detection:
            if self.config.layout is LayoutMode.AUTO:
                raise ValueError(
                    "split_panel_detection requires an explicit --layout (side_by_side or "
                    "top_bottom), since panel boundaries must be known before any detection "
                    "can happen -- it can't be auto-detected the way the whole-frame mode can."
                )
            landmarker1 = self._external_landmarker_person1 or _make_landmarker(self.config, num_faces=1)
            landmarker2 = self._external_landmarker_person2 or _make_landmarker(self.config, num_faces=1)
            owned_landmarkers = [lm for lm, external in [(landmarker1, self._external_landmarker_person1), (landmarker2, self._external_landmarker_person2)] if external is None]
            loop = SplitPanelDetectionLoop(
                self.video_path,
                landmarker1,
                landmarker2,
                layout=self.config.layout,
                frame_stride=self.config.frame_stride,
                max_frames=self.config.max_frames,
            )
        else:
            landmarker = self._external_landmarker or _make_landmarker(self.config)
            if self._external_landmarker is None:
                owned_landmarkers = [landmarker]
            loop = TwoPersonDetectionLoop(
                self.video_path,
                landmarker,
                layout=self.config.layout,
                frame_stride=self.config.frame_stride,
                max_frames=self.config.max_frames,
            )
        frame_w, frame_h = loop.frame_w, loop.frame_h

        gaze_estimators = {
            PersonId.PERSON_1: SmoothedGazeEstimator(self.config.gaze, self._calibration.get(PersonId.PERSON_1)),
            PersonId.PERSON_2: SmoothedGazeEstimator(self.config.gaze, self._calibration.get(PersonId.PERSON_2)),
        }
        heatmaps = {
            PersonId.PERSON_1: HeatmapAccumulator(frame_w, frame_h, self.config.heatmap),
            PersonId.PERSON_2: HeatmapAccumulator(frame_w, frame_h, self.config.heatmap),
        }
        boundary_stats = {PersonId.PERSON_1: BoundaryStats(), PersonId.PERSON_2: BoundaryStats()}

        background_frame: np.ndarray | None = None
        first_frame_seen: np.ndarray | None = None
        csv_rows: list[dict] = []
        csv_fieldnames = [
            "frame_index", "timestamp_ms",
            "person1_yaw_deg", "person1_pitch_deg", "person1_frame_x", "person1_frame_y", "person1_valid",
            "person2_yaw_deg", "person2_pitch_deg", "person2_frame_x", "person2_frame_y", "person2_valid",
        ]

        debug_writer = None
        if self.config.write_debug_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            debug_path = os.path.join(self.output_dir, "debug_annotated.mp4")
            debug_writer = cv2.VideoWriter(debug_path, fourcc, loop.fps / max(self.config.frame_stride, 1), (frame_w, frame_h))

        processed = 0
        frames_with_both = 0

        try:
            for detected in loop:
                processed += 1

                if first_frame_seen is None:
                    first_frame_seen = detected.frame.copy()

                person_obs = detected.person_obs
                if len(person_obs) == 2:
                    frames_with_both += 1
                    if background_frame is None:
                        # Prefer a frame that actually shows both people over
                        # frame 0, which can be a black/blank leading frame
                        # (e.g. a brief intro before the call video starts).
                        background_frame = detected.frame.copy()

                frame_gaze: dict[PersonId, GazeResult] = {}
                for person, obs in person_obs.items():
                    frame_gaze[person] = gaze_estimators[person].update(obs.landmarks_px, obs.transform_matrix, frame_w, frame_h)

                for person, gaze_result in frame_gaze.items():
                    if not gaze_result.valid:
                        continue
                    heatmaps[person].add_point(gaze_result.frame_x, gaze_result.frame_y)

                    other = person.other
                    if other in person_obs:
                        polygon = build_face_polygon(person_obs[other].landmarks_px)
                        inside = point_in_polygon(polygon, gaze_result.frame_x, gaze_result.frame_y)
                        boundary_stats[person].record(inside)

                if self.config.write_csv_log:
                    row = {"frame_index": detected.frame_index, "timestamp_ms": detected.timestamp_ms}
                    for person in (PersonId.PERSON_1, PersonId.PERSON_2):
                        g = frame_gaze.get(person)
                        prefix = person.value
                        row[f"{prefix}_yaw_deg"] = g.yaw_deg if g else ""
                        row[f"{prefix}_pitch_deg"] = g.pitch_deg if g else ""
                        row[f"{prefix}_frame_x"] = g.frame_x if g else ""
                        row[f"{prefix}_frame_y"] = g.frame_y if g else ""
                        row[f"{prefix}_valid"] = g.valid if g else ""
                    csv_rows.append(row)

                if debug_writer is not None:
                    gaze_points = {p: (g.frame_x, g.frame_y) for p, g in frame_gaze.items() if g.valid}
                    debug_writer.write(draw_debug_overlay(detected.frame, person_obs, loop.panels, gaze_points))
        finally:
            if debug_writer is not None:
                debug_writer.release()
            for lm in owned_landmarkers:
                if hasattr(lm, "close"):
                    lm.close()

        self._panels = loop.panels
        self._layout_used = loop.layout_used or self.config.layout
        if self._panels is None:
            raise RuntimeError(
                "Never detected two faces in the same frame; cannot establish a "
                "screen layout. Check that the video actually shows both participants."
            )

        if background_frame is None:
            background_frame = first_frame_seen

        heatmap_paths = self._render_outputs(background_frame, heatmaps)
        summary_path = self._write_summary(processed, frames_with_both, boundary_stats, heatmap_paths)
        csv_path = self._write_csv(csv_rows, csv_fieldnames) if self.config.write_csv_log else None

        return PipelineResult(
            output_dir=self.output_dir,
            frames_processed=processed,
            frames_with_both_faces=frames_with_both,
            layout_used=self._layout_used,
            boundary_stats={p.value: s.to_dict() for p, s in boundary_stats.items()},
            heatmap_paths=heatmap_paths,
            summary_json_path=summary_path,
            csv_path=csv_path,
        )

    def _render_outputs(self, background: np.ndarray | None, heatmaps: dict[PersonId, HeatmapAccumulator]) -> dict[str, dict[str, str]]:
        paths: dict[str, dict[str, str]] = {}
        panels = self._panels
        assert panels is not None

        for person, heatmap in heatmaps.items():
            other = person.other
            person_paths = {}

            if background is not None:
                full_normalized = heatmap.render_full(normalize=True)
                full_overlay = overlay_on_image(background, full_normalized, self.config.heatmap.overlay_alpha, self.config.heatmap.colormap)
                full_path = os.path.join(self.output_dir, f"{person.value}_gaze_full_frame.png")
                cv2.imwrite(full_path, full_overlay)
                person_paths["full_frame"] = full_path

                other_panel = panels[other]
                x0, y0, x1, y1 = int(other_panel.x0), int(other_panel.y0), int(other_panel.x1), int(other_panel.y1)
                panel_crop = background[y0:y1, x0:x1]
                region_normalized = heatmap.render_region(other_panel, normalize=True)
                if panel_crop.size > 0:
                    region_overlay = overlay_on_image(panel_crop, region_normalized, self.config.heatmap.overlay_alpha, self.config.heatmap.colormap)
                    region_path = os.path.join(self.output_dir, f"{person.value}_gaze_on_{other.value}_panel.png")
                    cv2.imwrite(region_path, region_overlay)
                    person_paths[f"gaze_on_{other.value}_panel"] = region_path

            paths[person.value] = person_paths
        return paths

    def _write_summary(self, processed: int, frames_with_both: int, boundary_stats: dict[PersonId, BoundaryStats], heatmap_paths: dict) -> str:
        summary = {
            "video_path": self.video_path,
            "layout_used": self._layout_used.value if self._layout_used else None,
            "frames_processed": processed,
            "frames_with_both_faces": frames_with_both,
            "boundary_stats": {
                p.value: {
                    "description": f"How much {p.value}'s gaze landed inside {p.other.value}'s face boundary",
                    **stats.to_dict(),
                }
                for p, stats in boundary_stats.items()
            },
            "heatmap_paths": heatmap_paths,
        }
        path = os.path.join(self.output_dir, "summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        return path

    def _write_csv(self, rows: list[dict], fieldnames: list[str]) -> str:
        path = os.path.join(self.output_dir, "gaze_log.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path
