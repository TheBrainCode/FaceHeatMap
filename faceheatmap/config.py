"""Configuration dataclasses for the faceheatmap pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class LayoutMode(str, Enum):
    """How the two participants are arranged in the recorded frame."""

    SIDE_BY_SIDE = "side_by_side"
    TOP_BOTTOM = "top_bottom"
    AUTO = "auto"


class PersonId(str, Enum):
    PERSON_1 = "person1"
    PERSON_2 = "person2"

    @property
    def other(self) -> "PersonId":
        return PersonId.PERSON_2 if self is PersonId.PERSON_1 else PersonId.PERSON_1


@dataclass
class GazeConfig:
    """Parameters for the calibration-free gaze-to-screen mapping.

    Gaze is approximated as head pose (yaw/pitch, extracted from MediaPipe's
    facial transformation matrix) plus an iris-offset correction, then
    projected onto the recording frame using an assumed horizontal/vertical
    field of view -- there is no real screen calibration available from a
    recording alone, so this is a best-effort heuristic, not ground truth.
    """

    fov_h_deg: float = 40.0
    fov_v_deg: float = 24.0
    eye_gain_h_deg: float = 22.0
    eye_gain_v_deg: float = 18.0
    smoothing_alpha: float = 0.4  # exponential smoothing of the gaze point, 0 = no smoothing
    flip_yaw: bool = False
    flip_pitch: bool = False
    max_valid_yaw_deg: float = 80.0
    max_valid_pitch_deg: float = 70.0


@dataclass
class HeatmapConfig:
    sigma_px: float = 35.0
    resolution_scale: float = 0.5  # fraction of frame resolution used for the accumulator grid
    colormap: str = "jet"
    overlay_alpha: float = 0.55


@dataclass
class PipelineConfig:
    layout: LayoutMode = LayoutMode.AUTO
    model_path: str = "models/face_landmarker.task"
    num_faces: int = 2
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    frame_stride: int = 1  # process every Nth frame
    max_frames: int | None = None
    gaze: GazeConfig = field(default_factory=GazeConfig)
    heatmap: HeatmapConfig = field(default_factory=HeatmapConfig)
    write_debug_video: bool = False
    write_csv_log: bool = True
    split_panel_detection: bool = False
    """Detect each panel independently instead of running num_faces=2 on the
    whole frame. Fixes cases where MediaPipe's detector only returns one
    face on a combined frame with two very differently-scaled faces (e.g.
    one participant much closer to their camera than the other). Requires
    `layout` to be explicit (side_by_side/top_bottom), not auto.
    """
