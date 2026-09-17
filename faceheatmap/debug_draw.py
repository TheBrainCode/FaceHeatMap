"""Shared debug-frame annotation, used by both the main pipeline's
--debug-video and the calibration tool's --debug-video.
"""
from __future__ import annotations

import cv2
import numpy as np

from faceheatmap.config import PersonId
from faceheatmap.face_boundary import build_face_polygon
from faceheatmap.landmarker import FaceObservation
from faceheatmap.layout import BBox

PERSON_COLORS = {PersonId.PERSON_1: (255, 100, 0), PersonId.PERSON_2: (0, 165, 255)}


def draw_debug_overlay(
    frame: np.ndarray,
    person_obs: dict[PersonId, FaceObservation],
    panels: dict[PersonId, BBox] | None = None,
    gaze_points: dict[PersonId, tuple[float, float]] | None = None,
    top_label: str | None = None,
) -> np.ndarray:
    """Draws face-boundary polygons, optional gaze markers, panel dividers,
    and an optional status label (e.g. current calibration point / whether
    this frame counted) onto a copy of `frame`.
    """
    out = frame.copy()

    for person, obs in person_obs.items():
        color = PERSON_COLORS[person]
        polygon = build_face_polygon(obs.landmarks_px).astype(int)
        cv2.polylines(out, [polygon], isClosed=True, color=color, thickness=2)

    for person, point in (gaze_points or {}).items():
        color = PERSON_COLORS[person]
        pt = (int(point[0]), int(point[1]))
        cv2.drawMarker(out, pt, color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    for panel in (panels or {}).values():
        cv2.rectangle(out, (int(panel.x0), int(panel.y0)), (int(panel.x1) - 1, int(panel.y1) - 1), (255, 255, 255), 1)

    if top_label:
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        (text_w, text_h), baseline = cv2.getTextSize(top_label, font, scale, thickness)
        cv2.rectangle(out, (4, 4), (12 + text_w, 12 + text_h + baseline), (0, 0, 0), thickness=-1)
        cv2.putText(out, top_label, (8, 8 + text_h), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return out
