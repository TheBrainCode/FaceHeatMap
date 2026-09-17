"""Face boundary polygons and inside/outside gaze containment stats."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from faceheatmap.landmark_indices import get_landmark_indices


def build_face_polygon(landmarks_px: np.ndarray) -> np.ndarray:
    """Returns the face-oval contour (Nx2) in pixel coordinates, in polygon order."""
    indices = get_landmark_indices()["face_oval"]
    return np.asarray(landmarks_px)[indices, :2].astype(np.float32)


def point_in_polygon(polygon: np.ndarray, x: float, y: float) -> bool:
    """True if (x, y) lies inside or on the boundary of `polygon`."""
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0


@dataclass
class BoundaryStats:
    total: int = 0
    inside: int = 0

    def record(self, is_inside: bool) -> None:
        self.total += 1
        if is_inside:
            self.inside += 1

    @property
    def outside(self) -> int:
        return self.total - self.inside

    @property
    def percent_inside(self) -> float:
        return 100.0 * self.inside / self.total if self.total else 0.0

    @property
    def percent_outside(self) -> float:
        return 100.0 - self.percent_inside if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "total_frames": self.total,
            "frames_inside_face": self.inside,
            "frames_outside_face": self.outside,
            "percent_inside_face": round(self.percent_inside, 2),
            "percent_outside_face": round(self.percent_outside, 2),
        }
