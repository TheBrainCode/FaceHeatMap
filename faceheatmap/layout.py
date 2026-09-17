"""Splits a two-person Zoom recording frame into per-participant screen panels.

We assume the recording is a standard two-person gallery/side-by-side
layout where each participant occupies one half of the frame -- this is
what Zoom produces by default for a 2-person call, whether recorded via
screen capture or the built-in gallery-layout recording option.
"""
from __future__ import annotations

from dataclasses import dataclass

from faceheatmap.config import LayoutMode, PersonId


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def contains_point(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


def infer_layout_from_bboxes(bbox_a: BBox, bbox_b: BBox) -> LayoutMode:
    """Guesses side-by-side vs top-bottom from which axis separates the two faces more."""
    cx_a, cy_a = bbox_a.center
    cx_b, cy_b = bbox_b.center
    if abs(cx_a - cx_b) >= abs(cy_a - cy_b):
        return LayoutMode.SIDE_BY_SIDE
    return LayoutMode.TOP_BOTTOM


def get_panels(
    frame_w: int, frame_h: int, layout: LayoutMode
) -> dict[PersonId, BBox]:
    """Returns the two half-frame panel rectangles for a resolved (non-AUTO) layout."""
    if layout is LayoutMode.AUTO:
        layout = LayoutMode.SIDE_BY_SIDE

    if layout is LayoutMode.SIDE_BY_SIDE:
        mid = frame_w / 2
        return {
            PersonId.PERSON_1: BBox(0, 0, mid, frame_h),
            PersonId.PERSON_2: BBox(mid, 0, frame_w, frame_h),
        }

    mid = frame_h / 2
    return {
        PersonId.PERSON_1: BBox(0, 0, frame_w, mid),
        PersonId.PERSON_2: BBox(0, mid, frame_w, frame_h),
    }


def iou(a: BBox, b: BBox) -> float:
    ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    intersection = iw * ih
    union = a.width * a.height + b.width * b.height - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def assign_panel(bbox: BBox, panels: dict[PersonId, BBox]) -> PersonId:
    """Assigns a detected face bbox to whichever panel contains its center.

    Falls back to the panel whose center is closest to the bbox's center if
    the point falls exactly on a boundary or (due to detection noise) just
    outside both panels.
    """
    cx, cy = bbox.center
    for person, panel in panels.items():
        if panel.contains_point(cx, cy):
            return person

    def dist(panel: BBox) -> float:
        pcx, pcy = panel.center
        return (pcx - cx) ** 2 + (pcy - cy) ** 2

    return min(panels, key=lambda p: dist(panels[p]))
