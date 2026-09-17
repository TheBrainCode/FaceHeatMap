"""Keeps a stable person1/person2 identity across frames.

MediaPipe's FaceLandmarker doesn't track identity across frames by itself
(it just returns up to N faces per call, in no guaranteed stable order), so
we match each frame's detections against the previous frame's bounding
boxes by IoU, and fall back to panel position when a match is ambiguous or
one participant briefly drops out of detection.
"""
from __future__ import annotations

from faceheatmap.config import PersonId
from faceheatmap.layout import BBox, assign_panel, iou


class IdentityTracker:
    def __init__(self, panels: dict[PersonId, BBox], iou_threshold: float = 0.15):
        self._panels = panels
        self._iou_threshold = iou_threshold
        self._last_bbox: dict[PersonId, BBox] = {}

    def assign(self, detections: list[BBox]) -> dict[PersonId, BBox]:
        """Maps this frame's detections to stable person identities.

        Returns a dict with 0, 1, or 2 entries (one per detection that
        could be assigned a person identity; two detections can never map
        to the same person).
        """
        if not detections:
            return {}

        if len(detections) == 1:
            person = self._best_match(detections[0])
            self._last_bbox[person] = detections[0]
            return {person: detections[0]}

        # Exactly two detections: try both pairings against known state and
        # keep whichever assignment is internally consistent (no person
        # claimed twice) and best matches prior positions.
        a, b = detections[0], detections[1]
        person_a = self._best_match(a)
        person_b = self._best_match(b)
        if person_a == person_b:
            # Conflict: resolve by comparing IoU/panel-fit scores directly.
            person_a, person_b = self._resolve_conflict(a, b)

        self._last_bbox[person_a] = a
        self._last_bbox[person_b] = b
        return {person_a: a, person_b: b}

    def _best_match(self, bbox: BBox) -> PersonId:
        best_person, best_iou = None, 0.0
        for person, prev in self._last_bbox.items():
            score = iou(bbox, prev)
            if score > best_iou:
                best_person, best_iou = person, score
        if best_person is not None and best_iou >= self._iou_threshold:
            return best_person
        return assign_panel(bbox, self._panels)

    def _resolve_conflict(self, a: BBox, b: BBox) -> tuple[PersonId, PersonId]:
        p1, p2 = PersonId.PERSON_1, PersonId.PERSON_2

        def score(bbox: BBox, person: PersonId) -> float:
            prev = self._last_bbox.get(person)
            if prev is not None:
                return iou(bbox, prev)
            return 1.0 if assign_panel(bbox, self._panels) is person else 0.0

        forward = score(a, p1) + score(b, p2)
        swapped = score(a, p2) + score(b, p1)
        return (p1, p2) if forward >= swapped else (p2, p1)
