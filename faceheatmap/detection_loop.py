"""Shared per-frame face-detection + identity-tracking loop.

Both the main pipeline and the calibration recorder need the same thing:
read a video frame by frame, run the landmarker, establish which half of
the frame is person1/person2 once both faces are first seen together, and
track that identity across frames. Factored out here so the two stay in
sync instead of drifting apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol

import cv2
import numpy as np

from faceheatmap.config import LayoutMode, PersonId
from faceheatmap.landmarker import FaceObservation
from faceheatmap.layout import BBox, get_panels, infer_layout_from_bboxes
from faceheatmap.tracker import IdentityTracker


class Landmarker(Protocol):
    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[FaceObservation]: ...


@dataclass
class DetectedFrame:
    frame_index: int
    timestamp_ms: int
    frame: np.ndarray
    person_obs: dict[PersonId, FaceObservation]  # empty until a layout has been established


class TwoPersonDetectionLoop:
    """Iterable over a video's frames, yielding one `DetectedFrame` per frame
    actually processed (i.e. after `frame_stride`/`max_frames`).

    `person_obs` is empty for any frame before both faces have first been
    detected together (no layout/identity can be assigned yet) or for a
    frame where detection dropped below 2 faces. `frame_w`/`frame_h`/`fps`
    are available immediately after construction; `layout_used`/`panels`
    are populated once a layout has been established during iteration.
    """

    def __init__(
        self,
        video_path: str,
        landmarker: Landmarker,
        layout: LayoutMode = LayoutMode.AUTO,
        frame_stride: int = 1,
        max_frames: int | None = None,
    ):
        self.video_path = video_path
        self._landmarker = landmarker
        self._layout_config = layout
        self._frame_stride = frame_stride
        self._max_frames = max_frames

        probe = cv2.VideoCapture(video_path)
        if not probe.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        self.fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        probe.release()

        self.layout_used: LayoutMode | None = None
        self.panels: dict[PersonId, BBox] | None = None
        self._tracker: IdentityTracker | None = None

    def __iter__(self) -> Iterator[DetectedFrame]:
        cap = cv2.VideoCapture(self.video_path)
        frame_idx = 0
        processed = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self._frame_stride != 0:
                    frame_idx += 1
                    continue
                if self._max_frames is not None and processed >= self._max_frames:
                    break

                timestamp_ms = int(frame_idx * 1000 / self.fps)
                observations = self._landmarker.detect(frame, timestamp_ms)

                if self.panels is None and len(observations) >= 2:
                    layout = self._layout_config
                    if layout is LayoutMode.AUTO:
                        layout = infer_layout_from_bboxes(observations[0].bbox, observations[1].bbox)
                    self.layout_used = layout
                    self.panels = get_panels(self.frame_w, self.frame_h, layout)
                    self._tracker = IdentityTracker(self.panels)

                person_obs: dict[PersonId, FaceObservation] = {}
                if self._tracker is not None:
                    assignment = self._tracker.assign([o.bbox for o in observations])
                    bbox_to_obs = {o.bbox: o for o in observations}
                    person_obs = {person: bbox_to_obs[bbox] for person, bbox in assignment.items()}

                yield DetectedFrame(frame_idx, timestamp_ms, frame, person_obs)

                frame_idx += 1
                processed += 1
        finally:
            cap.release()
