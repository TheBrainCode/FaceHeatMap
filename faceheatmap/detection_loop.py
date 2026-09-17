"""Shared per-frame face-detection loops.

Both the main pipeline and the calibration recorder need the same thing:
read a video frame by frame, run the landmarker, and produce a stable
person1/person2 identity per frame. Factored out here so the two stay in
sync instead of drifting apart.

Two strategies are provided:

- `TwoPersonDetectionLoop` runs one detection pass over the whole frame
  (num_faces=2) and uses `IdentityTracker` to assign identity. This is
  needed for `--layout auto` (it has to see both faces together at least
  once to infer the split direction), but MediaPipe's detector can fail to
  return a second face on the same call when the two faces differ a lot in
  scale within the frame (e.g. one person much closer to their camera than
  the other) -- confirmed empirically, not just a theoretical concern.
- `SplitPanelDetectionLoop` sidesteps that entirely: once the layout is
  known (must be given explicitly, not auto), it crops each panel and runs
  independent single-face detection per panel, so the two detections never
  compete against each other. Requires two landmarker instances (MediaPipe's
  VIDEO mode needs strictly increasing timestamps per session, so the two
  panels can't share one landmarker across frames).
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


def _offset_observation(obs: FaceObservation, dx: float, dy: float) -> FaceObservation:
    """Translates a FaceObservation detected in a panel crop back into
    full-frame pixel coordinates. The transformation matrix is left as-is:
    it's a 3D head-pose rotation (the only part we use downstream), not
    pixel coordinates, so it isn't affected by where the crop sat in the
    original frame.
    """
    bbox = BBox(obs.bbox.x0 + dx, obs.bbox.y0 + dy, obs.bbox.x1 + dx, obs.bbox.y1 + dy)
    landmarks_px = obs.landmarks_px.copy()
    landmarks_px[:, 0] += dx
    landmarks_px[:, 1] += dy
    return FaceObservation(bbox=bbox, landmarks_px=landmarks_px, transform_matrix=obs.transform_matrix)


class SplitPanelDetectionLoop:
    """Iterable over a video's frames, cropping each panel and running
    independent single-face detection on each -- see the module docstring
    for why this exists instead of always using `TwoPersonDetectionLoop`.

    Unlike `TwoPersonDetectionLoop`, `layout` must be resolved up front
    (not AUTO): panel boundaries have to be known before any detection can
    happen, since detection happens per panel crop. No identity tracker is
    needed either -- which panel a detection came from *is* the identity.
    """

    def __init__(
        self,
        video_path: str,
        landmarker_person1: Landmarker,
        landmarker_person2: Landmarker,
        layout: LayoutMode,
        frame_stride: int = 1,
        max_frames: int | None = None,
    ):
        if layout is LayoutMode.AUTO:
            raise ValueError("SplitPanelDetectionLoop requires an explicit layout (side_by_side/top_bottom), not auto -- panel boundaries must be known before any detection happens.")

        self.video_path = video_path
        self._landmarkers = {PersonId.PERSON_1: landmarker_person1, PersonId.PERSON_2: landmarker_person2}
        self._frame_stride = frame_stride
        self._max_frames = max_frames

        probe = cv2.VideoCapture(video_path)
        if not probe.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        self.fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        probe.release()

        self.layout_used = layout
        self.panels: dict[PersonId, BBox] = get_panels(self.frame_w, self.frame_h, layout)

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

                person_obs: dict[PersonId, FaceObservation] = {}
                for person, panel in self.panels.items():
                    x0, y0, x1, y1 = int(panel.x0), int(panel.y0), int(panel.x1), int(panel.y1)
                    crop = frame[y0:y1, x0:x1]
                    if crop.size == 0:
                        continue
                    detections = self._landmarkers[person].detect(crop, timestamp_ms)
                    if detections:
                        person_obs[person] = _offset_observation(detections[0], x0, y0)

                yield DetectedFrame(frame_idx, timestamp_ms, frame, person_obs)

                frame_idx += 1
                processed += 1
        finally:
            cap.release()
