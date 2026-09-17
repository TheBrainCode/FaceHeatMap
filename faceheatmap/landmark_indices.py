"""Resolves the MediaPipe FaceLandmarker landmark index groups we need.

MediaPipe exposes these as connection lists (edges between landmark
indices) rather than plain index sets, so we derive the sets once at import
time instead of hardcoding indices that could silently drift between model
versions.
"""
from __future__ import annotations

from functools import lru_cache


def _indices_from_connections(connections) -> list[int]:
    seen: set[int] = set()
    for c in connections:
        seen.add(c.start)
        seen.add(c.end)
    return sorted(seen)


def _ordered_loop_from_connections(connections) -> list[int]:
    """Walks the connection graph to recover polygon (contour) order.

    FACE_OVAL's connections form a single cycle; sorting the indices
    numerically (as `_indices_from_connections` does) would scramble that
    into an invalid, self-intersecting polygon.
    """
    adjacency: dict[int, list[int]] = {}
    for c in connections:
        adjacency.setdefault(c.start, []).append(c.end)
        adjacency.setdefault(c.end, []).append(c.start)

    start = next(iter(adjacency))
    ordered = [start]
    visited = {start}
    current = start
    prev = None
    while True:
        neighbors = [n for n in adjacency[current] if n != prev]
        nxt = next((n for n in neighbors if n not in visited), None)
        if nxt is None:
            break
        ordered.append(nxt)
        visited.add(nxt)
        prev, current = current, nxt
    return ordered


@lru_cache(maxsize=1)
def get_landmark_indices() -> dict[str, list[int]]:
    from mediapipe.tasks.python import vision

    flc = vision.FaceLandmarksConnections
    return {
        "face_oval": _ordered_loop_from_connections(flc.FACE_LANDMARKS_FACE_OVAL),
        "left_eye": _indices_from_connections(flc.FACE_LANDMARKS_LEFT_EYE),
        "right_eye": _indices_from_connections(flc.FACE_LANDMARKS_RIGHT_EYE),
        "left_iris": _indices_from_connections(flc.FACE_LANDMARKS_LEFT_IRIS),
        "right_iris": _indices_from_connections(flc.FACE_LANDMARKS_RIGHT_IRIS),
    }


# The iris boundary loops (4 points each) don't include the center point,
# since a center has no connection edges -- but MediaPipe's 478-landmark
# output always places it at a fixed index right after the boundary ring.
LEFT_IRIS_CENTER = 473
RIGHT_IRIS_CENTER = 468
NUM_LANDMARKS = 478
