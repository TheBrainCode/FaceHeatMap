import numpy as np
import pytest

from faceheatmap.face_boundary import BoundaryStats, build_face_polygon, point_in_polygon
from faceheatmap.landmark_indices import NUM_LANDMARKS, get_landmark_indices

SQUARE = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)


def test_point_inside_square():
    assert point_in_polygon(SQUARE, 50, 50)


def test_point_outside_square():
    assert not point_in_polygon(SQUARE, 200, 200)


def test_point_on_boundary_counts_as_inside():
    assert point_in_polygon(SQUARE, 0, 50)


def test_build_face_polygon_uses_face_oval_indices_in_order():
    indices = get_landmark_indices()["face_oval"]
    landmarks = np.zeros((NUM_LANDMARKS, 3))
    for i, idx in enumerate(indices):
        landmarks[idx] = [float(i), float(i) * 2, 0.0]

    polygon = build_face_polygon(landmarks)
    assert polygon.shape == (len(indices), 2)
    for i in range(len(indices)):
        assert polygon[i, 0] == pytest.approx(float(i))
        assert polygon[i, 1] == pytest.approx(float(i) * 2)


def test_boundary_stats_accumulation():
    stats = BoundaryStats()
    stats.record(True)
    stats.record(True)
    stats.record(False)
    assert stats.total == 3
    assert stats.inside == 2
    assert stats.outside == 1
    assert stats.percent_inside == pytest.approx(66.666, abs=0.01)
    assert stats.percent_outside == pytest.approx(33.333, abs=0.01)


def test_boundary_stats_empty_is_zero_not_nan():
    stats = BoundaryStats()
    assert stats.percent_inside == 0.0
    assert stats.percent_outside == 0.0
    assert stats.to_dict()["total_frames"] == 0
