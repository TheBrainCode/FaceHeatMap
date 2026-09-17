import numpy as np
import pytest

from faceheatmap.rotation import euler_to_matrix, matrix_to_euler


@pytest.mark.parametrize(
    "yaw,pitch,roll",
    [
        (0, 0, 0),
        (15, -10, 5),
        (-30, 20, -8),
        (45, 45, 0),
        (-60, -15, 10),
        (5, 0, 0),
        (0, 33, 0),
        (0, 0, -25),
    ],
)
def test_round_trip(yaw, pitch, roll):
    r = euler_to_matrix(yaw, pitch, roll)
    angles = matrix_to_euler(r)
    assert angles.yaw_deg == pytest.approx(yaw, abs=1e-6)
    assert angles.pitch_deg == pytest.approx(pitch, abs=1e-6)
    assert angles.roll_deg == pytest.approx(roll, abs=1e-6)


def test_identity_matrix_is_zero_angles():
    angles = matrix_to_euler(np.eye(3))
    assert angles.yaw_deg == pytest.approx(0.0)
    assert angles.pitch_deg == pytest.approx(0.0)
    assert angles.roll_deg == pytest.approx(0.0)


def test_accepts_4x4_homogeneous_matrix():
    r3 = euler_to_matrix(10, 5, 0)
    r4 = np.eye(4)
    r4[:3, :3] = r3
    r4[:3, 3] = [1.0, 2.0, 3.0]
    angles = matrix_to_euler(r4)
    assert angles.yaw_deg == pytest.approx(10, abs=1e-6)
    assert angles.pitch_deg == pytest.approx(5, abs=1e-6)
