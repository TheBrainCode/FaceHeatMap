"""Rotation matrix <-> yaw/pitch/roll (Tait-Bryan, Z-Y-X intrinsic order).

Convention: R = Rz(roll) @ Ry(yaw) @ Rx(pitch), with yaw about the vertical
(Y) axis, pitch about the horizontal (X) axis, roll about the depth (Z)
axis, all in radians unless a `_deg` suffix says otherwise.

MediaPipe's FaceLandmarker facial transformation matrix uses its own axis
convention (canonical face model -> camera space), so the sign of the
decomposed yaw/pitch relative to "the person's real-world left/right,
up/down" is a heuristic, not a verified ground truth -- see GazeConfig's
`flip_yaw` / `flip_pitch` escape hatch in config.py for adjusting it against
a real recording.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EulerAngles:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


def matrix_to_euler(rotation: np.ndarray) -> EulerAngles:
    """Decomposes a 3x3 (or 4x4, upper-left 3x3 used) rotation matrix."""
    r = np.asarray(rotation)[:3, :3]
    yaw = np.arctan2(-r[2, 0], np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2))
    pitch = np.arctan2(r[2, 1], r[2, 2])
    roll = np.arctan2(r[1, 0], r[0, 0])
    return EulerAngles(
        yaw_deg=float(np.degrees(yaw)),
        pitch_deg=float(np.degrees(pitch)),
        roll_deg=float(np.degrees(roll)),
    )


def euler_to_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Inverse of `matrix_to_euler`, used for testing the round trip."""
    yaw, pitch, roll = np.radians([yaw_deg, pitch_deg, roll_deg])

    cy, sy = np.cos(yaw), np.sin(yaw)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])

    cp, sp = np.cos(pitch), np.sin(pitch)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])

    cr, sr = np.cos(roll), np.sin(roll)
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])

    return rz @ ry @ rx
