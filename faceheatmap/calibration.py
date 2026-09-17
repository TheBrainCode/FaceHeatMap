"""Fits a per-person gaze-to-screen mapping from a 9-point calibration recording.

Protocol (both participants recorded simultaneously, same as a real call):
each person looks at 9 known points on their own screen, in a fixed order,
holding each for a fixed duration:

    1. center       2. top-left     3. top          4. top-right
    5. right        6. bottom-right 7. bottom       8. bottom-left
    9. left

For each point, we average the raw (head yaw/pitch, iris offset) signal
over the middle portion of that point's time window (the start/end are
trimmed to skip the saccade into/out of the point), then fit two small
linear regressions per person -- one mapping (head yaw, eye horizontal
offset) to the point's known normalized x, one mapping (head pitch, eye
vertical offset) to its known normalized y. That fitted mapping replaces
the generic assumed-FOV heuristic in `faceheatmap.gaze` for that person.

This calibration is specific to the physical setup it was recorded in
(camera position, screen size/distance, seating) -- see the README.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import numpy as np

from faceheatmap.config import LayoutMode, PersonId
from faceheatmap.detection_loop import Landmarker, TwoPersonDetectionLoop
from faceheatmap.gaze import RawGazeSignal, compute_raw_signal

# (name, target_x, target_y), normalized screen coords in [-1, 1] x [-1, 1].
# Order matters: it's how a calibration video's timeline gets sliced.
CALIBRATION_POINTS: list[tuple[str, float, float]] = [
    ("center", 0.0, 0.0),
    ("top-left", -1.0, -1.0),
    ("top", 0.0, -1.0),
    ("top-right", 1.0, -1.0),
    ("right", 1.0, 0.0),
    ("bottom-right", 1.0, 1.0),
    ("bottom", 0.0, 1.0),
    ("bottom-left", -1.0, 1.0),
    ("left", -1.0, 0.0),
]


@dataclass
class CalibrationConfig:
    seconds_per_point: float = 2.0
    trim_start_frac: float = 0.3  # skip this fraction at the start of each point's window (saccade transition)
    trim_end_frac: float = 0.1  # skip this fraction at the end (anticipating the next point)


@dataclass
class PersonCalibration:
    h_weights: tuple[float, float, float]  # norm_x = w0*head_yaw_deg + w1*eye_h + w2
    v_weights: tuple[float, float, float]  # norm_y = w0*head_pitch_deg + w1*eye_v + w2

    def apply(self, raw: RawGazeSignal) -> tuple[float, float]:
        wh0, wh1, wh2 = self.h_weights
        wv0, wv1, wv2 = self.v_weights
        norm_x = wh0 * raw.head_yaw_deg + wh1 * raw.eye_h + wh2
        norm_y = wv0 * raw.head_pitch_deg + wv1 * raw.eye_v + wv2
        return norm_x, norm_y

    def to_dict(self) -> dict:
        return {"h_weights": list(self.h_weights), "v_weights": list(self.v_weights)}

    @classmethod
    def from_dict(cls, data: dict) -> "PersonCalibration":
        return cls(h_weights=tuple(data["h_weights"]), v_weights=tuple(data["v_weights"]))


@dataclass
class CalibrationSample:
    point_name: str
    target_x: float
    target_y: float
    raw: RawGazeSignal


def fit_axis(feature_a: np.ndarray, feature_b: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    """Least-squares fit of target = w0*feature_a + w1*feature_b + w2 (bias)."""
    design = np.column_stack([feature_a, feature_b, np.ones_like(feature_a)])
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(solution[0]), float(solution[1]), float(solution[2])


def fit_person_calibration(samples: list[CalibrationSample]) -> PersonCalibration:
    if len(samples) < 3:
        raise ValueError(f"Need at least 3 calibration points to fit a mapping, got {len(samples)}")

    head_yaw = np.array([s.raw.head_yaw_deg for s in samples])
    eye_h = np.array([s.raw.eye_h for s in samples])
    target_x = np.array([s.target_x for s in samples])

    head_pitch = np.array([s.raw.head_pitch_deg for s in samples])
    eye_v = np.array([s.raw.eye_v for s in samples])
    target_y = np.array([s.target_y for s in samples])

    return PersonCalibration(
        h_weights=fit_axis(head_yaw, eye_h, target_x),
        v_weights=fit_axis(head_pitch, eye_v, target_y),
    )


def collect_calibration_samples(
    video_path: str,
    landmarker: Landmarker,
    calib_config: CalibrationConfig | None = None,
    layout: LayoutMode = LayoutMode.AUTO,
    frame_stride: int = 1,
) -> dict[PersonId, list[CalibrationSample]]:
    """Runs the calibration video through detection/tracking and buckets each
    frame's raw signal into the calibration point active at that timestamp.
    """
    calib_config = calib_config or CalibrationConfig()
    loop = TwoPersonDetectionLoop(video_path, landmarker, layout=layout, frame_stride=frame_stride)

    raw_by_person_point: dict[PersonId, dict[int, list[RawGazeSignal]]] = {
        PersonId.PERSON_1: {},
        PersonId.PERSON_2: {},
    }

    for detected in loop:
        t_s = detected.timestamp_ms / 1000.0
        point_index = int(t_s // calib_config.seconds_per_point)
        if point_index >= len(CALIBRATION_POINTS):
            continue

        progress = (t_s % calib_config.seconds_per_point) / calib_config.seconds_per_point
        if progress < calib_config.trim_start_frac or progress > (1.0 - calib_config.trim_end_frac):
            continue

        for person, obs in detected.person_obs.items():
            raw = compute_raw_signal(obs.landmarks_px, obs.transform_matrix)
            raw_by_person_point[person].setdefault(point_index, []).append(raw)

    samples: dict[PersonId, list[CalibrationSample]] = {PersonId.PERSON_1: [], PersonId.PERSON_2: []}
    for person, by_point in raw_by_person_point.items():
        for point_index, raws in sorted(by_point.items()):
            name, target_x, target_y = CALIBRATION_POINTS[point_index]
            averaged = RawGazeSignal(
                head_yaw_deg=float(np.mean([r.head_yaw_deg for r in raws])),
                head_pitch_deg=float(np.mean([r.head_pitch_deg for r in raws])),
                eye_h=float(np.mean([r.eye_h for r in raws])),
                eye_v=float(np.mean([r.eye_v for r in raws])),
            )
            samples[person].append(CalibrationSample(name, target_x, target_y, averaged))

    return samples


def run_calibration(
    video_path: str,
    landmarker: Landmarker,
    calib_config: CalibrationConfig | None = None,
    layout: LayoutMode = LayoutMode.AUTO,
    frame_stride: int = 1,
) -> dict[PersonId, PersonCalibration]:
    samples = collect_calibration_samples(video_path, landmarker, calib_config, layout, frame_stride)

    result: dict[PersonId, PersonCalibration] = {}
    for person, person_samples in samples.items():
        if len(person_samples) < 3:
            raise RuntimeError(
                f"Only captured {len(person_samples)} calibration point(s) for {person.value}; "
                f"need at least 3. Check the recording covers the full "
                f"{len(CALIBRATION_POINTS)}-point sequence with both faces visible throughout."
            )
        result[person] = fit_person_calibration(person_samples)
    return result


def save_calibration(calibration: dict[PersonId, PersonCalibration], path: str) -> None:
    data = {person.value: cal.to_dict() for person, cal in calibration.items()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_calibration(path: str) -> dict[PersonId, PersonCalibration]:
    with open(path) as f:
        data = json.load(f)
    return {PersonId(key): PersonCalibration.from_dict(value) for key, value in data.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="faceheatmap-calibrate",
        description="Fit a per-person gaze calibration from a 9-point calibration recording",
    )
    parser.add_argument("video", help="Path to the calibration recording")
    parser.add_argument("-o", "--output", default="calibration.json", help="Where to write the fitted calibration")
    parser.add_argument("--model-path", default="models/face_landmarker.task")
    parser.add_argument("--layout", choices=[m.value for m in LayoutMode], default=LayoutMode.AUTO.value)
    parser.add_argument("--seconds-per-point", type=float, default=2.0, help="How long each point was held for while recording")
    parser.add_argument("--frame-stride", type=int, default=1)
    args = parser.parse_args(argv)

    from faceheatmap.landmarker import FaceLandmarkerWrapper

    landmarker = FaceLandmarkerWrapper(model_path=args.model_path, num_faces=2)
    try:
        calibration = run_calibration(
            args.video,
            landmarker,
            CalibrationConfig(seconds_per_point=args.seconds_per_point),
            layout=LayoutMode(args.layout),
            frame_stride=args.frame_stride,
        )
    finally:
        landmarker.close()

    save_calibration(calibration, args.output)
    print(f"Calibration written to {args.output}")
    for person, cal in calibration.items():
        print(f"  {person.value}: h_weights={tuple(round(w, 4) for w in cal.h_weights)} v_weights={tuple(round(w, 4) for w in cal.v_weights)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
