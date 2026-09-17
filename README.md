# faceheatmap

Estimates where each participant in a two-person Zoom-style recording is
looking, and reports it as:

- A **gaze heatmap** for each person, both over the whole recording frame
  and cropped to the *other* participant's video panel (i.e. "where person 1
  was looking, within person 2's tile").
- A **face-boundary attention breakdown**: for each person, what percentage
  of frames their estimated gaze point fell *inside* vs *outside* the other
  participant's face contour.

## How it works

1. [MediaPipe FaceLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
   detects up to two faces per frame, returning 478 3D landmarks (including
   iris landmarks) and a facial transformation matrix (head pose) for each.
2. Faces are assigned to a stable `person1`/`person2` identity across frames
   using an IoU-based tracker, seeded from which half of the frame
   (left/right or top/bottom) each face is in.
3. **Gaze estimation** (`faceheatmap/gaze.py`) combines head pose
   (yaw/pitch, from the transformation matrix) with an iris-offset
   correction (how far the iris sits from the eye's center, normalized by
   eye width/height), then projects that angle onto an assumed screen field
   of view to get a point on the recording frame.
4. Gaze points are accumulated into a 2D histogram, Gaussian-blurred into a
   heatmap, and rendered as a colored overlay.
5. A face-boundary polygon (MediaPipe's `FACE_OVAL` landmark ring) is
   computed per frame for each person, and the counterpart's gaze point is
   tested against it for the attention-breakdown stats.

## Important limitation: this is not calibrated eye tracking

A recording alone gives no information about the viewer's screen size,
distance from the camera, or the camera's offset from the screen — so
there's no way to derive a true gaze-to-screen calibration. The gaze
mapping here is a documented **heuristic**: head pose + eye-offset, scaled
by an assumed field of view (`--fov-h-deg` / `--fov-v-deg`, default 40°x24°,
tunable via CLI flags). Treat outputs as an approximate, directional signal
for aggregate attention patterns ("did they spend more of the call looking
toward the other person's face"), not as pixel-accurate gaze tracking.

The sign of the estimated yaw/pitch (which direction on screen "positive"
head/eye rotation maps to) depends on MediaPipe's internal axis convention
and is not independently verified against ground-truth video in this
project. If a heatmap looks mirrored for your footage, use `--flip-yaw`
and/or `--flip-pitch` to correct it.

You can substantially improve on this heuristic by recording a short
per-setup **calibration** clip and fitting a real mapping instead — see
below.

## Calibration (recommended)

Record both participants, in the same physical setup (same camera, same
monitor size/distance, same seating) they'll use for the real call, looking
at 9 known points on their own screen in this fixed order, holding each for
**2 seconds**:

```
1. center        2. top-left     3. top          4. top-right
5. right         6. bottom-right 7. bottom       8. bottom-left
9. left
```

Both people look at their own corresponding point at the same time (e.g.
one person reads the sequence out loud, or you use a shared 2-second
timer/metronome) — a single ~18-second video covers both people's
calibration in one pass. Move directly from point to point; you don't need
to pause between them (the first/last fraction of each 2-second window is
automatically trimmed to skip the eye movement between points).

Fit the calibration:

```bash
faceheatmap-calibrate calibration_recording.mp4 -o calibration.json
```

Then use it for the real call recording:

```bash
faceheatmap real_call.mp4 -o output/ --calibration calibration.json
```

This replaces the generic assumed-FOV heuristic with a small linear
regression, fit per person, from their actual observed head-pose/iris
signal at each of the 9 known points to that point's real screen position.
It's still not lab-grade eye tracking, but it corrects for individual
differences (eye shape, camera offset, screen distance) that the generic
heuristic can't account for, and it also fixes the yaw/pitch sign-direction
guess automatically (no more `--flip-yaw` guessing).

**Caveats:**
- The calibration is tied to the *physical setup*, not just the person. If
  either person's camera, monitor, or seating differs between the
  calibration clip and the real call, accuracy degrades back toward the
  generic heuristic.
- If `--seconds-per-point` differed when you recorded (e.g. you used 3s
  holds instead of 2s), pass the same value to `faceheatmap-calibrate`
  with `--seconds-per-point`.
- `--layout` should match between the calibration recording and the real
  call (both default to `auto`, which works fine as long as the panel
  arrangement is consistent).

## Assumptions

- The recording shows a standard **two-person Zoom gallery layout**: each
  participant occupies one half of the frame, either side-by-side or
  top-bottom, for the whole recording. `--layout auto` (default)
  auto-detects which, from the first frame both faces are found in;
  `--layout side_by_side` / `--layout top_bottom` force it.
- Participants don't swap panels mid-recording.
- Both faces are visible (unobstructed, reasonably front-facing) for at
  least one frame, to establish the layout.

## Setup

Requires Python 3.9-3.12.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
scripts/download_models.sh   # downloads MediaPipe's face_landmarker.task (~4MB)
```

`mediapipe` is pinned to `<1.0` because MediaPipe dropped Python 3.9 wheels
starting at 1.0.0; the 0.10.x series still has the full Tasks API
(`FaceLandmarker`) this project uses, so nothing is lost by staying on it.

On Linux, `mediapipe`'s Tasks API requires a GPU/EGL-capable OpenGL stack
even when running on CPU. On a minimal install you may need:

```bash
apt-get install -y libegl1 libgl1 libgles2
```

(Not needed on macOS or Windows.)

## Usage

```bash
faceheatmap path/to/recording.mp4 -o output/
```

Outputs written to `output/`:

- `person1_gaze_full_frame.png`, `person2_gaze_full_frame.png` — heatmap
  over the whole recording frame.
- `person1_gaze_on_person2_panel.png` — where person 1 looked, restricted
  to and overlaid on person 2's panel (and the symmetric file for person 2).
- `summary.json` — frame counts, layout detected, and the face-boundary
  inside/outside percentages for both people.
- `gaze_log.csv` — per-frame yaw/pitch/mapped screen point for both people
  (unless `--no-csv`).
- `debug_annotated.mp4` — original video with face-boundary polygons and
  gaze markers drawn on, if `--debug-video` is passed.

Useful flags (see `faceheatmap --help` for the full list):

- `--layout {side_by_side,top_bottom,auto}`
- `--fov-h-deg`, `--fov-v-deg` — assumed screen field of view.
- `--eye-gain-h-deg`, `--eye-gain-v-deg` — how much weight iris offset (vs.
  head pose) gets in the gaze estimate.
- `--flip-yaw`, `--flip-pitch` — correct mirrored output.
- `--frame-stride N` — process every Nth frame, for faster runs on long
  recordings.
- `--debug-video` — write an annotated video for visually sanity-checking
  the detections and gaze mapping on your footage.

### Viewing the results

The PNGs in `output/` are already viewable images — open them directly
(`open output/person1_gaze_on_person2_panel.png` on macOS, or just browse
to the folder in Finder/Explorer).

For a combined view, generate an HTML report that embeds all four
heatmaps, the summary stats, and a gaze-over-time chart on one page:

```bash
faceheatmap-report output/
open output/report.html
```

## Development

```bash
pip install -r requirements.txt
pip install -e .
python -m pytest -q
```

Tests are split into:
- Pure-logic unit tests (rotation math, layout splitting, tracker, gaze
  math, face-boundary containment, heatmap accumulation, calibration
  fitting) — no ML model needed.
- `tests/test_pipeline_integration.py`, `tests/test_pipeline_calibration_integration.py`
  — exercise the full pipeline end-to-end against a fake landmarker
  (deterministic, no model download).
- `tests/test_real_landmarker_smoke.py` — runs the real MediaPipe model
  (including the calibration flow) against a synthetic two-panel frame;
  skipped automatically if `models/face_landmarker.task` hasn't been
  downloaded.
