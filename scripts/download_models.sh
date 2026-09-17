#!/usr/bin/env bash
# Downloads the MediaPipe FaceLandmarker model bundle used by faceheatmap.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"
mkdir -p "$MODELS_DIR"

URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
DEST="$MODELS_DIR/face_landmarker.task"

echo "Downloading FaceLandmarker model to $DEST"
curl -fSL "$URL" -o "$DEST"
echo "Done."
