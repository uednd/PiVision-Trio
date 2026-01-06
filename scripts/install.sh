#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$PROJECT_DIR/models"
PROXY="https://gh-proxy.com/"
MEDIAPIPE_VERSION="0.10.14"
VENV_DIR=".venv"

log() {
    echo "$@"
}

warn() {
    echo "[Warn] $*" >&2
}

fail() {
    echo "[Error] $*" >&2
    exit 1
}

download_file() {
    local url="$1"
    local destination="$2"

    if command -v curl &> /dev/null; then
        curl -fL --retry 3 --retry-delay 1 --retry-connrefused -o "$destination" "$url"
        return 0
    fi

    if command -v wget &> /dev/null; then
        wget -O "$destination" "$url"
        return 0
    fi

    fail "Missing downloader: install curl or wget."
}

download_with_fallback() {
    local primary_url="$1"
    local fallback_url="$2"
    local destination="$3"

    if download_file "$primary_url" "$destination"; then
        return 0
    fi

    warn "Primary download failed, retrying with direct URL..."
    download_file "$fallback_url" "$destination"
}

log "========================================"
log "PiVision-Trio"
log "========================================"
log ""

PYTHON_CMD="python3.11"

PYTHON_VERSION=$("$PYTHON_CMD" --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
log "[Check] Python: $PYTHON_CMD (version $PYTHON_VERSION)"

ARCH=$(uname -m)
log "[Check] Architecture: $ARCH"
log ""

log "[Step 1/5] Install system dependencies..."
if command -v sudo &> /dev/null && sudo -n true 2>/dev/null && command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y python3-pip python3-venv libopenblas-dev libopencv-dev python3-opencv curl
else
    warn "Skipping system dependencies (no sudo or apt-get)."
fi

log ""
log "[Step 2/5] Download face detection models..."
mkdir -p "$MODELS_DIR"

MODEL_URL="https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
MODEL_URL_PROXY="${PROXY}${MODEL_URL}"
MODEL_FILE="$MODELS_DIR/opencv_face_detector.caffemodel"
CONFIG_URL="https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt"
CONFIG_URL_PROXY="${PROXY}${CONFIG_URL}"
CONFIG_FILE="$MODELS_DIR/opencv_face_detector.prototxt"

if [ ! -f "$MODEL_FILE" ]; then
    download_with_fallback "$MODEL_URL_PROXY" "$MODEL_URL" "$MODEL_FILE"
    log "[Done] opencv_face_detector.caffemodel"
else
    log "[Skip] Model weights already exist"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    download_with_fallback "$CONFIG_URL_PROXY" "$CONFIG_URL" "$CONFIG_FILE"
    log "[Done] opencv_face_detector.prototxt"
else
    log "[Skip] Model config already exists"
fi

log ""
log "[Step 3/5] Create virtual environment..."
cd "$PROJECT_DIR"

VENV_PYTHON="$PROJECT_DIR/$VENV_DIR/bin/python"

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_CMD" -m venv "$VENV_DIR" --system-site-packages
    log "[Done] Virtual environment created (Python $PYTHON_VERSION)"
else
    log "[Skip] Virtual environment already exists"
fi

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    fail "Virtual environment activation script not found: $VENV_DIR/bin/activate"
fi
if [ ! -x "$VENV_PYTHON" ]; then
    fail "Virtual environment Python missing: $VENV_PYTHON"
fi

source "$VENV_DIR/bin/activate"

log ""
log "[Step 4/5] Install Python dependencies..."

"$VENV_PYTHON" -m pip install --upgrade pip
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    "$VENV_PYTHON" -m pip install -r "$PROJECT_DIR/requirements.txt"
else
    warn "requirements.txt not found, installing minimal dependencies."
    "$VENV_PYTHON" -m pip install "numpy>=1.24.0" "opencv-python>=4.8.0"
fi

log ""
log "[Install] MediaPipe..."

"$VENV_PYTHON" -m pip install "mediapipe>=${MEDIAPIPE_VERSION}"
log "[Success] MediaPipe installed"

log ""
log "[Step 5/5] Verify installation..."

"$VENV_PYTHON" << 'EOF'
import sys
print(f"Python: {sys.version}")

try:
    import cv2
    print(f"OpenCV: {cv2.__version__} ✓")
except ImportError:
    print("OpenCV: Not installed ✗")

try:
    import numpy as np
    print(f"NumPy: {np.__version__} ✓")
except ImportError:
    print("NumPy: Not installed ✗")

try:
    import mediapipe as mp
    print(f"MediaPipe: {mp.__version__} ✓")
except ImportError:
    print("MediaPipe: Not installed ✗")

print("")
print("Installation complete!")
EOF

log ""
log "========================================"
log "Run PiVision-Trio:"
log "  cd $PROJECT_DIR"
log "  source $VENV_DIR/bin/activate"
log "  $VENV_PYTHON src/main.py"
log "========================================"
