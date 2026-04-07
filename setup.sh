#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON:-python3}"

echo
echo "Apollo setup"
echo "============"
echo

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python 3 is required."
    echo "Set PYTHON=/path/to/python3 if python3 is not on your PATH."
    exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required to install portaudio and ffmpeg."
    echo "Install Homebrew first: https://brew.sh"
    exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

echo "Installing Homebrew packages..."
brew install portaudio ffmpeg

echo "Installing Python packages into ${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/pip" install -r "${ROOT_DIR}/requirements-dev.txt"

echo
echo "Optional extras:"
echo "  ${VENV_DIR}/bin/pip install openai-whisper"
echo "  ${VENV_DIR}/bin/pip install opencv-python mediapipe"
echo
echo "Next steps:"
echo "  1. Export DEEPGRAM_API_KEY and GEMINI_API_KEY"
echo "  2. Grant Microphone and Accessibility access to your terminal app"
echo "  3. Run ${VENV_DIR}/bin/python -m apollo"
echo
