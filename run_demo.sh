#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

MODEL_PATH="$PROJECT_DIR/models/brain_tumor_efficientnetb0.keras"
CLASS_MAP_PATH="$PROJECT_DIR/models/class_indices.json"
VENV_DIR="$PROJECT_DIR/.venv"
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"

PYTHON_CMD=""
if [[ -x "$VENV_DIR/bin/python" ]]; then
  PYTHON_CMD="$VENV_DIR/bin/python"
else
  if command -v python3.11 >/dev/null 2>&1; then
    BASE_PYTHON="python3.11"
  elif command -v python3.10 >/dev/null 2>&1; then
    BASE_PYTHON="python3.10"
  elif command -v python3 >/dev/null 2>&1; then
    BASE_PYTHON="python3"
  else
    echo "Error: python3 is not installed." >&2
    exit 1
  fi

  echo "Creating virtual environment at $VENV_DIR using $BASE_PYTHON"
  "$BASE_PYTHON" -m venv "$VENV_DIR"
  PYTHON_CMD="$VENV_DIR/bin/python"
fi

PIP_CMD=("$PYTHON_CMD" -m pip)

echo "Upgrading pip..."
"${PIP_CMD[@]}" install --upgrade pip

echo "Installing dependencies from requirements.txt..."
"${PIP_CMD[@]}" install -r "$REQUIREMENTS_FILE"

if [[ ! -f "$MODEL_PATH" || ! -f "$CLASS_MAP_PATH" ]]; then
  echo "Model artifacts not found. Starting training..."
  "$PYTHON_CMD" "$PROJECT_DIR/train_model.py"
else
  echo "Found existing model artifacts. Skipping training."
fi

echo "Starting Flask app at http://127.0.0.1:5000"
exec "$PYTHON_CMD" "$PROJECT_DIR/app.py"
