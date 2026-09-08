#!/usr/bin/env bash
# ModelScope Studio entry: install dependencies and launch the Gradio UI.
#
# ModelScope Studio usually runs `python app.py` directly and manages its own
# dependency install; this wrapper is a convenience for manual/VPS runs and
# documents the exact launch path.
#
# This deliberately stays on pip + requirements.txt: the platform runtime is not
# guaranteed to provide uv. Local development and CI use uv (pyproject.toml +
# uv.lock), and CI fails if requirements.txt drifts from pyproject.toml.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "ERROR: Python is not available" >&2
  exit 1
fi

if [ -r "$PROJECT_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$PROJECT_ROOT/.env"
  set +a
fi

"$PYTHON_BIN" -m pip install --disable-pip-version-check -q -r requirements.txt
exec "$PYTHON_BIN" app.py
