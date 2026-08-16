#!/usr/bin/env bash
# ModelScope Studio entry: install dependencies and launch the Gradio UI.
#
# ModelScope Studio usually runs `python app.py` directly and manages its own
# dependency install; this wrapper is a convenience for manual/VPS runs and
# documents the exact launch path.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "ERROR: Python is not available" >&2
  exit 1
fi

"$PYTHON_BIN" -m pip install --disable-pip-version-check -q -r requirements.txt
exec "$PYTHON_BIN" app.py
