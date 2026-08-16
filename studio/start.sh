#!/usr/bin/env bash
# ModelScope 创空间 entry — install deps and launch the Gradio UI.
#
# ModelScope Studio usually runs `python app.py` directly and manages its own
# dependency install; this wrapper is a convenience for manual/VPS runs and
# documents the exact launch path.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ ! -x "$(command -v python3)" ] && [ -x "$(command -v python)" ]; then
  alias python3=python
fi

python3 -m pip install --upgrade -q -r requirements.txt
exec python3 app.py
