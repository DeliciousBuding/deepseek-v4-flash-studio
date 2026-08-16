#!/usr/bin/env bash
# Health check: verify the Gradio UI responds and (optionally) the vLLM
# backend /health is green. Non-zero exit = unhealthy.
set -euo pipefail

UI_PORT="${GRADIO_SERVER_PORT:-7860}"
curl -fsS --max-time 10 "http://127.0.0.1:${UI_PORT}/" >/dev/null
echo "[health] gradio UI on :${UI_PORT} ok"

BACKEND="${VLLM_BASE_URL:-http://127.0.0.1:8000}"
if curl -fsS --max-time 10 "${BACKEND%/}/health" >/dev/null 2>&1; then
  echo "[health] backend ${BACKEND} ok"
fi
