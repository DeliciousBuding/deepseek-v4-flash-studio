#!/usr/bin/env bash
# Health check for the Gradio UI and its configured OpenAI-compatible backend.
set -euo pipefail

UI_PORT="${GRADIO_SERVER_PORT:-${PORT:-7860}}"
curl -fsS --max-time 10 "http://127.0.0.1:${UI_PORT}/" >/dev/null
echo "[health] gradio UI on :${UI_PORT} ok"

BACKEND_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000}"
BACKEND_BASE_URL="${BACKEND_BASE_URL%/}"
if [[ "$BACKEND_BASE_URL" == */v1 ]]; then
  BACKEND_MODELS_URL="${BACKEND_BASE_URL}/models"
else
  BACKEND_MODELS_URL="${BACKEND_BASE_URL}/v1/models"
fi

API_KEY="${VLLM_API_KEY:-}"
if [ -z "$API_KEY" ] && [ -n "${VLLM_API_KEY_FILE:-}" ]; then
  if [ ! -r "$VLLM_API_KEY_FILE" ]; then
    echo "ERROR: VLLM_API_KEY_FILE is not readable" >&2
    exit 1
  fi
  API_KEY="$(<"$VLLM_API_KEY_FILE")"
fi

CURL_AUTH_ARGS=()
if [ -n "$API_KEY" ]; then
  CURL_AUTH_ARGS=(-H "Authorization: Bearer ${API_KEY}")
fi

if curl -fsS --max-time 10 "${CURL_AUTH_ARGS[@]}" "$BACKEND_MODELS_URL" >/dev/null; then
  echo "[health] inference backend ok"
elif [ "${REQUIRE_BACKEND_HEALTH:-1}" = "1" ]; then
  echo "ERROR: inference backend is unavailable" >&2
  exit 1
else
  echo "[health] warning: inference backend is unavailable" >&2
fi
