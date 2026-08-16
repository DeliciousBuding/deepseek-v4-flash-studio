# deepseek-v4-flash-studio

<div align="center">

**"MI308X DeepSeek Lab" — the presentation layer for DeepSeek-V4-Flash-0731 on a single AMD Instinct MI308X.**

A thin Gradio UI + optional LiteLLM/Open WebUI gateway over the OpenAI-compatible vLLM endpoint. Long-context probe · reasoning display · TTFT/throughput panel.

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-MI308X%20%7C%20192GB-ED1C24)]()
[![ROCm](https://img.shields.io/badge/ROCm-7.2-red)]()

</div>

This repository is the **application shell**, not the inference kernel. The
actual serving stack — vLLM + ROCm + DeepSeek-V4-Flash + MI308X tuning — lives
in the sibling repo [`deepseek-v4-flash-mi308x`](https://github.com/DeliciousBuding/deepseek-v4-flash-mi308x).
This project consumes its OpenAI-compatible API and turns it into an interactive
"lab" that showcases long context, reasoning, and measured performance.

## Architecture

```text
browser
   │
   ▼
┌──────────────┐   lite mode    ┌───────────────────────────┐
│  Gradio UI   │ ─────────────▶ │  OpenAI-compatible /v1    │
│  (app.py)    │                │  vLLM on ROCm             │
└──────────────┘                │  DeepSeek-V4-Flash-0731   │
   │  (full mode)               │  AMD Instinct MI308X      │
   ▼                            └───────────────────────────┘
Open WebUI ──▶ LiteLLM ──▶ same /v1 endpoint
```

- **Lite mode** (default): `python app.py` — a self-contained Gradio app that
  streams chat, folds reasoning, and runs 32K–512K long-context probes with
  measured TTFT / output tok/s / cached-token accounting.
- **Full mode** (optional): `docker compose up` — Open WebUI + LiteLLM in front
  of the same backend, for a richer multi-user chat experience.

The vLLM backend is **not** containerized here and **not** re-implemented; it is
the native ROCm stack from the sibling repo.

## Quick start (lite mode)

```bash
cp .env.example .env          # then edit VLLM_BASE_URL / VLLM_API_KEY
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py                 # http://127.0.0.1:7860
```

With the backend already running (see the sibling repo):

```bash
VLLM_BASE_URL=http://127.0.0.1:8000 MODEL_NAME=deepseek-v4-flash python app.py
```

## Full mode (Open WebUI + LiteLLM)

```bash
VLLM_BASE_URL=http://host.docker.internal:8000 docker compose up
# Open WebUI: http://127.0.0.1:3000   LiteLLM: http://127.0.0.1:4000
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `VLLM_BASE_URL` | `http://127.0.0.1:8000` | OpenAI-compatible backend base URL |
| `VLLM_API_KEY` | *(empty)* | Bearer key when the backend is authenticated |
| `MODEL_NAME` | `deepseek-v4-flash` | Model id to request |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Gradio bind address |
| `GRADIO_SERVER_PORT` | `7860` | Gradio bind port |

## Repository layout

```text
app.py                  Gradio UI ("MI308X DeepSeek Lab") — ModelScope Studio entry
requirements.txt        runtime deps (gradio + httpx)
litellm/config.yaml     optional full-mode gateway config (model aliases)
open-webui/             optional full-mode Open WebUI notes
studio/                 ModelScope Studio packaging (start/healthcheck/submission notes)
docs/                   ARCHITECTURE.md / DEPLOYMENT.md
docker-compose.yml      optional full-mode stack (Open WebUI + LiteLLM)
```

## ModelScope 创空间 (Studio)

The app can be deployed to ModelScope Studio: select a MI308X GPU instance with
a ROCm image, entry file `app.py`, dependencies from `requirements.txt`. See
[`studio/modelscope.md`](studio/modelscope.md) for the exact steps and
environment variables.

## Boundary

This repo intentionally does **not** own: the vLLM/ROCm serving stack, model
weights, patch overlays, GPU tuning tables, SSH/bootstrap, or secret persistence.
Those belong to [`deepseek-v4-flash-mi308x`](https://github.com/DeliciousBuding/deepseek-v4-flash-mi308x)
(public) and a private infrastructure layer.
