# deepseek-v4-flash-studio

<div align="center">

**"MI308X DeepSeek Lab" — the presentation layer for DeepSeek-V4-Flash-0731 on a single AMD Instinct MI308X.**

A thin Gradio UI + optional LiteLLM/Open WebUI gateway over the OpenAI-compatible vLLM endpoint. Long-context probe · reasoning display · client-observed TTFT/throughput panel.

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
Lite mode:
browser -> Gradio -> OpenAI-compatible /v1 -> vLLM/ROCm -> MI308X

Full mode:
browser -> Open WebUI -> LiteLLM -> same OpenAI-compatible backend
```

- **Lite mode** (default): `python app.py` — a self-contained Gradio app that
  streams chat, folds reasoning, and runs bounded 32K–475K long-context probes
  with client-observed TTFT / output tok/s / cached-token accounting.
- **Full mode** (optional): `docker compose up` — Open WebUI + LiteLLM in front
  of the same backend, for a richer multi-user chat experience.

The vLLM backend is **not** containerized here and **not** re-implemented; it is
the native ROCm stack from the sibling repo.

## Quick start (lite mode)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then edit VLLM_BASE_URL / VLLM_API_KEY
set -a && source .env && set +a
python app.py                 # http://127.0.0.1:7860
```

With the backend already running (see the sibling repo):

```bash
VLLM_BASE_URL=http://127.0.0.1:8000 MODEL_NAME=deepseek-v4-flash python app.py
```

## Full mode (Open WebUI + LiteLLM)

```bash
cp .env.example .env
# Set a long random LITELLM_MASTER_KEY and the reachable VLLM_BASE_URL in .env.
docker compose up -d
# Open WebUI: http://127.0.0.1:3000   LiteLLM: http://127.0.0.1:4000
```

Compose binds both ports to loopback, keeps Open WebUI authentication enabled,
and persists its database in a named volume. Put an authenticated reverse proxy
in front when exposing full mode beyond the local host.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `VLLM_BASE_URL` | `http://127.0.0.1:8000` | OpenAI-compatible backend base URL |
| `VLLM_API_KEY` | *(empty)* | Bearer key when the backend is authenticated |
| `VLLM_API_KEY_FILE` | *(empty)* | Read the bearer key from a mounted secret file |
| `MODEL_NAME` | `deepseek-v4-flash` | Model id to request |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Gradio bind address |
| `GRADIO_SERVER_PORT` | `7860` | Gradio bind port |
| `GRADIO_ROOT_PATH` | *(empty)* | Reverse-proxy URL prefix |
| `MAX_CONTEXT_TOKENS` | `524288` | Backend context ceiling used to bound probes |
| `CONTEXT_PROBE_SAFETY_TOKENS` | `4096` | Space reserved for output and chat framing |
| `INFERENCE_CONCURRENCY_LIMIT` | `1` | Shared Gradio chat/probe concurrency |
| `QUEUE_MAX_SIZE` | `32` | Maximum queued UI requests |

## Repository layout

```text
app.py                  Minimal application entry point
deepseek_lab/           configuration, backend client, probes, and Gradio UI
requirements.txt        runtime deps (gradio + httpx)
litellm/config.yaml     optional full-mode gateway config (model aliases)
open-webui/             optional full-mode Open WebUI notes
studio/                 platform-neutral start/health scripts + Studio guide
docs/                   ARCHITECTURE.md / DEPLOYMENT.md
docker-compose.yml      optional full-mode stack (Open WebUI + LiteLLM)
tests/                  focused protocol, history, and probe safety tests
```

## ModelScope 创空间 (Studio)

The UI can be deployed to ModelScope Studio with entry file `app.py` and
dependencies from `requirements.txt`. The UI itself does not load weights or
consume a GPU: it needs an already-running backend. Use a CPU application when
the backend is remote, or start the sibling serving recipe first when both
processes are intentionally co-located on an MI308X instance. See
[`studio/modelscope.md`](studio/modelscope.md) for the exact topology and
environment variables.

## Boundary

This repo intentionally does **not** own: the vLLM/ROCm serving stack, model
weights, patch overlays, GPU tuning tables, SSH/bootstrap, or secret persistence.
Those belong to [`deepseek-v4-flash-mi308x`](https://github.com/DeliciousBuding/deepseek-v4-flash-mi308x)
(public) and a private infrastructure layer.
