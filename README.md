# deepseek-v4-flash-studio

<div align="center">

**"MI308X DeepSeek Lab" — the presentation layer for DeepSeek-V4-Flash-0731 on a single AMD Instinct MI308X.**

A thin Gradio UI and LiteLLM gateway over the OpenAI-compatible vLLM endpoint. Long-context probe · reasoning display · client-observed TTFT/throughput panel.

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-MI308X%20%7C%20192GB-ED1C24)](https://www.amd.com/en/products/accelerators/instinct)
[![ROCm](https://img.shields.io/badge/ROCm-7.2-red)](https://rocm.docs.amd.com/)

</div>

This repository is the **application shell**, not the inference kernel. The
actual serving stack — vLLM + ROCm + DeepSeek-V4-Flash + MI308X tuning — lives
in the sibling repo [`deepseek-v4-flash-mi308x`](https://github.com/DeliciousBuding/deepseek-v4-flash-mi308x).
This project consumes its OpenAI-compatible API and turns it into an interactive
"lab" that showcases long context, reasoning, and measured performance.

## Architecture

```text
browser -> Gradio -> LiteLLM /v1 -> vLLM/ROCm -> MI308X
```

`python app.py` runs the self-contained Gradio application. It streams chat,
folds reasoning, and runs bounded 32K–475K long-context probes with
client-observed TTFT, output throughput, and cached-token accounting. LiteLLM
provides the stable OpenAI-compatible gateway; direct vLLM access remains a
diagnostic option.

The vLLM backend is **not** containerized here and **not** re-implemented; it is
the native ROCm stack from the sibling repo.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then edit OPENAI_BASE_URL / OPENAI_API_KEY
set -a && source .env && set +a
python app.py                 # http://127.0.0.1:7860
```

With the backend already running (see the sibling repo):

```bash
OPENAI_BASE_URL=http://127.0.0.1:4000/v1 MODEL_NAME=deepseek-v4-flash python app.py
```

## LiteLLM gateway

```bash
cp .env.example .env
# Set a long random LITELLM_MASTER_KEY and the reachable VLLM_BASE_URL in .env.
docker compose up -d
# LiteLLM: http://127.0.0.1:4000 (OpenAI-compatible API under /v1)
```

Point the Gradio app at `OPENAI_BASE_URL=http://127.0.0.1:4000/v1` and use the
LiteLLM master key as `OPENAI_API_KEY`. Compose binds the gateway to loopback;
put an authenticated TLS reverse proxy in front before exposing it.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_BASE_URL` | `http://127.0.0.1:4000/v1` | OpenAI-compatible gateway base URL |
| `OPENAI_API_KEY` | *(empty)* | Gateway Bearer key; preferred over the key file |
| `OPENAI_API_KEY_FILE` | *(empty)* | Read the gateway key from a mounted secret file |
| `VLLM_BASE_URL`, `VLLM_API_KEY`, `VLLM_API_KEY_FILE` | *(empty)* | Backward-compatible direct-vLLM aliases |
| `MODEL_NAME` | `deepseek-v4-flash` | Model id to request |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Gradio bind address |
| `GRADIO_SERVER_PORT` | `7860` | Gradio bind port |
| `GRADIO_ROOT_PATH` | *(empty)* | Reverse-proxy URL prefix |
| `MAX_CONTEXT_TOKENS` | `524288` | Backend context ceiling used to bound probes |
| `CONTEXT_PROBE_SAFETY_TOKENS` | `4096` | Space reserved for output and chat framing |
| `INFERENCE_CONCURRENCY_LIMIT` | `1` | Shared Gradio chat/probe concurrency |
| `QUEUE_MAX_SIZE` | `32` | Maximum queued UI requests |
| `CHAT_COMPLETIONS_PATH` | `/v1/chat/completions` | Override the chat endpoint path |
| `MODELS_PATH` | `/v1/models` | Override the models-list endpoint path |
| `BACKEND_REQUEST_TIMEOUT_SECONDS` | `900` | Per-request timeout to the backend |
| `BACKEND_CONNECT_TIMEOUT_SECONDS` | `10` | Connect timeout to the backend |

`PORT` and `VLLM_MODEL` are accepted as fallback aliases for `GRADIO_SERVER_PORT`
and `MODEL_NAME` respectively.

## Repository layout

```text
app.py                  Minimal application entry point
deepseek_lab/           configuration, backend client, probes, and Gradio UI
requirements.txt        runtime deps (gradio + httpx)
litellm/config.yaml     gateway routing config and model aliases
studio/                 platform-neutral start/health scripts + Studio guide
docs/                   ARCHITECTURE.md / DEPLOYMENT.md
docker-compose.yml      optional LiteLLM gateway container
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
