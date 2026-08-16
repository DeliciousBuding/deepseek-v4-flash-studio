# Architecture

This repository is a strict presentation layer. It converts the
OpenAI-compatible API served by the sibling repo
`deepseek-v4-flash-mi308x` into an interactive "lab" experience, without
re-implementing or containerizing the serving stack.

## Component boundaries

```text
┌─────────────────────────────────────────────┐
│ deepseek-v4-flash-studio (this repo)         │
│                                             │
│  app.py         Gradio UI, streaming chat   │
│  litellm/       optional gateway (full mode) │
│  open-webui/    optional rich chat UX        │
│  docker-compose optional full-mode stack     │
└──────────────────┬──────────────────────────┘
                   │  OpenAI-compatible /v1
                   ▼
┌─────────────────────────────────────────────┐
│ deepseek-v4-flash-mi308x (sibling repo)      │
│  vLLM + ROCm + DeepSeek-V4-Flash + MI308X    │
│  owns serving config, patches, benchmarks    │
└─────────────────────────────────────────────┘
```

## Design decisions

1. **Lite mode is the default and the fallback.** The Gradio app depends only on
   `gradio` + `httpx`, so it runs in ModelScope Studio with minimal surface area
   for dependency or startup failures. Open WebUI + LiteLLM are opt-in.

2. **Raw HTTP, not an SDK.** The app streams SSE from the OpenAI-compatible
   endpoint with `httpx` rather than pulling in the `openai` SDK, avoiding
   version drift against the backend and keeping the dependency list tiny.

3. **Everything is environment-driven.** The same `app.py` runs against a local
   vLLM, a ModelScope Studio backend, or a LiteLLM gateway — only
   `VLLM_BASE_URL` / `VLLM_API_KEY` / `MODEL_NAME` change.

4. **No secrets, no bootstrap.** This repo never generates, stores, or commits
   API keys, model weights, or instance-specific connection details.

## The "lab" surface

The UI is deliberately more than a chatbot wrapper. It exposes:

- **Chat** with streaming and collapsible reasoning (`reasoning_content`).
- **Long-context probe** (32K / 128K / 256K / 512K) reporting prefill speed,
  TTFT, output throughput, and cached-token accounting — demonstrating a real
  AMD deployment rather than a stock demo.
- **API reference** so the OpenAI-compatible surface is documented in-product.

## Full mode (optional)

`docker-compose.yml` runs Open WebUI → LiteLLM → the same backend. LiteLLM is
configured purely through `litellm/config.yaml` model aliases and
`os.environ/...` values, so adding model profiles or fallbacks does not require
forking either upstream project.
