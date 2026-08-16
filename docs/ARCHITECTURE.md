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
│  app.py         minimal process entry point │
│  deepseek_lab/  config, API client, UI      │
│  litellm/       OpenAI gateway configuration│
│  docker-compose optional gateway container  │
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

1. **One UI and one gateway.** The Gradio app depends only on `gradio` + `httpx`.
   LiteLLM standardizes authentication and the OpenAI-compatible API without
   adding another user interface or duplicating the inference service.

2. **Raw HTTP, not an SDK.** The app streams SSE from the OpenAI-compatible
   endpoint with `httpx` rather than pulling in the `openai` SDK, avoiding
   version drift against the backend and keeping the dependency list tiny.

3. **Everything is environment-driven.** The same `app.py` runs against a local
   or remote OpenAI-compatible gateway using `OPENAI_BASE_URL`,
   `OPENAI_API_KEY`, and `MODEL_NAME`. The former `VLLM_*` variables remain
   backward-compatible aliases for direct backend diagnostics.

4. **No secrets, no bootstrap.** This repo never generates, stores, or commits
   API keys, model weights, or instance-specific connection details.

5. **Display state is not model state.** Collapsible reasoning is escaped and
   kept only in the Gradio history. The API conversation contains plain user and
   assistant content, so presentation HTML and hidden reasoning are never
   replayed into later prompts.

6. **Long probes are bounded.** The app calibrates a small fixture with the
   gateway's `/tokenize` pass-through, reserves output/template headroom, and
   caps the largest UI option at 475K for a 524,288-token backend. If the route
   is unavailable, a conservative fallback undershoots rather than overflows.

## The "lab" surface

The UI is deliberately more than a chatbot wrapper. It exposes:

- **Chat** with streaming and collapsible reasoning (`reasoning_content`).
- **Long-context probe** (32K / 128K / 256K / 475K) reporting client-observed
  input throughput, TTFT, output throughput, and cached-token accounting.
  Input throughput is derived from end-to-end TTFT and therefore includes
  queueing, chat-template processing, and network overhead; it is not presented
  as an engine-only prefill benchmark.
- **API reference** so the OpenAI-compatible surface is documented in-product.

## LiteLLM gateway

`docker-compose.yml` runs only LiteLLM in front of the native backend. LiteLLM
uses its `hosted_vllm/` provider and environment-backed credentials, so model
aliases or fallbacks do not require forking either upstream project. The
Compose port binds to loopback and the master key has no repository default.
