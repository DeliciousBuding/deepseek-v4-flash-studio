#!/usr/bin/env python3
"""MI308X DeepSeek Lab — a thin Gradio presentation layer over the
DeepSeek-V4-Flash-0731 OpenAI-compatible vLLM endpoint.

The serving stack itself (vLLM + ROCm + MI308X) lives in the sibling repository
``deepseek-v4-flash-mi308x``. This project only owns the UI/gateway shell:

    browser -> Gradio -> OpenAI-compatible API -> vLLM -> DeepSeek-V4-Flash -> MI308X

Configuration is environment-driven so the same app runs against a local vLLM,
a ModelScope Studio backend, or any OpenAI-compatible gateway (e.g. LiteLLM):

    VLLM_BASE_URL   base URL of the OpenAI-compatible server (default http://127.0.0.1:8000)
    VLLM_API_KEY    optional bearer key (omit when the backend is unauthenticated)
    MODEL_NAME      model id to request (default deepseek-v4-flash)
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Iterator

import gradio as gr
import httpx

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("VLLM_API_KEY", "")
MODEL = os.environ.get("MODEL_NAME", os.environ.get("VLLM_MODEL", "deepseek-v4-flash"))
CHAT_PATH = os.environ.get("CHAT_COMPLETIONS_PATH", "/v1/chat/completions")

# Prefix sizes exposed by the long-context probe tab (approximate token counts).
CONTEXT_PROBES = {
    "32K": 32_000,
    "128K": 128_000,
    "256K": 256_000,
    "512K": 512_000,
}


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


def _chat_url() -> str:
    return BASE_URL.rstrip("/") + CHAT_PATH


def _models_url() -> str:
    return BASE_URL.rstrip("/") + "/v1/models"


def probe_system() -> dict[str, str]:
    """Collect a small, human-readable snapshot of the backend it is talking to.

    Never blocks hard on an unavailable backend; every probe is individually
    guarded so the UI still renders while the serving stack is starting up.
    """
    info: dict[str, str] = {
        "base_url": BASE_URL,
        "model": MODEL,
    }

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(_models_url(), headers=_headers())
            resp.raise_for_status()
            ids = [m.get("id", "?") for m in resp.json().get("data", [])]
            info["models"] = ", ".join(ids) if ids else "(none)"
    except Exception as exc:  # noqa: BLE001 — the UI must tolerate a cold backend
        info["models"] = f"(unreachable: {exc.__class__.__name__})"

    # VRAM is only meaningful when the UI runs on the GPU host itself (the
    # ModelScope Studio case); elsewhere the probe degrades gracefully.
    try:
        out = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=5.0, check=False,
        ).stdout
        lines = [ln.strip() for ln in out.splitlines() if "MiB" in ln]
        info["vram"] = lines[0] if lines else "(rocm-smi unavailable)"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["vram"] = "(rocm-smi not on this host)"

    return info


def _synthetic_prefix(tokens: int) -> str:
    """Return a repeatable, token-dense filler for the long-context probe.

    This is deliberately approximate: it drives a long prefill to exercise the
    prefix-cache / KV path rather than producing an exact token count.
    """
    unit = (
        "AMD Instinct MI308X exposes 192 GB of HBM3 and 80 compute units on "
        "the gfx942 ISA. DeepSeek-V4-Flash-0731 runs as a sparse-MLA MoE with "
        "MXFP4 experts and an FP8 trunk. "
    )
    return unit * (tokens // 24 + 1)


def stream_completion(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> Iterator[tuple[str, str, dict | None]]:
    """Stream one chat completion, yielding (content_delta, reasoning_delta, usage)."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    with httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
        with client.stream("POST", _chat_url(), headers=_headers(), json=payload) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                delta = choices[0].get("delta", {}) if choices else {}
                content = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or ""
                usage = obj.get("usage")
                yield content, reasoning, usage


def _render_assistant(content: str, reasoning: str, show_reasoning: bool) -> str:
    if show_reasoning and reasoning:
        return (
            "<details open><summary>Reasoning</summary>"
            f"\n\n{reasoning}\n\n</details>\n\n{content}"
        )
    return content


def _format_metrics(ttft: float | None, output_tps: float | None, usage: dict | None) -> str:
    lines = [
        "**Latency / throughput**",
    ]
    lines.append(f"- TTFT: `{ttft:.2f}s`" if ttft is not None else "- TTFT: `—`")
    lines.append(f"- Output: `{output_tps:.1f} tok/s`" if output_tps is not None else "- Output: `— tok/s`")
    if usage:
        lines.append(
            f"- Tokens: `{usage.get('prompt_tokens', 0)}` prompt "
            f"(cached `{usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)}`) "
            f"/ `{usage.get('completion_tokens', 0)}` completion"
        )
    return "\n".join(lines)


def chat(
    message: str,
    history: list[dict],
    temperature: float,
    max_tokens: int,
    show_reasoning: bool,
):
    """Gradio chat callback. Yields (history, metrics) so the UI streams."""
    history = list(history or [])
    history.append({"role": "user", "content": message})

    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict | None = None
    ttft: float | None = None
    output_tps: float | None = None

    start = time.time()
    first_token_at: float | None = None
    try:
        for content_delta, reasoning_delta, chunk_usage in stream_completion(
            messages, temperature, max_tokens
        ):
            if content_delta and first_token_at is None:
                first_token_at = time.time()
                ttft = first_token_at - start
            content_parts.append(content_delta)
            reasoning_parts.append(reasoning_delta)
            if chunk_usage:
                usage = chunk_usage

            assistant = _render_assistant(
                "".join(content_parts), "".join(reasoning_parts), show_reasoning
            )
            yield history + [{"role": "assistant", "content": assistant}], _format_metrics(
                ttft, output_tps, usage
            )
    except Exception as exc:  # noqa: BLE001 — surface transport errors in the UI
        yield history + [{"role": "assistant", "content": f"⚠️ `{exc}`"}], ""

    if first_token_at is not None and usage:
        decode_tokens = usage.get("completion_tokens", 0)
        decode_seconds = time.time() - first_token_at
        if decode_seconds > 0:
            output_tps = decode_tokens / decode_seconds

    final = _render_assistant(
        "".join(content_parts), "".join(reasoning_parts), show_reasoning
    )
    yield history + [{"role": "assistant", "content": final}], _format_metrics(
        ttft, output_tps, usage
    )


def run_context_probe(size_label: str, max_tokens: int) -> Iterator[str]:
    """Run a single long-prefill request and report TTFT / throughput / usage."""
    tokens = CONTEXT_PROBES[size_label]
    prefix = _synthetic_prefix(tokens)
    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": prefix + "\n\nCount the words in the text above."},
    ]
    yield f"Running {size_label} probe (~{tokens} prompt tokens)…\n\n"

    start = time.time()
    first_token_at: float | None = None
    content_parts: list[str] = []
    usage: dict | None = None
    for content_delta, _reasoning, chunk_usage in stream_completion(messages, 0.0, max_tokens):
        if content_delta and first_token_at is None:
            first_token_at = time.time()
        content_parts.append(content_delta)
        if chunk_usage:
            usage = chunk_usage

    ttft = (first_token_at - start) if first_token_at else None
    prefill_seconds = ttft
    decode_tokens = (usage or {}).get("completion_tokens", 0)
    decode_seconds = (time.time() - first_token_at) if first_token_at else None
    output_tps = decode_tokens / decode_seconds if decode_seconds else None
    prompt_tokens = (usage or {}).get("prompt_tokens", 0)
    prefill_tps = prompt_tokens / prefill_seconds if (prefill_seconds and prompt_tokens) else None

    lines = [
        f"## {size_label} context probe result",
        "",
        f"- Prompt tokens: `{prompt_tokens}`",
        f"- Prefill: `{prefill_tps:.0f} tok/s`" if prefill_tps else "- Prefill: `—`",
        f"- TTFT: `{ttft:.2f}s`" if ttft is not None else "- TTFT: `—`",
        f"- Output: `{output_tps:.1f} tok/s`" if output_tps is not None else "- Output: `—`",
        "",
        "Answer:",
        "".join(content_parts).strip(),
    ]
    yield "\n".join(lines)


def build_demo() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="red", secondary_hue="slate")
    with gr.Blocks(title="MI308X DeepSeek Lab", theme=theme) as demo:
        gr.Markdown(
            "# MI308X DeepSeek Lab\n"
            "**DeepSeek-V4-Flash-0731** on a single **AMD Instinct MI308X (192 GB)** "
            "— long-context inference lab. Backend: vLLM on ROCm, exposed as an "
            "OpenAI-compatible API."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### System")
                system_info = gr.JSON(value=probe_system(), label="Backend")
                refresh = gr.Button("Refresh")
                refresh.click(probe_system, outputs=system_info)

                gr.Markdown("### Inference")
                temperature = gr.Slider(0.0, 2.0, value=0.6, step=0.1, label="Temperature")
                max_tokens = gr.Slider(16, 8192, value=2048, step=16, label="Max output tokens")
                show_reasoning = gr.Checkbox(value=True, label="Show reasoning")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Chat"):
                        chatbot = gr.Chatbot(type="messages", height=520, label="DeepSeek-V4-Flash")
                        metrics = gr.Markdown("")
                        with gr.Row():
                            msg = gr.Textbox(
                                placeholder="Ask anything; use long context, reasoning, or tools…",
                                show_label=False, scale=6,
                            )
                            send = gr.Button("Send", variant="primary", scale=1)
                            clear = gr.Button("Clear", scale=1)
                        send.click(
                            chat,
                            inputs=[msg, chatbot, temperature, max_tokens, show_reasoning],
                            outputs=[chatbot, metrics],
                            queue=True,
                        ).then(lambda: "", outputs=msg)
                        msg.submit(
                            chat,
                            inputs=[msg, chatbot, temperature, max_tokens, show_reasoning],
                            outputs=[chatbot, metrics],
                            queue=True,
                        ).then(lambda: "", outputs=msg)
                        clear.click(lambda: (None, ""), outputs=[chatbot, metrics])

                    with gr.Tab("Long-context test"):
                        gr.Markdown(
                            "Send a single long prefill and measure prefill speed, "
                            "TTFT and output throughput. Best run once the backend "
                            "is warm; repeat the same size to observe prefix-cache hits."
                        )
                        with gr.Row():
                            probe_size = gr.Dropdown(
                                list(CONTEXT_PROBES.keys()), value="128K", label="Prefix size"
                            )
                            probe_max = gr.Slider(16, 2048, value=256, step=16, label="Max output tokens")
                            probe_run = gr.Button("Run probe", variant="primary")
                        probe_output = gr.Markdown("")
                        probe_run.click(
                            run_context_probe,
                            inputs=[probe_size, probe_max],
                            outputs=probe_output,
                            queue=True,
                        )

                    with gr.Tab("API reference"):
                        gr.Markdown(
                            "The backend is OpenAI-compatible. Point any OpenAI SDK or "
                            "raw HTTP client at `{base}`.\n\n"
                            "```bash\n"
                            "curl {base}/v1/chat/completions \\\n"
                            "  -H \"Content-Type: application/json\" \\\n"
                            "  -H \"Authorization: Bearer $VLLM_API_KEY\" \\\n"
                            "  -d '{{\n"
                            "    \"model\": \"{model}\",\n"
                            "    \"messages\": [{{\"role\": \"user\", \"content\": \"Hello\"}}],\n"
                            "    \"stream\": true\n"
                            "  }}'\n"
                            "```".format(base=BASE_URL.rstrip("/"), model=MODEL)
                        )

    return demo


if __name__ == "__main__":
    build_demo().queue().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_api=False,
    )
