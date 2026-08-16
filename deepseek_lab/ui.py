"""Gradio interface and streaming callbacks for the DeepSeek lab."""

from __future__ import annotations

import html
import time
from collections.abc import Iterator
from typing import Any

import gradio as gr

from .backend import BackendRequestError, OpenAIBackend
from .config import AppConfig
from .probes import (
    CONTEXT_PROBE_TARGETS,
    build_context_probe_prompt,
    extract_cached_tokens,
)

DisplayMessage = dict[str, Any]
ApiMessage = dict[str, Any]


def _render_assistant_message(
    content: str,
    reasoning: str,
    show_reasoning: bool,
) -> str:
    """Render hidden reasoning without allowing it to inject raw HTML."""
    if not show_reasoning or not reasoning:
        return content

    escaped_reasoning = html.escape(reasoning)
    reasoning_panel = (
        "<details><summary>Reasoning</summary>"
        f"<pre>{escaped_reasoning}</pre></details>"
    )
    return f"{reasoning_panel}\n\n{content}"


def _read_usage_integer(usage: dict[str, Any] | None, field_name: str) -> int:
    if not usage:
        return 0
    field_value = usage.get(field_name, 0)
    return field_value if isinstance(field_value, int) else 0


def _format_chat_metrics(
    time_to_first_token: float | None,
    output_tokens_per_second: float | None,
    usage: dict[str, Any] | None,
) -> str:
    prompt_tokens = _read_usage_integer(usage, "prompt_tokens")
    completion_tokens = _read_usage_integer(usage, "completion_tokens")
    cached_tokens = extract_cached_tokens(usage)

    lines = ["**Client-observed request metrics**"]
    lines.append(
        f"- Time to first token: `{time_to_first_token:.2f}s`"
        if time_to_first_token is not None
        else "- Time to first token: pending"
    )
    lines.append(
        f"- Completion throughput: `{output_tokens_per_second:.1f} tok/s`"
        if output_tokens_per_second is not None
        else "- Completion throughput: pending"
    )
    if usage:
        lines.append(
            f"- Tokens: `{prompt_tokens}` prompt (cached `{cached_tokens}`) / "
            f"`{completion_tokens}` completion"
        )
    return "\n".join(lines)


def stream_chat(
    message: str,
    display_history: list[DisplayMessage] | None,
    api_conversation: list[ApiMessage] | None,
    temperature: float,
    maximum_output_tokens: int,
    show_reasoning: bool,
    backend: OpenAIBackend,
) -> Iterator[tuple[list[DisplayMessage], list[ApiMessage], str]]:
    """Stream a chat turn while keeping API history separate from UI markup."""
    normalized_message = (message or "").strip()
    existing_display_history = list(display_history or [])
    existing_api_conversation = list(api_conversation or [])

    if not normalized_message:
        yield (
            existing_display_history,
            existing_api_conversation,
            "Enter a message before sending.",
        )
        return

    user_display_message: DisplayMessage = {
        "role": "user",
        "content": normalized_message,
    }
    user_api_message: ApiMessage = {
        "role": "user",
        "content": normalized_message,
    }
    request_messages = existing_api_conversation + [user_api_message]
    pending_display_history = existing_display_history + [user_display_message]

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] | None = None
    request_started_at = time.perf_counter()
    first_token_at: float | None = None

    try:
        for chunk in backend.stream_chat_completion(
            request_messages,
            float(temperature),
            int(maximum_output_tokens),
        ):
            received_model_text = bool(chunk.content or chunk.reasoning)
            if received_model_text and first_token_at is None:
                first_token_at = time.perf_counter()

            content_parts.append(chunk.content)
            reasoning_parts.append(chunk.reasoning)
            if chunk.usage:
                usage = chunk.usage

            current_content = "".join(content_parts)
            current_reasoning = "".join(reasoning_parts)
            rendered_assistant = _render_assistant_message(
                current_content,
                current_reasoning,
                show_reasoning,
            )
            assistant_display_message: DisplayMessage = {
                "role": "assistant",
                "content": rendered_assistant,
            }
            assistant_api_message: ApiMessage = {
                "role": "assistant",
                "content": current_content,
            }
            time_to_first_token = (
                first_token_at - request_started_at
                if first_token_at is not None
                else None
            )
            yield (
                pending_display_history + [assistant_display_message],
                request_messages + [assistant_api_message],
                _format_chat_metrics(time_to_first_token, None, usage),
            )
    except BackendRequestError as error:
        error_message: DisplayMessage = {
            "role": "assistant",
            "content": f"**Request failed:** {error}",
        }
        # A failed user turn is visible for diagnosis but is not replayed to the
        # model on the next successful request.
        yield (
            pending_display_history + [error_message],
            existing_api_conversation,
            "Backend request failed; no conversation state was committed.",
        )
        return

    completed_at = time.perf_counter()
    final_content = "".join(content_parts)
    final_reasoning = "".join(reasoning_parts)
    if not final_content and not final_reasoning:
        final_content = "The backend returned no visible completion content."

    time_to_first_token = (
        first_token_at - request_started_at if first_token_at is not None else None
    )
    completion_tokens = _read_usage_integer(usage, "completion_tokens")
    decode_duration = (
        completed_at - first_token_at if first_token_at is not None else None
    )
    output_tokens_per_second = (
        completion_tokens / decode_duration
        if completion_tokens > 0 and decode_duration and decode_duration > 0
        else None
    )

    final_display_message: DisplayMessage = {
        "role": "assistant",
        "content": _render_assistant_message(
            final_content,
            final_reasoning,
            show_reasoning,
        ),
    }
    final_api_message: ApiMessage = {
        "role": "assistant",
        "content": final_content,
    }
    yield (
        pending_display_history + [final_display_message],
        request_messages + [final_api_message],
        _format_chat_metrics(
            time_to_first_token,
            output_tokens_per_second,
            usage,
        ),
    )


def run_context_probe(
    size_label: str,
    maximum_output_tokens: int,
    backend: OpenAIBackend,
    config: AppConfig,
) -> Iterator[str]:
    """Run one bounded long-prefill request and report client-side metrics."""
    requested_tokens = CONTEXT_PROBE_TARGETS.get(size_label)
    if requested_tokens is None:
        yield f"Unknown context probe size: `{size_label}`"
        return

    try:
        probe_prompt = build_context_probe_prompt(
            backend,
            config,
            requested_tokens,
            int(maximum_output_tokens),
        )
    except (BackendRequestError, ValueError) as error:
        yield f"**Unable to build probe:** {error}"
        return

    yield (
        f"Running {size_label} probe (estimated `{probe_prompt.estimated_tokens}` "
        f"prompt tokens; calibration: `{probe_prompt.calibration}`).\n\n"
        "This request is serialized with other UI inference requests."
    )

    request_started_at = time.perf_counter()
    first_token_at: float | None = None
    content_parts: list[str] = []
    usage: dict[str, Any] | None = None

    try:
        for chunk in backend.stream_chat_completion(
            probe_prompt.messages,
            0.0,
            int(maximum_output_tokens),
        ):
            if (chunk.content or chunk.reasoning) and first_token_at is None:
                first_token_at = time.perf_counter()
            content_parts.append(chunk.content)
            if chunk.usage:
                usage = chunk.usage
    except BackendRequestError as error:
        yield f"**Probe failed:** {error}"
        return

    completed_at = time.perf_counter()
    time_to_first_token = (
        first_token_at - request_started_at if first_token_at is not None else None
    )
    prompt_tokens = _read_usage_integer(usage, "prompt_tokens")
    completion_tokens = _read_usage_integer(usage, "completion_tokens")
    cached_tokens = extract_cached_tokens(usage)
    observed_input_throughput = (
        prompt_tokens / time_to_first_token
        if prompt_tokens > 0 and time_to_first_token and time_to_first_token > 0
        else None
    )
    decode_duration = (
        completed_at - first_token_at if first_token_at is not None else None
    )
    output_tokens_per_second = (
        completion_tokens / decode_duration
        if completion_tokens > 0 and decode_duration and decode_duration > 0
        else None
    )

    result_lines = [
        f"## {size_label} context probe",
        "",
        f"- Requested target: `{probe_prompt.requested_tokens}` tokens",
        f"- Actual prompt: `{prompt_tokens}` tokens (cached `{cached_tokens}`)",
        (
            f"- Client-observed input throughput: `{observed_input_throughput:.0f} tok/s`"
            if observed_input_throughput is not None
            else "- Client-observed input throughput: unavailable"
        ),
        (
            f"- Time to first token: `{time_to_first_token:.2f}s`"
            if time_to_first_token is not None
            else "- Time to first token: unavailable"
        ),
        (
            f"- Completion throughput: `{output_tokens_per_second:.1f} tok/s`"
            if output_tokens_per_second is not None
            else "- Completion throughput: unavailable"
        ),
        "",
        (
            "Input throughput is derived from end-to-end TTFT and therefore "
            "includes queueing, template processing, and network overhead."
        ),
        "",
        "**Answer**",
        "",
        "".join(content_parts).strip() or "(no visible content)",
    ]
    yield "\n".join(result_lines)


def build_demo(config: AppConfig, backend: OpenAIBackend) -> gr.Blocks:
    """Construct the complete Gradio application."""
    theme = gr.themes.Soft(primary_hue="red", secondary_hue="slate")
    with gr.Blocks(title="MI308X DeepSeek Lab", theme=theme) as demo:
        gr.Markdown(
            "# MI308X DeepSeek Lab\n"
            "**DeepSeek-V4-Flash-0731** served by vLLM on ROCm for a single "
            "**AMD Instinct MI308X (192 GB)**. This UI consumes the backend's "
            "OpenAI-compatible API; it does not load model weights itself."
        )

        api_conversation = gr.State([])
        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Backend status")
                system_info = gr.JSON(
                    value={"status": "not checked"},
                    label="Runtime",
                )
                refresh_button = gr.Button("Refresh status")
                refresh_button.click(backend.probe_system, outputs=system_info)

                gr.Markdown("### Inference controls")
                temperature = gr.Slider(
                    0.0,
                    2.0,
                    value=0.6,
                    step=0.1,
                    label="Temperature",
                )
                maximum_output_tokens = gr.Slider(
                    16,
                    8_192,
                    value=2_048,
                    step=16,
                    label="Maximum output tokens",
                )
                show_reasoning = gr.Checkbox(
                    value=True,
                    label="Show reasoning when provided",
                )

            with gr.Column(scale=2, min_width=520), gr.Tabs():
                    with gr.Tab("Chat"):
                        chatbot = gr.Chatbot(
                            type="messages",
                            height=520,
                            label="DeepSeek-V4-Flash",
                        )
                        chat_metrics = gr.Markdown("")
                        with gr.Row():
                            message_input = gr.Textbox(
                                placeholder="Ask a question or paste repository context...",
                                show_label=False,
                                scale=6,
                            )
                            send_button = gr.Button(
                                "Send",
                                variant="primary",
                                scale=1,
                            )
                            stop_button = gr.Button("Stop", scale=1)
                            clear_button = gr.Button("Clear", scale=1)

                        chat_inputs = [
                            message_input,
                            chatbot,
                            api_conversation,
                            temperature,
                            maximum_output_tokens,
                            show_reasoning,
                        ]
                        chat_outputs = [chatbot, api_conversation, chat_metrics]
                        send_event = send_button.click(
                            lambda *arguments: stream_chat(
                                *arguments,
                                backend=backend,
                            ),
                            inputs=chat_inputs,
                            outputs=chat_outputs,
                            concurrency_id="backend-inference",
                            concurrency_limit=config.inference_concurrency_limit,
                        )
                        send_event.then(lambda: "", outputs=message_input)
                        submit_event = message_input.submit(
                            lambda *arguments: stream_chat(
                                *arguments,
                                backend=backend,
                            ),
                            inputs=chat_inputs,
                            outputs=chat_outputs,
                            concurrency_id="backend-inference",
                            concurrency_limit=config.inference_concurrency_limit,
                        )
                        submit_event.then(lambda: "", outputs=message_input)
                        stop_button.click(
                            fn=None,
                            cancels=[send_event, submit_event],
                        )
                        clear_button.click(
                            lambda: ([], [], ""),
                            outputs=chat_outputs,
                            cancels=[send_event, submit_event],
                        )

                    with gr.Tab("Long-context probe"):
                        gr.Markdown(
                            "Send a deterministic synthetic prefix and measure "
                            "client-observed TTFT, throughput, and cached tokens. "
                            "The 475K option is intentionally below the 524,288-token "
                            "backend ceiling so output and chat-template framing fit."
                        )
                        with gr.Row():
                            probe_size = gr.Dropdown(
                                list(CONTEXT_PROBE_TARGETS),
                                value="128K",
                                label="Target prompt size",
                            )
                            probe_maximum_output_tokens = gr.Slider(
                                16,
                                512,
                                value=64,
                                step=16,
                                label="Maximum output tokens",
                            )
                            probe_button = gr.Button("Run probe", variant="primary")
                            stop_probe_button = gr.Button("Stop")
                        probe_output = gr.Markdown("")
                        probe_event = probe_button.click(
                            lambda size, output_tokens: run_context_probe(
                                size,
                                output_tokens,
                                backend,
                                config,
                            ),
                            inputs=[probe_size, probe_maximum_output_tokens],
                            outputs=probe_output,
                            concurrency_id="backend-inference",
                            concurrency_limit=config.inference_concurrency_limit,
                        )
                        stop_probe_button.click(fn=None, cancels=[probe_event])

                    with gr.Tab("API reference"):
                        gr.Markdown(
                            "The inference service exposes an OpenAI-compatible API. "
                            "Configure an OpenAI SDK or raw HTTP client with the same "
                            "base URL, model name, and bearer key. Set "
                            "`OPENAI_BASE_URL` to an endpoint ending in `/v1`.\n\n"
                            "```bash\n"
                            "curl \"$OPENAI_BASE_URL/chat/completions\" \\\n"
                            "  -H \"Content-Type: application/json\" \\\n"
                            "  -H \"Authorization: Bearer $OPENAI_API_KEY\" \\\n"
                            "  -d '{\n"
                            f"    \"model\": \"{config.model_name}\",\n"
                            "    \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}],\n"
                            "    \"stream\": true\n"
                            "  }'\n"
                            "```"
                        )

        gr.Markdown(
            "Serving and benchmark implementation: "
            "[`deepseek-v4-flash-mi308x`](https://github.com/DeliciousBuding/"
            "deepseek-v4-flash-mi308x)."
        )
        demo.load(backend.probe_system, outputs=system_info)

    return demo
