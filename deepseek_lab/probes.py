"""Long-context probe generation and client-observed metric helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .backend import BackendRequestError, OpenAIBackend
from .config import AppConfig

CONTEXT_PROBE_TARGETS = {
    "32K": 32_000,
    "128K": 128_000,
    "256K": 256_000,
    "475K": 475_000,
}

_PROBE_RECORD = (
    "\nRepository record: the service validates requests, streams model output, "
    "and reports client-observed latency metrics."
)


@dataclass(frozen=True, slots=True)
class ContextProbePrompt:
    """A bounded synthetic prompt and the assumptions used to construct it."""

    messages: list[dict[str, str]]
    requested_tokens: int
    estimated_tokens: int
    calibration: str


def _estimate_tokens_per_record(backend: OpenAIBackend) -> tuple[float, str]:
    calibration_record_count = 64
    calibration_prompt = _PROBE_RECORD * calibration_record_count

    try:
        measured_tokens = backend.count_tokens(calibration_prompt)
        tokens_per_record = measured_tokens / calibration_record_count
        return max(tokens_per_record, 1.0), "backend /tokenize"
    except BackendRequestError:
        # Two English characters per token deliberately overestimates the token
        # density. The fallback may undershoot the selected label, but it will
        # not risk overflowing a backend whose tokenizer cannot be queried.
        estimated_tokens = len(calibration_prompt) / 2.0
        tokens_per_record = estimated_tokens / calibration_record_count
        return max(tokens_per_record, 1.0), "2 characters/token safe fallback"


def build_context_probe_prompt(
    backend: OpenAIBackend,
    config: AppConfig,
    requested_tokens: int,
    maximum_output_tokens: int,
) -> ContextProbePrompt:
    """Build a repeatable prompt that stays below the configured context limit."""
    if requested_tokens <= 0:
        raise ValueError("requested_tokens must be greater than zero")
    if maximum_output_tokens <= 0:
        raise ValueError("maximum_output_tokens must be greater than zero")

    available_prompt_tokens = (
        config.maximum_context_tokens
        - maximum_output_tokens
        - config.context_probe_safety_tokens
    )
    if available_prompt_tokens <= 0:
        raise ValueError(
            "MAX_CONTEXT_TOKENS is too small for the requested output and safety margin"
        )

    bounded_target_tokens = min(requested_tokens, available_prompt_tokens)
    tokens_per_record, calibration = _estimate_tokens_per_record(backend)

    # Leave an additional two percent for chat-template framing and tokenizer
    # boundary effects. The response usage field reports the actual token count.
    synthetic_content_budget = max(int(bounded_target_tokens * 0.98), 1)
    record_count = max(int(synthetic_content_budget / tokens_per_record), 1)
    estimated_tokens = int(record_count * tokens_per_record)

    synthetic_document = _PROBE_RECORD * record_count
    messages = [
        {
            "role": "system",
            "content": (
                "You are validating long-context retrieval. Answer concisely and "
                "do not repeat the supplied records."
            ),
        },
        {
            "role": "user",
            "content": (
                synthetic_document
                + "\n\nWhat three operations are repeated in every repository record?"
            ),
        },
    ]
    return ContextProbePrompt(
        messages=messages,
        requested_tokens=requested_tokens,
        estimated_tokens=estimated_tokens,
        calibration=calibration,
    )


def extract_cached_tokens(usage: dict[str, Any] | None) -> int:
    """Read OpenAI-compatible cached-token details defensively."""
    if not usage:
        return 0
    prompt_details = usage.get("prompt_tokens_details")
    if not isinstance(prompt_details, dict):
        return 0
    cached_tokens = prompt_details.get("cached_tokens", 0)
    return cached_tokens if isinstance(cached_tokens, int) else 0
