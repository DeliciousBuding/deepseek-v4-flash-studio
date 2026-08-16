"""Small OpenAI-compatible HTTP client used by the Gradio application."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx

from .config import AppConfig


class BackendRequestError(RuntimeError):
    """A user-safe error raised when the inference backend rejects a request."""


@dataclass(frozen=True, slots=True)
class CompletionChunk:
    """The useful fields from one streaming chat-completion event."""

    content: str
    reasoning: str
    usage: dict[str, Any] | None


def parse_server_sent_event(raw_line: str) -> CompletionChunk | None:
    """Parse one OpenAI-compatible SSE data line.

    Keep-alives, comments, malformed third-party extension events, and the
    terminal ``[DONE]`` marker do not produce a chunk.
    """
    stripped_line = raw_line.strip()
    if not stripped_line.startswith("data:"):
        return None

    event_data = stripped_line[len("data:") :].strip()
    if not event_data or event_data == "[DONE]":
        return None

    try:
        event_payload = json.loads(event_data)
    except json.JSONDecodeError:
        return None
    if not isinstance(event_payload, dict):
        return None

    error_payload = event_payload.get("error")
    if error_payload:
        if isinstance(error_payload, dict):
            error_message = error_payload.get("message", "Backend streaming error")
        else:
            error_message = str(error_payload)
        raise BackendRequestError(str(error_message))

    choices = event_payload.get("choices") or []
    first_choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    delta = first_choice.get("delta", {})
    if not isinstance(delta, dict):
        delta = {}
    content = delta.get("content") or ""
    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
    usage = event_payload.get("usage")

    if not isinstance(content, str):
        content = str(content)
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)
    if usage is not None and not isinstance(usage, dict):
        usage = None

    return CompletionChunk(content=content, reasoning=reasoning, usage=usage)


class OpenAIBackend:
    """Environment-configured client for one OpenAI-compatible backend."""

    def __init__(self, config: AppConfig):
        self.config = config

    def request_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            float(self.config.request_timeout_seconds),
            connect=float(self.config.connect_timeout_seconds),
        )

    def _raise_for_status(self, response: httpx.Response, operation: str) -> None:
        if response.is_success:
            return

        response_body = response.read().decode("utf-8", errors="replace").strip()
        if self.config.api_key:
            response_body = response_body.replace(self.config.api_key, "[redacted]")
        response_excerpt = response_body[:500]
        detail = f": {response_excerpt}" if response_excerpt else ""
        raise BackendRequestError(
            f"Backend {operation} failed with HTTP {response.status_code}{detail}"
        )

    def stream_chat_completion(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        maximum_output_tokens: int,
    ) -> Iterator[CompletionChunk]:
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": maximum_output_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        try:
            with (
                httpx.Client(timeout=self._timeout()) as client,
                client.stream(
                    "POST",
                    self.config.chat_completions_url,
                    headers=self.request_headers(),
                    json=payload,
                ) as response,
            ):
                self._raise_for_status(response, "chat request")
                for raw_line in response.iter_lines():
                    parsed_chunk = parse_server_sent_event(raw_line)
                    if parsed_chunk is not None:
                        yield parsed_chunk
        except BackendRequestError:
            raise
        except httpx.TimeoutException as error:
            raise BackendRequestError(
                "The inference backend timed out before completing the request"
            ) from error
        except httpx.HTTPError as error:
            raise BackendRequestError(
                f"Unable to reach the inference backend ({error.__class__.__name__})"
            ) from error

    def count_tokens(self, prompt: str) -> int:
        """Use vLLM's root /tokenize endpoint for a small calibration prompt."""
        payload = {"model": self.config.model_name, "prompt": prompt}
        try:
            tokenization_timeout = httpx.Timeout(
                30.0,
                connect=float(self.config.connect_timeout_seconds),
            )
            with httpx.Client(timeout=tokenization_timeout) as client:
                response = client.post(
                    self.config.tokenize_url,
                    headers=self.request_headers(),
                    json=payload,
                )
                self._raise_for_status(response, "tokenize request")
                response_payload = response.json()
        except BackendRequestError:
            raise
        except (httpx.HTTPError, ValueError) as error:
            raise BackendRequestError(
                "The backend tokenization endpoint is unavailable"
            ) from error

        if not isinstance(response_payload, dict):
            raise BackendRequestError(
                "The backend tokenization endpoint returned an invalid response"
            )
        token_count = response_payload.get("count")
        if not isinstance(token_count, int) or token_count <= 0:
            raise BackendRequestError(
                "The backend tokenization endpoint returned an invalid count"
            )
        return token_count

    def probe_system(self) -> dict[str, Any]:
        """Return a bounded backend/GPU status snapshot without blocking startup."""
        status: dict[str, Any] = {
            "status": "unreachable",
            "configured_model": self.config.model_name,
            "protocol": "OpenAI-compatible HTTP",
        }

        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(
                    self.config.models_url,
                    headers=self.request_headers(),
                )
                self._raise_for_status(response, "model-list request")
                response_payload = response.json()

            if not isinstance(response_payload, dict):
                raise BackendRequestError("The model-list response is not an object")
            model_entries = response_payload.get("data", [])
            if not isinstance(model_entries, list):
                raise BackendRequestError("The model-list data field is not a list")
            status["status"] = "ready"
            status["served_models"] = [
                entry.get("id", "unknown")
                for entry in model_entries
                if isinstance(entry, dict)
            ]

            configured_entry = next(
                (
                    entry
                    for entry in model_entries
                    if isinstance(entry, dict)
                    and entry.get("id") == self.config.model_name
                ),
                None,
            )
            if configured_entry and configured_entry.get("max_model_len"):
                status["maximum_context_tokens"] = configured_entry[
                    "max_model_len"
                ]
        except BackendRequestError:
            status["detail"] = "Backend model-list check failed"
        except (httpx.HTTPError, ValueError) as error:
            status["detail"] = f"Backend probe failed ({error.__class__.__name__})"

        gpu_summary = _read_rocm_gpu_summary()
        if gpu_summary:
            status["local_gpu"] = gpu_summary
        return status


def _read_rocm_gpu_summary() -> dict[str, str] | None:
    """Read a few non-sensitive ROCm metrics when the UI is on the GPU host."""
    try:
        completed_process = subprocess.run(
            ["rocm-smi", "--showuse", "--showmemuse", "--json"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if completed_process.returncode != 0 or not completed_process.stdout.strip():
        return None

    try:
        gpu_payload = json.loads(completed_process.stdout)
    except json.JSONDecodeError:
        return None

    if not isinstance(gpu_payload, dict) or not gpu_payload:
        return None

    first_gpu = next(
        (value for value in gpu_payload.values() if isinstance(value, dict)),
        None,
    )
    if not first_gpu:
        return None

    summary: dict[str, str] = {}
    for metric_name, metric_value in first_gpu.items():
        normalized_name = metric_name.lower()
        if "gpu use" in normalized_name:
            summary["compute_usage"] = str(metric_value)
        elif "memory use" in normalized_name:
            summary["memory_usage"] = str(metric_value)
    return summary or None
