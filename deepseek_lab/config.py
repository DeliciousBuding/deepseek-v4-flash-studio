"""Environment-backed configuration for the Studio application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def _read_positive_integer(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    try:
        parsed_value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from error

    if parsed_value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed_value


def _read_api_key() -> str:
    environment_key = os.environ.get("VLLM_API_KEY", "").strip()
    if environment_key:
        return environment_key

    key_file_value = os.environ.get("VLLM_API_KEY_FILE", "").strip()
    if not key_file_value:
        return ""

    key_file_path = Path(key_file_value).expanduser()
    try:
        file_key = key_file_path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ValueError(
            f"Unable to read VLLM_API_KEY_FILE at {key_file_path}"
        ) from error

    if not file_key:
        raise ValueError(f"VLLM_API_KEY_FILE is empty: {key_file_path}")
    return file_key


def normalize_base_url(base_url: str) -> str:
    """Validate and normalize an HTTP(S) backend base URL."""
    normalized_url = base_url.strip().rstrip("/")
    parsed_url = urlsplit(normalized_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError(
            "VLLM_BASE_URL must be an absolute http:// or https:// URL"
        )
    return normalized_url


def join_api_path(base_url: str, endpoint_path: str) -> str:
    """Join a base URL with an API path without duplicating a trailing /v1."""
    normalized_base_url = normalize_base_url(base_url)
    normalized_endpoint_path = "/" + endpoint_path.lstrip("/")

    if normalized_base_url.endswith("/v1") and normalized_endpoint_path.startswith(
        "/v1/"
    ):
        normalized_endpoint_path = normalized_endpoint_path[len("/v1") :]
    return normalized_base_url + normalized_endpoint_path


def join_root_path(base_url: str, endpoint_path: str) -> str:
    """Join a vLLM root endpoint such as /health or /tokenize."""
    parsed_url = urlsplit(normalize_base_url(base_url))
    base_path = parsed_url.path.rstrip("/").removesuffix("/v1")

    endpoint = "/" + endpoint_path.lstrip("/")
    joined_path = base_path + endpoint
    return urlunsplit(
        (parsed_url.scheme, parsed_url.netloc, joined_path, "", "")
    )


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Complete runtime configuration for the UI and backend client."""

    base_url: str
    api_key: str
    model_name: str
    chat_completions_path: str
    models_path: str
    server_name: str
    server_port: int
    root_path: str
    request_timeout_seconds: int
    connect_timeout_seconds: int
    maximum_context_tokens: int
    context_probe_safety_tokens: int
    inference_concurrency_limit: int
    queue_max_size: int

    @classmethod
    def from_environment(cls) -> AppConfig:
        configured_port = os.environ.get(
            "GRADIO_SERVER_PORT", os.environ.get("PORT", "7860")
        )
        try:
            server_port = int(configured_port)
        except ValueError as error:
            raise ValueError(
                f"GRADIO_SERVER_PORT/PORT must be an integer, got {configured_port!r}"
            ) from error

        if not 1 <= server_port <= 65535:
            raise ValueError("GRADIO_SERVER_PORT/PORT must be between 1 and 65535")

        model_name = os.environ.get(
            "MODEL_NAME",
            os.environ.get("VLLM_MODEL", "deepseek-v4-flash"),
        ).strip()
        if not model_name:
            raise ValueError("MODEL_NAME must not be empty")

        return cls(
            base_url=normalize_base_url(
                os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
            ),
            api_key=_read_api_key(),
            model_name=model_name,
            chat_completions_path=os.environ.get(
                "CHAT_COMPLETIONS_PATH", "/v1/chat/completions"
            ),
            models_path=os.environ.get("MODELS_PATH", "/v1/models"),
            server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
            server_port=server_port,
            root_path=os.environ.get("GRADIO_ROOT_PATH", "").strip(),
            request_timeout_seconds=_read_positive_integer(
                "BACKEND_REQUEST_TIMEOUT_SECONDS", 900
            ),
            connect_timeout_seconds=_read_positive_integer(
                "BACKEND_CONNECT_TIMEOUT_SECONDS", 10
            ),
            maximum_context_tokens=_read_positive_integer(
                "MAX_CONTEXT_TOKENS", 524_288
            ),
            context_probe_safety_tokens=_read_positive_integer(
                "CONTEXT_PROBE_SAFETY_TOKENS", 4_096
            ),
            inference_concurrency_limit=_read_positive_integer(
                "INFERENCE_CONCURRENCY_LIMIT", 1
            ),
            queue_max_size=_read_positive_integer("QUEUE_MAX_SIZE", 32),
        )

    @property
    def chat_completions_url(self) -> str:
        return join_api_path(self.base_url, self.chat_completions_path)

    @property
    def models_url(self) -> str:
        return join_api_path(self.base_url, self.models_path)

    @property
    def health_url(self) -> str:
        return join_root_path(self.base_url, "/health")

    @property
    def tokenize_url(self) -> str:
        return join_root_path(self.base_url, "/tokenize")
