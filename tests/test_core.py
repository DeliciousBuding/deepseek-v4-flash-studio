from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepseek_lab.backend import (
    BackendRequestError,
    CompletionChunk,
    parse_server_sent_event,
)
from deepseek_lab.config import AppConfig, join_api_path, join_root_path
from deepseek_lab.probes import build_context_probe_prompt, extract_cached_tokens
from deepseek_lab.ui import stream_chat


def create_test_config(**overrides: object) -> AppConfig:
    config_values: dict[str, object] = {
        "base_url": "http://127.0.0.1:8000",
        "api_key": "",
        "model_name": "deepseek-v4-flash",
        "chat_completions_path": "/v1/chat/completions",
        "models_path": "/v1/models",
        "server_name": "127.0.0.1",
        "server_port": 7860,
        "root_path": "",
        "request_timeout_seconds": 900,
        "connect_timeout_seconds": 10,
        "maximum_context_tokens": 524_288,
        "context_probe_safety_tokens": 4_096,
        "inference_concurrency_limit": 1,
        "queue_max_size": 32,
    }
    config_values.update(overrides)
    return AppConfig(**config_values)  # type: ignore[arg-type]


class FakeBackend:
    def __init__(self, chunks: list[CompletionChunk] | None = None):
        self.chunks = chunks or []
        self.received_messages: list[dict[str, object]] = []

    def count_tokens(self, prompt: str) -> int:
        return max(len(prompt) // 4, 1)

    def stream_chat_completion(
        self,
        messages: list[dict[str, object]],
        temperature: float,
        maximum_output_tokens: int,
    ):
        self.received_messages = messages
        yield from self.chunks


class FailingBackend(FakeBackend):
    def stream_chat_completion(
        self,
        messages: list[dict[str, object]],
        temperature: float,
        maximum_output_tokens: int,
    ):
        self.received_messages = messages
        raise BackendRequestError("test backend failure")
        yield


class ConfigurationTests(unittest.TestCase):
    def test_api_paths_accept_base_url_with_or_without_v1(self) -> None:
        expected_url = "https://example.test/proxy/v1/chat/completions"
        self.assertEqual(
            join_api_path(
                "https://example.test/proxy",
                "/v1/chat/completions",
            ),
            expected_url,
        )
        self.assertEqual(
            join_api_path(
                "https://example.test/proxy/v1",
                "/v1/chat/completions",
            ),
            expected_url,
        )

    def test_root_path_removes_only_trailing_v1(self) -> None:
        self.assertEqual(
            join_root_path("https://example.test/proxy/v1", "/tokenize"),
            "https://example.test/proxy/tokenize",
        )

    def test_api_key_file_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            key_file_path = Path(temporary_directory) / "api-key"
            key_file_path.write_text("file-secret\n", encoding="utf-8")
            with patch.dict(
                "os.environ",
                {
                    "VLLM_BASE_URL": "http://127.0.0.1:8000",
                    "VLLM_API_KEY_FILE": str(key_file_path),
                },
                clear=True,
            ):
                config = AppConfig.from_environment()
        self.assertEqual(config.api_key, "file-secret")


class StreamingParserTests(unittest.TestCase):
    def test_parser_extracts_reasoning_content_and_usage(self) -> None:
        chunk = parse_server_sent_event(
            'data: {"choices":[{"delta":{"content":"answer",'
            '"reasoning_content":"hidden"}}],"usage":{"completion_tokens":1}}'
        )
        self.assertIsNotNone(chunk)
        assert chunk is not None
        self.assertEqual(chunk.content, "answer")
        self.assertEqual(chunk.reasoning, "hidden")
        self.assertEqual(chunk.usage, {"completion_tokens": 1})

    def test_parser_ignores_keepalive_and_done_events(self) -> None:
        self.assertIsNone(parse_server_sent_event(": keep-alive"))
        self.assertIsNone(parse_server_sent_event("data: [DONE]"))


class ConversationTests(unittest.TestCase):
    def test_display_reasoning_is_not_replayed_to_backend_history(self) -> None:
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
        backend = FakeBackend(
            [
                CompletionChunk(
                    content="Visible answer",
                    reasoning="<script>hidden</script>",
                    usage=usage,
                )
            ]
        )

        callback_results = list(
            stream_chat(
                "Question",
                [],
                [],
                0.0,
                64,
                True,
                backend,  # type: ignore[arg-type]
            )
        )
        display_history, api_conversation, metrics = callback_results[-1]

        self.assertIn("&lt;script&gt;hidden&lt;/script&gt;", display_history[-1]["content"])
        self.assertEqual(api_conversation[-1]["content"], "Visible answer")
        self.assertNotIn("Reasoning", api_conversation[-1]["content"])
        self.assertIn("cached `4`", metrics)

    def test_failed_turn_does_not_commit_api_history(self) -> None:
        existing_api_history = [
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
        ]
        callback_results = list(
            stream_chat(
                "Failing question",
                [],
                existing_api_history,
                0.0,
                64,
                False,
                FailingBackend(),  # type: ignore[arg-type]
            )
        )
        _display_history, api_conversation, _metrics = callback_results[-1]
        self.assertEqual(api_conversation, existing_api_history)


class ContextProbeTests(unittest.TestCase):
    def test_probe_leaves_room_for_output_and_framing(self) -> None:
        config = create_test_config(
            maximum_context_tokens=10_000,
            context_probe_safety_tokens=1_000,
        )
        prompt = build_context_probe_prompt(
            FakeBackend(),  # type: ignore[arg-type]
            config,
            requested_tokens=9_500,
            maximum_output_tokens=500,
        )

        self.assertLessEqual(prompt.estimated_tokens, 8_500)
        self.assertEqual(prompt.requested_tokens, 9_500)

    def test_cached_token_extraction_handles_null_details(self) -> None:
        self.assertEqual(
            extract_cached_tokens({"prompt_tokens_details": None}),
            0,
        )


if __name__ == "__main__":
    unittest.main()
