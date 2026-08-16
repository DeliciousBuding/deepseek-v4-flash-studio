#!/usr/bin/env python3
"""Launch the MI308X DeepSeek Lab Gradio application."""

from deepseek_lab.backend import OpenAIBackend
from deepseek_lab.config import AppConfig
from deepseek_lab.ui import build_demo


def main() -> None:
    config = AppConfig.from_environment()
    backend = OpenAIBackend(config)
    demo = build_demo(config, backend)
    demo.queue(
        default_concurrency_limit=config.inference_concurrency_limit,
        max_size=config.queue_max_size,
    ).launch(
        server_name=config.server_name,
        server_port=config.server_port,
        root_path=config.root_path or None,
    )


if __name__ == "__main__":
    main()
