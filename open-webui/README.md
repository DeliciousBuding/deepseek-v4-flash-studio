# Open WebUI (full mode)

Open WebUI is the rich chat UX shell used by the optional "full mode" stack. It
is configured entirely through environment variables in `../docker-compose.yml`
(`OPENAI_API_BASE_URL`, `OPENAI_API_KEY`, `DEFAULT_MODELS`, …) — no source fork
is maintained here, so upstream upgrades stay cheap.

The default "lite mode" (`python app.py`) is the reliable fallback used by
ModelScope Studio and does not require this directory.
