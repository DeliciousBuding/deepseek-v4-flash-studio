# Deployment

The application and gateway can run together or on separate hosts.

## 1. Local Gradio application

Prereqs: a running OpenAI-compatible backend (see the sibling repo
`deepseek-v4-flash-mi308x` for how to start vLLM on MI308X).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && vim .env          # set OPENAI_BASE_URL, OPENAI_API_KEY
set -a && source .env && set +a
python app.py
```

Open `http://127.0.0.1:7860`.

## 2. ModelScope 创空间 (Studio)

The application supports two explicit topologies:

- **Remote gateway:** deploy this UI on a CPU application and point
  `OPENAI_BASE_URL` at an authenticated, reachable LiteLLM service.
- **Co-located backend:** select an AMD MI308X/ROCm application, start the
  sibling `deepseek-v4-flash-mi308x` serving recipe and LiteLLM first, then
  start this UI against `http://127.0.0.1:4000/v1`.

The default `python app.py` command starts only the UI; it never downloads model
weights or launches vLLM implicitly.

1. Configure the repository with entry file `app.py` and dependencies from
   `requirements.txt`.
2. Inject `OPENAI_BASE_URL`, `MODEL_NAME`, and either `OPENAI_API_KEY` or a
   mounted `OPENAI_API_KEY_FILE` through the platform settings.
3. Set `MAX_CONTEXT_TOKENS` to the backend's actual context ceiling.
4. Run `python app.py` or `bash studio/start.sh`.

See [`../studio/modelscope.md`](../studio/modelscope.md) for the full deployment
steps and environment variables.

## 3. LiteLLM gateway

```bash
cp .env.example .env
# Configure the native backend and a random LITELLM_MASTER_KEY.
docker compose up -d
```

- LiteLLM: `http://127.0.0.1:4000`

The published port binds to `127.0.0.1`. Add an authenticated TLS reverse proxy
before exposing it to another network. Linux Compose receives an explicit
`host.docker.internal:host-gateway` mapping for a host-native vLLM process.

## Health checks

```bash
bash studio/healthcheck.sh
```

The check requires both the UI and authenticated backend model list by default.
Set `REQUIRE_BACKEND_HEALTH=0` only for a deliberately UI-only readiness check.

## Verification

```bash
python -m compileall -q app.py deepseek_lab tests
python -m unittest discover -s tests -t . -v
bash -n studio/start.sh studio/healthcheck.sh
```
