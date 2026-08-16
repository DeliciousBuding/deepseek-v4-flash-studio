# Deployment

Three supported targets, from simplest to richest:

## 1. Local lite mode (Gradio)

Prereqs: a running OpenAI-compatible backend (see the sibling repo
`deepseek-v4-flash-mi308x` for how to start vLLM on MI308X).

```bash
cp .env.example .env && vim .env          # set VLLM_BASE_URL, VLLM_API_KEY
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:7860`.

## 2. ModelScope 创空间 (Studio)

1. Push this repository to a public GitHub repo.
2. In ModelScope Studio, create an application: select the **AMD MI308X** GPU
   instance and the ROCm image, point it at this repo, entry file `app.py`,
   requirements from `requirements.txt`.
3. Configure the environment variables (`VLLM_BASE_URL`, `VLLM_API_KEY`,
   `MODEL_NAME`) in the application settings.
4. Run `python app.py`.

See [`../studio/modelscope.md`](../studio/modelscope.md) for the full deployment
steps and environment variables.

## 3. Full mode (Open WebUI + LiteLLM)

```bash
# Backend must be reachable from the Docker host.
VLLM_BASE_URL=http://host.docker.internal:8000 docker compose up
```

- Open WebUI: `http://127.0.0.1:3000`
- LiteLLM: `http://127.0.0.1:4000`

To point the whole stack at a remote backend instead, override `VLLM_BASE_URL`
with the public URL of your OpenAI-compatible endpoint.

## Health checks

```bash
bash studio/healthcheck.sh
```

The UI must respond on `GRADIO_SERVER_PORT` and, when reachable, the backend
`/health` should return 200.
