# ModelScope 创空间（Studio）部署说明

本目录提供 ModelScope 创空间的技术部署入口。`app.py` 只启动 Gradio UI，
通过 OpenAI-compatible API 访问 DeepSeek-V4-Flash 推理服务；它不会隐式下载
权重、安装 ROCm/vLLM 或启动模型进程。

## 拓扑选择

### 远端网关

UI 使用 CPU 资源即可。把 `OPENAI_BASE_URL` 指向已通过 TLS/鉴权暴露的
LiteLLM 网关，并通过平台 secret 注入 API key。

### 同机后端

选择 AMD MI308X + ROCm 环境，先按公开仓
`deepseek-v4-flash-mi308x` 启动 vLLM，再启动 LiteLLM 和本仓 `app.py`。同机
默认网关地址是 `http://127.0.0.1:4000/v1`。推理参数和模型生命周期归
serving 仓；LiteLLM 统一认证和 OpenAI API；UI 只消费网关。

## 部署配置

| 项 | 值 |
|---|---|
| 入口文件 | `app.py` |
| 依赖 | `requirements.txt`（gradio + httpx） |
| 运行命令 | `python app.py` |
| 资源 | 远端后端用 CPU；同机后端用 AMD MI308X + ROCm |
| 后端 | 已运行的 LiteLLM OpenAI-compatible API（`OPENAI_BASE_URL`） |

## 环境变量

- `OPENAI_BASE_URL`：LiteLLM OpenAI 接口地址（同实例一般为 `http://127.0.0.1:4000/v1`）。
- `OPENAI_API_KEY`：网关 Bearer key，优先于 key file。
- `OPENAI_API_KEY_FILE`：从平台挂载的 secret 文件读取 Bearer key。
- `VLLM_BASE_URL` / `VLLM_API_KEY` / `VLLM_API_KEY_FILE`：兼容旧直连部署的回退变量。
- `MODEL_NAME`：默认 `deepseek-v4-flash`，须与后端 `--served-model-name` 一致。
- `MAX_CONTEXT_TOKENS`：默认 `524288`，用于限制长上下文探针，不应高于后端。
- `INFERENCE_CONCURRENCY_LIMIT`：默认 `1`，Chat 与长上下文探针共享此并发限制。
- `GRADIO_ROOT_PATH`：仅在创空间反向代理使用 URL 前缀时设置。

## 发布前自检

- [ ] `python app.py` 能起、页面能开。
- [ ] Chat 能流式返回，推理块折叠正常。
- [ ] 长上下文探针 32K/128K 能跑完并回显实际 token、TTFT 和缓存 token。
- [ ] `bash studio/healthcheck.sh` 同时验证 UI 与鉴权后的 `/v1/models`。
- [ ] README 写清楚如何复现后端（指向 `deepseek-v4-flash-mi308x`）。
