# ModelScope 创空间（Studio）部署说明

本目录是 ModelScope 创空间的打包入口：把 `app.py` 作为可交互入口，展示
DeepSeek-V4-Flash 在单卡 AMD MI308X 上的长上下文推理与性能指标。

## 部署配置

| 项 | 值 |
|---|---|
| 入口文件 | `app.py` |
| 依赖 | `requirements.txt`（gradio + httpx） |
| 运行命令 | `python app.py` |
| 资源 | AMD MI308X（192GB）单卡 + ROCm 镜像 |
| 后端 | 同实例 vLLM OpenAI-compatible API（`VLLM_BASE_URL`） |

## 环境变量

- `VLLM_BASE_URL`：vLLM OpenAI 接口地址（同实例一般为 `http://127.0.0.1:8000`）。
- `VLLM_API_KEY`：若后端开启 `--api-key`，填对应 Bearer key；否则留空。
- `MODEL_NAME`：默认 `deepseek-v4-flash`，须与后端 `--served-model-name` 一致。

## 发布前自检

- [ ] `python app.py` 能起、页面能开。
- [ ] Chat 能流式返回，推理块折叠正常。
- [ ] 长上下文测试 Tab 32K/128K 能跑完并回显 TTFT/tok/s。
- [ ] 后端 `/health` 200。
- [ ] README 写清楚如何复现后端（指向 `deepseek-v4-flash-mi308x`）。
