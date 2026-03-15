# SGLang Custom Generation Service

This repository exposes a small FastAPI service that runs custom generation logic through SGLang.

## What it does

- Starts an SGLang runtime for a model specified with environment variables.
- Exposes a `POST /generate` endpoint backed by an `@sgl.function`.
- Loads configuration from environment variables and an optional `.env` file.

## Setup

1. Sync the project environment with `uv`.
2. Copy `.env.example` to `.env`.
3. Set `MODEL_PATH` to the model you want to serve.

Example install:

```bash
uv sync
cp .env.example .env
```

## Configuration

The service reads these environment variables:

- `MODEL_PATH`: required when `MANAGE_RUNTIME=true`; passed to `python -m sglang.launch_server`.
- `MANAGE_RUNTIME`: when `true`, `main.py` launches the SGLang runtime; when `false`, it connects to an existing runtime.
- `SGLANG_RUNTIME_HOST` and `SGLANG_RUNTIME_PORT`: host and port for the runtime.
- `APP_HOST` and `APP_PORT`: host and port for the FastAPI app.
- `TENSOR_PARALLEL_SIZE`: tensor parallelism passed to the runtime.
- `MEM_FRACTION_STATIC`: static memory fraction passed to the runtime.
- `RUNTIME_STARTUP_TIMEOUT`: seconds to wait for the runtime health check.
- `DTYPE`: runtime dtype flag, for example `auto`, `float16`, or `bfloat16`.
- `SYSTEM_PROMPT`: default system prompt for generation.
- `MAX_TOKENS`: default max generation length.
- `TEMPERATURE`: default sampling temperature.

## Run

```bash
uv run python main.py
```

When the service is up, call it like this:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain what SGLang is in two sentences.",
    "system_prompt": "You are a helpful assistant.",
    "max_tokens": 128,
    "temperature": 0.4
  }'
```

Health check:

```bash
curl http://127.0.0.1:8000/healthz
```

## Notes

- `MODEL_PATH` is intentionally not hardcoded in the application.
- If you already run an SGLang runtime elsewhere, set `MANAGE_RUNTIME=false` and point the runtime host and port to that server.

## Troubleshoots

If you see an error or warning which says pytorch incompatible then run this command. Make sure the Read the error message
SGLANG_DISABLE_CUDNN_CHECK=1 uv run python main.py
