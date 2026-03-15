from __future__ import annotations

import logging
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import sglang as sgl
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint

from config import Settings, load_settings


logger = logging.getLogger("sglang-service")


@sgl.function
def custom_generation(
    s: sgl.SglGen,
    user_prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
) -> None:
    s += sgl.system(system_prompt)
    s += sgl.user(user_prompt)
    s += sgl.assistant(
        sgl.gen(
            "answer",
            max_tokens=max_tokens,
            temperature=temperature,
        )
    )


class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User prompt to send to the model.")
    system_prompt: str | None = Field(
        default=None,
        description="Optional system prompt. Falls back to SYSTEM_PROMPT from the environment.",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Optional generation length override.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional sampling temperature override.",
    )


class GenerationResponse(BaseModel):
    text: str
    model_path: str
    runtime_url: str


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def build_runtime_command(settings: Settings) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        settings.model_path,
        "--host",
        settings.runtime_host,
        "--port",
        str(settings.runtime_port),
        "--tp-size",
        str(settings.tensor_parallel_size),
        "--mem-fraction-static",
        str(settings.mem_fraction_static),
    ]
    if settings.dtype:
        command.extend(["--dtype", settings.dtype])
    return command


def wait_for_runtime(settings: Settings) -> None:
    deadline = time.time() + settings.runtime_startup_timeout
    last_error: Exception | None = None

    while time.time() < deadline:
        try:
            with urlopen(settings.runtime_healthcheck_url, timeout=2) as response:
                if response.status < 500:
                    return
        except (OSError, URLError) as exc:
            last_error = exc
            time.sleep(1)

    raise RuntimeError(
        f"SGLang runtime did not become ready within {settings.runtime_startup_timeout}s."
    ) from last_error


def stop_runtime(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    runtime_process: subprocess.Popen[Any] | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal runtime_process

        if settings.manage_runtime:
            logger.info("Starting SGLang runtime for model %s", settings.model_path)
            runtime_process = subprocess.Popen(build_runtime_command(settings))
            try:
                wait_for_runtime(settings)
            except Exception:
                stop_runtime(runtime_process)
                raise
        else:
            logger.info("Using external SGLang runtime at %s", settings.runtime_url)
            wait_for_runtime(settings)

        backend = RuntimeEndpoint(settings.runtime_url)
        sgl.set_default_backend(backend)
        logger.info("SGLang backend configured at %s", settings.runtime_url)

        try:
            yield
        finally:
            stop_runtime(runtime_process)

    app = FastAPI(
        title="SGLang Custom Generation Service",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate", response_model=GenerationResponse)
    async def generate(request: GenerationRequest) -> GenerationResponse:
        try:
            state = custom_generation.run(
                user_prompt=request.prompt,
                system_prompt=request.system_prompt or settings.system_prompt,
                max_tokens=request.max_tokens or settings.max_tokens,
                temperature=request.temperature or settings.temperature,
            )
        except Exception as exc:
            logger.exception("Generation failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return GenerationResponse(
            text=state["answer"],
            model_path=settings.model_path,
            runtime_url=settings.runtime_url,
        )

    return app


def main() -> None:
    configure_logging()
    settings = load_settings()
    app = create_app(settings)

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    uvicorn.run(app, host=settings.app_host, port=settings.app_port)


if __name__ == "__main__":
    main()
