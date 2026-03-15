from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model_path: str
    runtime_host: str
    runtime_port: int
    app_host: str
    app_port: int
    tensor_parallel_size: int
    mem_fraction_static: float
    runtime_startup_timeout: int
    manage_runtime: bool
    dtype: str
    system_prompt: str
    max_tokens: int
    temperature: float

    @property
    def runtime_url(self) -> str:
        return f"http://{self.runtime_host}:{self.runtime_port}"

    @property
    def runtime_healthcheck_url(self) -> str:
        return f"{self.runtime_url}/health"


def load_settings() -> Settings:
    model_path = os.getenv("MODEL_PATH", "").strip()
    manage_runtime = _get_bool("MANAGE_RUNTIME", True)

    if manage_runtime and not model_path:
        raise ValueError("MODEL_PATH must be set when MANAGE_RUNTIME=true.")

    if not model_path:
        model_path = os.getenv("EXTERNAL_MODEL_NAME", "external-runtime")

    return Settings(
        model_path=model_path,
        runtime_host=os.getenv("SGLANG_RUNTIME_HOST", "127.0.0.1"),
        runtime_port=int(os.getenv("SGLANG_RUNTIME_PORT", "30000")),
        app_host=os.getenv("APP_HOST", "0.0.0.0"),
        app_port=int(os.getenv("APP_PORT", "8000")),
        tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
        mem_fraction_static=float(os.getenv("MEM_FRACTION_STATIC", "0.8")),
        runtime_startup_timeout=int(os.getenv("RUNTIME_STARTUP_TIMEOUT", "180")),
        manage_runtime=manage_runtime,
        dtype=os.getenv("DTYPE", "auto"),
        system_prompt=os.getenv(
            "SYSTEM_PROMPT",
            "You are a concise assistant operating through an SGLang custom app.",
        ),
        max_tokens=int(os.getenv("MAX_TOKENS", "256")),
        temperature=float(os.getenv("TEMPERATURE", "0.7")),
    )
