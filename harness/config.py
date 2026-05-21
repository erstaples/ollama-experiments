from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    ollama_host: str
    model: str
    temperature: float
    num_ctx: int
    workspace: Path
    max_tool_iterations: int
    shell_timeout_sec: int

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        temperature: float | None = None,
        num_ctx: int | None = None,
        workspace: Path | None = None,
    ) -> Config:
        ws = workspace or Path(os.environ.get("WORKSPACE_ROOT", "./workspace"))
        return cls(
            ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            model=model or os.environ.get("OLLAMA_MODEL", "qwen3-coder-32k:latest"),
            temperature=temperature if temperature is not None else float(os.environ.get("TEMPERATURE", "0.7")),
            num_ctx=num_ctx if num_ctx is not None else int(os.environ.get("NUM_CTX", "32768")),
            workspace=ws.resolve(),
            max_tool_iterations=int(os.environ.get("MAX_TOOL_ITERATIONS", "10")),
            shell_timeout_sec=int(os.environ.get("SHELL_TIMEOUT_SEC", "30")),
        )
