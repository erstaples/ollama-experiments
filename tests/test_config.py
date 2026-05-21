import os
from pathlib import Path
from unittest.mock import patch

from harness.config import Config


def test_config_defaults():
    with patch.dict(os.environ, {}, clear=True):
        c = Config.from_env()
    assert c.ollama_host == "http://localhost:11434"
    assert c.model == "qwen3-coder-32k:latest"
    assert c.temperature == 0.7
    assert c.num_ctx == 32768
    assert c.workspace.name == "workspace"
    assert c.max_tool_iterations == 10
    assert c.shell_timeout_sec == 30


def test_config_env_overrides():
    env = {
        "OLLAMA_HOST": "http://example:11434",
        "OLLAMA_MODEL": "llama3:8b",
        "TEMPERATURE": "0.2",
        "NUM_CTX": "8192",
        "MAX_TOOL_ITERATIONS": "5",
        "SHELL_TIMEOUT_SEC": "60",
    }
    with patch.dict(os.environ, env, clear=True):
        c = Config.from_env()
    assert c.ollama_host == "http://example:11434"
    assert c.model == "llama3:8b"
    assert c.temperature == 0.2
    assert c.num_ctx == 8192
    assert c.max_tool_iterations == 5
    assert c.shell_timeout_sec == 60


def test_config_cli_overrides_env(tmp_path):
    env = {"OLLAMA_MODEL": "from-env:latest", "TEMPERATURE": "0.5"}
    with patch.dict(os.environ, env, clear=True):
        c = Config.from_env(model="from-cli:latest", workspace=tmp_path)
    assert c.model == "from-cli:latest"
    assert c.temperature == 0.5
    assert c.workspace == tmp_path


def test_workspace_is_resolved_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch.dict(os.environ, {}, clear=True):
        c = Config.from_env()
    assert c.workspace.is_absolute()
