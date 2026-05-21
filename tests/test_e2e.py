"""End-to-end test against a live Ollama server.

Gated behind OLLAMA_E2E=1 because it requires:
  - `ollama serve` running locally
  - the qwen3-coder-32k:latest model available
"""

import os
from pathlib import Path

import pytest

from harness.agent import Agent, ToolCallStart, TurnComplete
from harness.config import Config
from harness.ollama_client import OllamaClient
from harness.tools import build_registry


pytestmark = pytest.mark.skipif(
    os.environ.get("OLLAMA_E2E") != "1",
    reason="set OLLAMA_E2E=1 to run live Ollama tests",
)


def test_live_tool_call(tmp_path):
    (tmp_path / "hello.txt").write_text("world")
    config = Config.from_env(workspace=tmp_path)
    client = OllamaClient(
        host=config.ollama_host,
        model=config.model,
        temperature=config.temperature,
        num_ctx=config.num_ctx,
    )
    registry = build_registry(tmp_path, shell_timeout_sec=10)
    agent = Agent(client=client, registry=registry, max_iterations=5)

    events = list(agent.run_turn("List the files in the workspace."))

    starts = [e for e in events if isinstance(e, ToolCallStart)]
    assert any(s.name == "list_directory" for s in starts), \
        f"expected a list_directory call, got: {[s.name for s in starts]}"
    assert isinstance(events[-1], TurnComplete)
