from __future__ import annotations

from pathlib import Path

from harness.registry import ToolRegistry
from harness.tools.filesystem import make_filesystem_tools
from harness.tools.shell import make_shell_tool
from harness.tools.web import make_web_tool


def build_registry(workspace: Path, shell_timeout_sec: int = 30) -> ToolRegistry:
    registry = ToolRegistry()

    fs_tools = make_filesystem_tools(workspace)
    for func in fs_tools.values():
        registry.register(func)

    registry.register(make_shell_tool(workspace, default_timeout_sec=shell_timeout_sec))
    registry.register(make_web_tool())

    return registry


__all__ = ["build_registry"]
