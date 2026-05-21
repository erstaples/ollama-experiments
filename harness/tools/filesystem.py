from __future__ import annotations

from pathlib import Path
from typing import Callable

from harness.tools._sandbox import SandboxError, resolve_in_workspace


def make_filesystem_tools(workspace: Path) -> dict[str, Callable]:
    """Return a dict {tool_name: function} bound to the given workspace."""

    def list_directory(path: str = ".") -> list[str] | dict:
        """List the names of files and directories under a workspace-relative path.

        Returns a sorted list of basenames. Use this to explore the workspace
        before reading specific files.
        """
        try:
            target = resolve_in_workspace(path, workspace)
        except SandboxError as exc:
            return {"error": str(exc)}
        if not target.exists():
            return {"error": f"path does not exist: {path}"}
        if not target.is_dir():
            return {"error": f"not a directory: {path}"}
        return sorted(p.name for p in target.iterdir())

    def read_file(path: str, max_bytes: int = 100_000) -> str | dict:
        """Read the UTF-8 contents of a workspace file.

        Contents larger than max_bytes are truncated and an explicit marker
        is appended so the model knows the file was cut off.
        """
        try:
            target = resolve_in_workspace(path, workspace)
        except SandboxError as exc:
            return {"error": str(exc)}
        if not target.exists():
            return {"error": f"file does not exist: {path}"}
        if not target.is_file():
            return {"error": f"not a file: {path}"}
        data = target.read_bytes()
        if len(data) > max_bytes:
            text = data[:max_bytes].decode("utf-8", errors="replace")
            return f"{text}\n\n[truncated: {len(data) - max_bytes} more bytes]"
        return data.decode("utf-8", errors="replace")

    def write_file(path: str, content: str, append: bool = False) -> dict:
        """Write or append UTF-8 content to a workspace file.

        Creates parent directories as needed. Returns the absolute path
        written and the number of bytes written.
        """
        try:
            target = resolve_in_workspace(path, workspace)
        except SandboxError as exc:
            return {"error": str(exc)}
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with target.open(mode, encoding="utf-8") as f:
            written = f.write(content)
        return {"path": str(target), "bytes_written": written}

    return {
        "list_directory": list_directory,
        "read_file": read_file,
        "write_file": write_file,
    }
