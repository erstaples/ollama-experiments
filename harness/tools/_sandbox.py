from __future__ import annotations

from pathlib import Path


class SandboxError(Exception):
    """Raised when a path resolves outside the workspace."""


def resolve_in_workspace(path: str, workspace: Path) -> Path:
    """Resolve `path` against `workspace` and ensure it stays inside.

    Catches `..` traversal, absolute paths outside the root, and symlink
    escapes because `.resolve()` follows symlinks before the relative check.
    """
    workspace_resolved = workspace.resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = workspace_resolved / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace_resolved)
    except ValueError:
        raise SandboxError(f"path escapes workspace: {path}") from None
    return resolved
