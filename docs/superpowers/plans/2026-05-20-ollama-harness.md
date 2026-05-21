# Ollama Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python harness that runs local LLMs via Ollama with native tool-calling, exposing filesystem/shell/web tools through both a terminal REPL and a Streamlit chat UI.

**Architecture:** Pure `Agent` class drives the tool-calling loop and yields typed events. Two thin front-ends (REPL and Streamlit) consume the same event stream. All filesystem and shell tools are sandboxed to a workspace directory.

**Tech Stack:** Python 3.11+, `uv` for package management, `ollama` SDK, `ddgs`, `rich`, `streamlit`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-05-20-ollama-harness-design.md`

---

## File Structure

```
ollama-experiments/
├── pyproject.toml
├── .gitignore
├── README.md
├── harness/
│   ├── __init__.py
│   ├── __main__.py             # dispatches to REPL or Streamlit
│   ├── config.py               # Config dataclass (env + CLI)
│   ├── registry.py             # ToolRegistry + schema introspection
│   ├── ollama_client.py        # ollama.Client wrapper, streaming
│   ├── agent.py                # Event types + Agent.run_turn
│   ├── ui_repl.py              # terminal front-end
│   ├── ui_streamlit.py         # Streamlit page
│   └── tools/
│       ├── __init__.py         # build_registry() — wires all tools
│       ├── _sandbox.py         # resolve_in_workspace() helper
│       ├── filesystem.py       # list_directory, read_file, write_file
│       ├── shell.py            # run_shell
│       └── web.py              # search_web
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_registry.py
│   ├── test_sandbox.py
│   ├── test_tools_filesystem.py
│   ├── test_tools_shell.py
│   ├── test_ollama_client.py
│   ├── test_agent_events.py
│   └── test_e2e.py             # gated by OLLAMA_E2E=1
├── workspace/.gitkeep          # sandbox root (contents gitignored)
└── docs/superpowers/{specs,plans}/
```

**Key boundaries:**
- `Agent` knows nothing about UIs — only emits events.
- Tool modules export plain Python functions; `tools/__init__.py:build_registry()` collects and registers them. No module-level side effects.
- `_sandbox.py` is the single source of truth for path containment.

---

## Task 1: Project bootstrap

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md` (skeleton)
- Create: `workspace/.gitkeep`
- Create: `harness/__init__.py` (empty)
- Create: `harness/tools/__init__.py` (empty for now)
- Create: `tests/__init__.py` (empty)

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "ollama-experiments"
version = "0.1.0"
description = "Local LLM harness with tool-calling, REPL + Streamlit UIs"
requires-python = ">=3.11"
dependencies = [
    "ollama>=0.4.0",
    "ddgs>=6.0.0",
    "rich>=13.0.0",
    "streamlit>=1.30.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-mock>=3.12.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["harness"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 2: Create .gitignore**

```
# Python
__pycache__/
*.py[cod]
.venv/
*.egg-info/

# uv
.uv-cache/

# Tests
.pytest_cache/
.coverage
htmlcov/

# Editors
.vscode/
.idea/
*.swp

# Workspace (sandboxed agent ops — never commit contents)
workspace/*
!workspace/.gitkeep

# OS
.DS_Store
```

- [ ] **Step 3: Create workspace/.gitkeep and empty __init__.py files**

```bash
touch workspace/.gitkeep
touch harness/__init__.py
touch harness/tools/__init__.py
touch tests/__init__.py
```

- [ ] **Step 4: Create README.md skeleton**

```markdown
# ollama-experiments

A local-LLM harness for experimenting with Ollama models, native tool-calling, and agent loops.

See `docs/superpowers/specs/2026-05-20-ollama-harness-design.md` for full design.

## Quick start

```bash
uv sync
ollama serve  # in another terminal
uv run python -m harness        # REPL
uv run python -m harness --ui streamlit  # browser UI
```
```

- [ ] **Step 5: Install deps and verify**

Run: `uv sync && uv run python -c "import ollama, ddgs, rich, streamlit; print('ok')"`
Expected output ends with: `ok`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore README.md workspace/.gitkeep harness/__init__.py harness/tools/__init__.py tests/__init__.py
git commit -m "chore: project bootstrap with uv, deps, and structure"
```

---

## Task 2: Config

**Files:**
- Create: `harness/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
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
    assert c.temperature == 0.5  # not overridden
    assert c.workspace == tmp_path


def test_workspace_is_resolved_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch.dict(os.environ, {}, clear=True):
        c = Config.from_env()
    assert c.workspace.is_absolute()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness.config'`

- [ ] **Step 3: Implement Config**

Create `harness/config.py`:

```python
"""Configuration loaded from env vars with optional CLI overrides."""

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add harness/config.py tests/test_config.py
git commit -m "feat(config): Config dataclass with env + CLI overrides"
```

---

## Task 3: Tool registry

**Files:**
- Create: `harness/registry.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write failing tests for basic registration and schema**

Create `tests/test_registry.py`:

```python
import pytest

from harness.registry import ToolRegistry


def test_register_simple_function():
    r = ToolRegistry()

    @r.tool
    def greet(name: str) -> str:
        """Say hello to someone."""
        return f"hello, {name}"

    schemas = r.schemas()
    assert len(schemas) == 1
    schema = schemas[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "greet"
    assert schema["function"]["description"] == "Say hello to someone."
    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert params["properties"]["name"] == {"type": "string"}
    assert params["required"] == ["name"]


def test_default_arg_is_not_required():
    r = ToolRegistry()

    @r.tool
    def list_dir(path: str = ".") -> list[str]:
        """List a directory."""
        return []

    params = r.schemas()[0]["function"]["parameters"]
    assert params["properties"]["path"] == {"type": "string"}
    assert params["required"] == []


def test_supported_scalar_types():
    r = ToolRegistry()

    @r.tool
    def mixed(s: str, i: int, f: float, b: bool) -> str:
        """Mixed scalars."""
        return ""

    props = r.schemas()[0]["function"]["parameters"]["properties"]
    assert props["s"] == {"type": "string"}
    assert props["i"] == {"type": "integer"}
    assert props["f"] == {"type": "number"}
    assert props["b"] == {"type": "boolean"}


def test_list_type():
    r = ToolRegistry()

    @r.tool
    def join(parts: list[str]) -> str:
        """Join parts."""
        return ""

    props = r.schemas()[0]["function"]["parameters"]["properties"]
    assert props["parts"] == {"type": "array", "items": {"type": "string"}}


def test_optional_type():
    r = ToolRegistry()

    @r.tool
    def maybe(name: str | None = None) -> str:
        """Maybe name."""
        return ""

    params = r.schemas()[0]["function"]["parameters"]
    assert params["properties"]["name"] == {"type": "string"}
    assert params["required"] == []


def test_unsupported_type_raises():
    r = ToolRegistry()
    with pytest.raises(TypeError, match="unsupported"):
        @r.tool
        def bad(thing: dict) -> str:
            """No."""
            return ""


def test_missing_docstring_raises():
    r = ToolRegistry()
    with pytest.raises(ValueError, match="docstring"):
        @r.tool
        def undocumented(x: str) -> str:
            return x


def test_dispatch_calls_function():
    r = ToolRegistry()

    @r.tool
    def upper(text: str) -> str:
        """Uppercase."""
        return text.upper()

    result = r.dispatch("upper", {"text": "hi"})
    assert result == "HI"


def test_dispatch_unknown_tool_returns_error_dict():
    r = ToolRegistry()
    result = r.dispatch("nope", {})
    assert result == {"error": "no such tool: nope"}


def test_dispatch_catches_function_exception():
    r = ToolRegistry()

    @r.tool
    def boom(x: int) -> int:
        """Explodes."""
        raise RuntimeError("boom")

    result = r.dispatch("boom", {"x": 1})
    assert result == {"error": "boom"}


def test_register_function_directly_without_decorator():
    """Tools can also be registered via .register(func) — used by build_registry."""
    r = ToolRegistry()

    def hello(name: str) -> str:
        """Hello."""
        return f"hi {name}"

    r.register(hello)
    assert r.dispatch("hello", {"name": "x"}) == "hi x"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness.registry'`

- [ ] **Step 3: Implement ToolRegistry**

Create `harness/registry.py`:

```python
"""Tool registry: introspects typed Python functions into Ollama tool schemas."""

from __future__ import annotations

import inspect
import types
from dataclasses import dataclass
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints

_SCALAR_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _is_union(origin: Any) -> bool:
    return origin is Union or origin is types.UnionType


def _type_to_json_schema(py_type: Any) -> dict:
    origin = get_origin(py_type)

    if _is_union(origin):
        non_none = [a for a in get_args(py_type) if a is not type(None)]
        if len(non_none) == 1:
            return _type_to_json_schema(non_none[0])
        raise TypeError(f"unsupported Union type: {py_type}")

    if origin is list:
        (inner,) = get_args(py_type)
        return {"type": "array", "items": _type_to_json_schema(inner)}

    if py_type in _SCALAR_MAP:
        return {"type": _SCALAR_MAP[py_type]}

    raise TypeError(f"unsupported parameter type: {py_type!r}")


def _is_optional(py_type: Any) -> bool:
    origin = get_origin(py_type)
    return _is_union(origin) and type(None) in get_args(py_type)


@dataclass
class _ToolEntry:
    name: str
    description: str
    func: Callable[..., Any]
    parameters: dict


class ToolRegistry:
    """Holds a set of tool functions and exposes Ollama-compatible schemas."""

    def __init__(self) -> None:
        self._tools: dict[str, _ToolEntry] = {}

    def register(self, func: Callable[..., Any]) -> Callable[..., Any]:
        if not func.__doc__:
            raise ValueError(f"tool {func.__name__!r} requires a docstring")

        description = func.__doc__.strip().split("\n\n", 1)[0].strip()
        sig = inspect.signature(func)
        hints = get_type_hints(func)

        properties: dict[str, dict] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name == "return":
                continue
            if param_name not in hints:
                raise TypeError(f"parameter {param_name!r} of {func.__name__!r} lacks a type hint")
            py_type = hints[param_name]
            properties[param_name] = _type_to_json_schema(py_type)
            if param.default is inspect.Parameter.empty and not _is_optional(py_type):
                required.append(param_name)

        self._tools[func.__name__] = _ToolEntry(
            name=func.__name__,
            description=description,
            func=func,
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
            },
        )
        return func

    # Alias so it can be used as a decorator: @registry.tool
    tool = register

    def schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": entry.name,
                    "description": entry.description,
                    "parameters": entry.parameters,
                },
            }
            for entry in self._tools.values()
        ]

    def dispatch(self, name: str, args: dict) -> Any:
        if name not in self._tools:
            return {"error": f"no such tool: {name}"}
        try:
            return self._tools[name].func(**args)
        except Exception as exc:
            return {"error": str(exc)}

    def names(self) -> list[str]:
        return list(self._tools.keys())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_registry.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add harness/registry.py tests/test_registry.py
git commit -m "feat(registry): ToolRegistry with type-hint -> JSON-schema introspection"
```

---

## Task 4: Path sandbox helper

**Files:**
- Create: `harness/tools/_sandbox.py`
- Test: `tests/test_sandbox.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_sandbox.py`:

```python
import os
import pytest

from harness.tools._sandbox import resolve_in_workspace, SandboxError


def test_relative_path_resolves(tmp_path):
    resolved = resolve_in_workspace("subdir/file.txt", tmp_path)
    assert resolved == tmp_path / "subdir" / "file.txt"


def test_dot_resolves_to_workspace(tmp_path):
    assert resolve_in_workspace(".", tmp_path) == tmp_path


def test_dot_dot_escape_rejected(tmp_path):
    with pytest.raises(SandboxError, match="escapes workspace"):
        resolve_in_workspace("../etc/passwd", tmp_path)


def test_absolute_outside_rejected(tmp_path):
    with pytest.raises(SandboxError, match="escapes workspace"):
        resolve_in_workspace("/etc/passwd", tmp_path)


def test_absolute_inside_workspace_allowed(tmp_path):
    inside = tmp_path / "ok.txt"
    resolved = resolve_in_workspace(str(inside), tmp_path)
    assert resolved == inside


def test_symlink_escape_rejected(tmp_path):
    # Create a symlink inside workspace pointing outside.
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    link = tmp_path / "link"
    os.symlink(outside, link)
    with pytest.raises(SandboxError, match="escapes workspace"):
        resolve_in_workspace("link/secret.txt", tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sandbox.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the sandbox helper**

Create `harness/tools/_sandbox.py`:

```python
"""Path containment for tools that touch the filesystem.

Single source of truth for workspace-rooted path resolution.
"""

from __future__ import annotations

from pathlib import Path


class SandboxError(Exception):
    """Raised when a path resolves outside the workspace."""


def resolve_in_workspace(path: str, workspace: Path) -> Path:
    """Resolve `path` against `workspace` and ensure it stays inside.

    Catches `..` traversal, absolute paths outside the root, and symlink
    escapes (because `.resolve()` follows symlinks before the relative check).
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sandbox.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add harness/tools/_sandbox.py tests/test_sandbox.py
git commit -m "feat(tools): workspace path sandbox helper"
```

---

## Task 5: Filesystem tools

**Files:**
- Create: `harness/tools/filesystem.py`
- Test: `tests/test_tools_filesystem.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_tools_filesystem.py`:

```python
import pytest

from harness.tools.filesystem import make_filesystem_tools


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.txt").write_text("bravo")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.txt").write_text("charlie")
    return tmp_path


def test_list_directory_root(ws):
    tools = make_filesystem_tools(ws)
    result = tools["list_directory"](".")
    assert sorted(result) == ["a.txt", "b.txt", "sub"]


def test_list_directory_subdir(ws):
    tools = make_filesystem_tools(ws)
    assert tools["list_directory"]("sub") == ["c.txt"]


def test_list_directory_escape_returns_error(ws):
    tools = make_filesystem_tools(ws)
    result = tools["list_directory"]("../")
    assert isinstance(result, dict) and "error" in result


def test_read_file(ws):
    tools = make_filesystem_tools(ws)
    assert tools["read_file"]("a.txt") == "alpha"


def test_read_file_truncates(ws):
    (ws / "big.txt").write_text("x" * 200)
    tools = make_filesystem_tools(ws)
    result = tools["read_file"]("big.txt", max_bytes=50)
    assert len(result) <= 100  # includes truncation marker
    assert "truncated" in result.lower()


def test_read_file_missing(ws):
    tools = make_filesystem_tools(ws)
    result = tools["read_file"]("nope.txt")
    assert isinstance(result, dict) and "error" in result


def test_write_file_creates(ws):
    tools = make_filesystem_tools(ws)
    result = tools["write_file"]("new.txt", "hello")
    assert result["bytes_written"] == 5
    assert (ws / "new.txt").read_text() == "hello"


def test_write_file_creates_parents(ws):
    tools = make_filesystem_tools(ws)
    tools["write_file"]("deep/nested/file.txt", "data")
    assert (ws / "deep" / "nested" / "file.txt").read_text() == "data"


def test_write_file_append(ws):
    tools = make_filesystem_tools(ws)
    tools["write_file"]("a.txt", " extra", append=True)
    assert (ws / "a.txt").read_text() == "alpha extra"


def test_write_file_escape_returns_error(ws):
    tools = make_filesystem_tools(ws)
    result = tools["write_file"]("../oops.txt", "x")
    assert isinstance(result, dict) and "error" in result
    assert not (ws.parent / "oops.txt").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools_filesystem.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement filesystem tools**

Create `harness/tools/filesystem.py`:

```python
"""Filesystem tools — all paths sandboxed to a workspace directory.

Tools are produced via `make_filesystem_tools(workspace)` rather than
imported as bare functions, so the workspace path is closed over.
"""

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tools_filesystem.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add harness/tools/filesystem.py tests/test_tools_filesystem.py
git commit -m "feat(tools): filesystem tools — list_directory, read_file, write_file"
```

---

## Task 6: Shell tool

**Files:**
- Create: `harness/tools/shell.py`
- Test: `tests/test_tools_shell.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_tools_shell.py`:

```python
import pytest

from harness.tools.shell import make_shell_tool


def test_run_shell_basic(tmp_path):
    (tmp_path / "marker").write_text("hi")
    run_shell = make_shell_tool(tmp_path, default_timeout_sec=10)
    result = run_shell("ls")
    assert result["exit_code"] == 0
    assert "marker" in result["stdout"]
    assert result["timed_out"] is False


def test_run_shell_cwd_is_workspace(tmp_path):
    run_shell = make_shell_tool(tmp_path, default_timeout_sec=10)
    result = run_shell("pwd")
    assert result["stdout"].strip() == str(tmp_path.resolve())


def test_run_shell_captures_stderr(tmp_path):
    run_shell = make_shell_tool(tmp_path, default_timeout_sec=10)
    result = run_shell("echo oops >&2; exit 3")
    assert result["exit_code"] == 3
    assert "oops" in result["stderr"]


def test_run_shell_timeout(tmp_path):
    run_shell = make_shell_tool(tmp_path, default_timeout_sec=10)
    result = run_shell("sleep 5", timeout_sec=1)
    assert result["timed_out"] is True
    assert result["exit_code"] == -1


def test_run_shell_env_is_stripped(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET", "must-not-leak")
    run_shell = make_shell_tool(tmp_path, default_timeout_sec=10)
    result = run_shell("echo ${SECRET:-empty}")
    assert "must-not-leak" not in result["stdout"]
    assert "empty" in result["stdout"]


def test_run_shell_home_is_workspace(tmp_path):
    run_shell = make_shell_tool(tmp_path, default_timeout_sec=10)
    result = run_shell("echo $HOME")
    assert result["stdout"].strip() == str(tmp_path.resolve())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools_shell.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement shell tool**

Create `harness/tools/shell.py`:

```python
"""Sandboxed shell execution tool.

The command runs with cwd=workspace, a stripped environment (only PATH, HOME,
USER), and a hard timeout. There is no shell history, no inherited env,
no parent stdin.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable


def make_shell_tool(workspace: Path, default_timeout_sec: int = 30) -> Callable:
    """Return the run_shell tool bound to the given workspace."""
    workspace = workspace.resolve()

    def run_shell(command: str, timeout_sec: int = default_timeout_sec) -> dict:
        """Run a bash command inside the sandboxed workspace.

        The command runs with cwd set to the workspace, a stripped environment,
        and a configurable timeout. Returns exit_code, stdout, stderr, and a
        timed_out flag.
        """
        env = {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HOME": str(workspace),
            "USER": os.environ.get("USER", "harness"),
            "LANG": "C.UTF-8",
        }
        try:
            proc = subprocess.run(
                ["bash", "-c", command],
                cwd=str(workspace),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "exit_code": -1,
                "stdout": (exc.stdout or b"").decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                "stderr": (exc.stderr or b"").decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
                "timed_out": True,
            }
        return {
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timed_out": False,
        }

    return run_shell
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tools_shell.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add harness/tools/shell.py tests/test_tools_shell.py
git commit -m "feat(tools): sandboxed run_shell with timeout and stripped env"
```

---

## Task 7: Web search tool + registry wiring

**Files:**
- Create: `harness/tools/web.py`
- Modify: `harness/tools/__init__.py`

No unit tests for `search_web` — it hits the live DuckDuckGo service and would be flaky in CI. The agent-level test in Task 9 will exercise it indirectly via mock.

- [ ] **Step 1: Implement web.py**

Create `harness/tools/web.py`:

```python
"""Web search via DuckDuckGo (no API key required)."""

from __future__ import annotations

from typing import Callable


def make_web_tool() -> Callable:
    def search_web(query: str, max_results: int = 5) -> list[dict] | dict:
        """Search the web via DuckDuckGo and return top results.

        Each result is a dict with title, url, and snippet keys. Use this when
        you need information beyond what is in the workspace.
        """
        try:
            from ddgs import DDGS  # imported lazily so unit tests don't need network
        except ImportError as exc:
            return {"error": f"ddgs not installed: {exc}"}

        try:
            with DDGS() as engine:
                hits = list(engine.text(query, max_results=max_results))
        except Exception as exc:
            return {"error": f"search failed: {exc}"}

        return [
            {
                "title": h.get("title", ""),
                "url": h.get("href", h.get("url", "")),
                "snippet": h.get("body", ""),
            }
            for h in hits
        ]

    return search_web
```

- [ ] **Step 2: Wire build_registry()**

Replace `harness/tools/__init__.py`:

```python
"""Tool registry wiring.

`build_registry(workspace)` constructs a ToolRegistry populated with the
filesystem, shell, and web tools, each closed over the given workspace.
"""

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
```

- [ ] **Step 3: Smoke-test the wiring**

Run: `uv run python -c "
from pathlib import Path
from harness.tools import build_registry
r = build_registry(Path('./workspace'))
print(sorted(r.names()))
"`
Expected output: `['list_directory', 'read_file', 'run_shell', 'search_web', 'write_file']`

- [ ] **Step 4: Commit**

```bash
git add harness/tools/web.py harness/tools/__init__.py
git commit -m "feat(tools): web search + build_registry wiring"
```

---

## Task 8: Ollama client wrapper

**Files:**
- Create: `harness/ollama_client.py`
- Test: `tests/test_ollama_client.py`

The wrapper is intentionally thin — it exists to make mocking easy in agent tests and to set Ollama options (temperature, num_ctx) per-request.

- [ ] **Step 1: Write failing tests**

Create `tests/test_ollama_client.py`:

```python
from unittest.mock import MagicMock

from harness.ollama_client import OllamaClient


def test_chat_passes_messages_and_tools():
    underlying = MagicMock()
    underlying.chat.return_value = iter([
        {"message": {"role": "assistant", "content": "hi", "tool_calls": None}, "done": True},
    ])
    client = OllamaClient(
        host="http://x:11434",
        model="testmodel",
        temperature=0.5,
        num_ctx=2048,
        _client=underlying,
    )

    chunks = list(client.chat(messages=[{"role": "user", "content": "hello"}], tools=[]))

    assert len(chunks) == 1
    underlying.chat.assert_called_once()
    kwargs = underlying.chat.call_args.kwargs
    assert kwargs["model"] == "testmodel"
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert kwargs["tools"] == []
    assert kwargs["stream"] is True
    assert kwargs["options"]["temperature"] == 0.5
    assert kwargs["options"]["num_ctx"] == 2048


def test_chat_yields_streamed_chunks():
    underlying = MagicMock()
    underlying.chat.return_value = iter([
        {"message": {"role": "assistant", "content": "Hel"}, "done": False},
        {"message": {"role": "assistant", "content": "lo"}, "done": False},
        {"message": {"role": "assistant", "content": "", "tool_calls": None}, "done": True},
    ])
    client = OllamaClient(host="x", model="m", temperature=0.7, num_ctx=1024, _client=underlying)

    chunks = list(client.chat(messages=[{"role": "user", "content": "hi"}], tools=[]))
    assert len(chunks) == 3
    assert chunks[0]["message"]["content"] == "Hel"
    assert chunks[2]["done"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ollama_client.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement OllamaClient**

Create `harness/ollama_client.py`:

```python
"""Thin wrapper around the Ollama Python SDK.

Provides a streaming chat() that yields raw response chunks. The wrapper
exists to inject per-request options (temperature, num_ctx) consistently
and to make the underlying client injectable for tests.
"""

from __future__ import annotations

from typing import Any, Iterator


class OllamaClient:
    def __init__(
        self,
        *,
        host: str,
        model: str,
        temperature: float,
        num_ctx: int,
        _client: Any | None = None,
    ) -> None:
        self.host = host
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        if _client is None:
            import ollama
            self._client = ollama.Client(host=host)
        else:
            self._client = _client

    def chat(self, *, messages: list[dict], tools: list[dict]) -> Iterator[dict]:
        """Stream a chat completion. Yields raw response chunks from the SDK."""
        yield from self._client.chat(
            model=self.model,
            messages=messages,
            tools=tools or None,
            stream=True,
            options={
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ollama_client.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add harness/ollama_client.py tests/test_ollama_client.py
git commit -m "feat(client): OllamaClient streaming wrapper with injectable backend"
```

---

## Task 9: Agent + event types

**Files:**
- Create: `harness/agent.py`
- Test: `tests/test_agent_events.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_agent_events.py`:

```python
from unittest.mock import MagicMock

from harness.agent import Agent, TextChunk, ToolCallStart, ToolCallResult, TurnComplete
from harness.registry import ToolRegistry


def _make_client(stream_responses):
    """stream_responses: list-of-lists of chunk dicts (one inner list per chat() call)."""
    client = MagicMock()
    call_iter = iter(stream_responses)
    client.chat.side_effect = lambda **_: iter(next(call_iter))
    return client


def test_plain_text_reply_emits_text_chunks_then_complete():
    client = _make_client([[
        {"message": {"role": "assistant", "content": "Hi "}, "done": False},
        {"message": {"role": "assistant", "content": "there", "tool_calls": None}, "done": True},
    ]])
    agent = Agent(client=client, registry=ToolRegistry(), system_prompt="sys", max_iterations=10)

    events = list(agent.run_turn("hello"))
    text_chunks = [e for e in events if isinstance(e, TextChunk)]
    assert "".join(e.text for e in text_chunks) == "Hi there"
    assert isinstance(events[-1], TurnComplete)


def test_one_tool_call_then_reply():
    client = _make_client([
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "echo", "arguments": {"text": "hi"}}}
                    ],
                },
                "done": True,
            }
        ],
        [
            {"message": {"role": "assistant", "content": "done", "tool_calls": None}, "done": True},
        ],
    ])
    registry = ToolRegistry()

    @registry.tool
    def echo(text: str) -> str:
        """Echo."""
        return text.upper()

    agent = Agent(client=client, registry=registry, system_prompt="sys", max_iterations=10)
    events = list(agent.run_turn("say hi"))

    starts = [e for e in events if isinstance(e, ToolCallStart)]
    results = [e for e in events if isinstance(e, ToolCallResult)]
    assert len(starts) == 1 and starts[0].name == "echo"
    assert len(results) == 1 and results[0].result == "HI"
    assert isinstance(events[-1], TurnComplete)


def test_two_sequential_tool_calls_then_reply():
    client = _make_client([
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "one", "arguments": {}}},
                        {"function": {"name": "two", "arguments": {}}},
                    ],
                },
                "done": True,
            }
        ],
        [{"message": {"role": "assistant", "content": "ok", "tool_calls": None}, "done": True}],
    ])
    registry = ToolRegistry()

    @registry.tool
    def one() -> str:
        """One."""
        return "1"

    @registry.tool
    def two() -> str:
        """Two."""
        return "2"

    agent = Agent(client=client, registry=registry, system_prompt="sys", max_iterations=10)
    events = list(agent.run_turn("go"))
    starts = [e for e in events if isinstance(e, ToolCallStart)]
    assert [s.name for s in starts] == ["one", "two"]


def test_max_iterations_terminates():
    # Model keeps requesting a tool forever; agent must stop and produce TurnComplete.
    def looping_response():
        return [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "loop", "arguments": {}}}],
            },
            "done": True,
        }]

    # Final iteration after cap is hit returns a plain text reply.
    final = [{"message": {"role": "assistant", "content": "stopped", "tool_calls": None}, "done": True}]

    client = _make_client([looping_response() for _ in range(3)] + [final])
    registry = ToolRegistry()

    @registry.tool
    def loop() -> str:
        """Loop."""
        return "again"

    agent = Agent(client=client, registry=registry, system_prompt="sys", max_iterations=3)
    events = list(agent.run_turn("go"))
    assert isinstance(events[-1], TurnComplete)
    # Should have stopped after 3 tool iterations + 1 final.
    assert client.chat.call_count == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_agent_events.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement Agent and events**

Create `harness/agent.py`:

```python
"""Agent: the tool-calling loop. UI-agnostic.

Yields a typed event stream that any front-end can consume.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator

from harness.ollama_client import OllamaClient
from harness.registry import ToolRegistry


# --- Event types ---------------------------------------------------------


@dataclass
class TextChunk:
    text: str


@dataclass
class ToolCallStart:
    name: str
    args: dict
    call_id: str


@dataclass
class ToolCallResult:
    call_id: str
    result: Any
    error: str | None = None


@dataclass
class TurnComplete:
    pass


@dataclass
class AgentError:
    message: str


Event = TextChunk | ToolCallStart | ToolCallResult | TurnComplete | AgentError


DEFAULT_SYSTEM_PROMPT = """You are a coding assistant with access to filesystem, shell, and web search tools.

The filesystem and shell tools are sandboxed to a workspace directory — paths
are resolved relative to it. Prefer using tools to inspect state rather than
asking the user clarifying questions about it. Be concise."""


# --- Agent ---------------------------------------------------------------


@dataclass
class Agent:
    client: OllamaClient
    registry: ToolRegistry
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_iterations: int = 10
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.history:
            self.history.append({"role": "system", "content": self.system_prompt})

    def reset(self) -> None:
        self.history = [{"role": "system", "content": self.system_prompt}]

    def run_turn(self, user_input: str) -> Iterator[Event]:
        self.history.append({"role": "user", "content": user_input})

        for iteration in range(self.max_iterations):
            assistant_text, tool_calls = yield from self._stream_one_response()

            self.history.append({
                "role": "assistant",
                "content": assistant_text,
                "tool_calls": tool_calls if tool_calls else None,
            })

            if not tool_calls:
                yield TurnComplete()
                return

            for call in tool_calls:
                name = call["function"]["name"]
                args = call["function"].get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                call_id = call.get("id") or str(uuid.uuid4())

                yield ToolCallStart(name=name, args=args, call_id=call_id)
                raw = self.registry.dispatch(name, args)
                error = raw["error"] if isinstance(raw, dict) and "error" in raw else None
                yield ToolCallResult(call_id=call_id, result=raw, error=error)

                self.history.append({
                    "role": "tool",
                    "name": name,
                    "tool_call_id": call_id,
                    "content": json.dumps(raw, default=str),
                })

        # Max iterations reached — ask for a summary and emit one final response.
        self.history.append({
            "role": "system",
            "content": "Max tool iterations reached. Summarize what you found without calling more tools.",
        })
        yield from self._stream_one_response_as_final()
        yield TurnComplete()

    def _stream_one_response(self) -> Iterator[Event]:
        """Stream one /api/chat response. Yields TextChunk events.

        Returns (assistant_text, tool_calls_list) via `return` (generator return value).
        """
        text_parts: list[str] = []
        tool_calls: list[dict] = []

        for chunk in self.client.chat(messages=self.history, tools=self.registry.schemas()):
            msg = chunk.get("message", {})
            piece = msg.get("content") or ""
            if piece:
                text_parts.append(piece)
                yield TextChunk(text=piece)
            calls = msg.get("tool_calls")
            if calls:
                tool_calls.extend(calls)

        return "".join(text_parts), tool_calls

    def _stream_one_response_as_final(self) -> Iterator[Event]:
        """Same as above but discards any tool_calls — used after max iterations."""
        text_parts: list[str] = []
        for chunk in self.client.chat(messages=self.history, tools=[]):
            piece = chunk.get("message", {}).get("content") or ""
            if piece:
                text_parts.append(piece)
                yield TextChunk(text=piece)
        self.history.append({"role": "assistant", "content": "".join(text_parts)})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_agent_events.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add harness/agent.py tests/test_agent_events.py
git commit -m "feat(agent): event-driven tool-calling loop with mocked-client tests"
```

---

## Task 10: REPL front-end

**Files:**
- Create: `harness/ui_repl.py`

No automated tests — interactive REPL is verified by manual smoke check at the end.

- [ ] **Step 1: Implement the REPL**

Create `harness/ui_repl.py`:

```python
"""Terminal front-end: prompt loop that consumes Agent events and renders via rich."""

from __future__ import annotations

import json
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from harness.agent import (
    Agent,
    AgentError,
    TextChunk,
    ToolCallResult,
    ToolCallStart,
    TurnComplete,
)
from harness.config import Config
from harness.ollama_client import OllamaClient
from harness.tools import build_registry


_HELP = """\
Built-in commands:
  /help      Show this help and current config
  /reset     Clear the conversation history
  /history   Print the message history
  /tokens    Approximate token count (chars / 4)
  /quit      Exit (Ctrl-D also works)
"""


def _approx_tokens(history: list[dict]) -> int:
    return sum(len(str(m.get("content", ""))) for m in history) // 4


def run(config: Config) -> None:
    console = Console()
    config.workspace.mkdir(parents=True, exist_ok=True)

    client = OllamaClient(
        host=config.ollama_host,
        model=config.model,
        temperature=config.temperature,
        num_ctx=config.num_ctx,
    )
    registry = build_registry(config.workspace, shell_timeout_sec=config.shell_timeout_sec)
    agent = Agent(client=client, registry=registry, max_iterations=config.max_tool_iterations)

    console.print(Panel.fit(
        f"[bold]ollama-experiments[/bold]\n"
        f"model: [cyan]{config.model}[/cyan]   workspace: [cyan]{config.workspace}[/cyan]\n"
        f"tools: [cyan]{', '.join(registry.names())}[/cyan]\n"
        f"Type [bold]/help[/bold] for commands, Ctrl-D to quit.",
        border_style="blue",
    ))

    while True:
        try:
            user_input = console.input("[bold green]you ›[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return

        if not user_input:
            continue
        if user_input == "/quit":
            return
        if user_input == "/help":
            console.print(Markdown(_HELP))
            continue
        if user_input == "/reset":
            agent.reset()
            console.print("[dim]history cleared[/dim]")
            continue
        if user_input == "/history":
            console.print_json(json.dumps(agent.history, default=str))
            continue
        if user_input == "/tokens":
            console.print(f"[dim]~{_approx_tokens(agent.history)} tokens[/dim]")
            continue

        console.print("[bold magenta]assistant ›[/bold magenta] ", end="")
        try:
            for event in agent.run_turn(user_input):
                _render_event(console, event)
        except KeyboardInterrupt:
            console.print("\n[dim]turn cancelled[/dim]")


def _render_event(console: Console, event) -> None:
    if isinstance(event, TextChunk):
        sys.stdout.write(event.text)
        sys.stdout.flush()
    elif isinstance(event, ToolCallStart):
        args_preview = json.dumps(event.args)[:120]
        console.print(f"\n[dim cyan]▶ {event.name}({args_preview})[/dim cyan]")
    elif isinstance(event, ToolCallResult):
        preview = json.dumps(event.result, default=str)[:500]
        style = "dim red" if event.error else "dim cyan"
        console.print(f"[{style}]◀ {preview}[/{style}]")
    elif isinstance(event, TurnComplete):
        console.print()  # final newline
    elif isinstance(event, AgentError):
        console.print(f"[red]error: {event.message}[/red]")
```

- [ ] **Step 2: Smoke-check import**

Run: `uv run python -c "from harness.ui_repl import run; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add harness/ui_repl.py
git commit -m "feat(ui): terminal REPL front-end rendering Agent events"
```

---

## Task 11: Streamlit front-end

**Files:**
- Create: `harness/ui_streamlit.py`

- [ ] **Step 1: Implement the Streamlit page**

Create `harness/ui_streamlit.py`:

```python
"""Streamlit front-end: same Agent, rendered as a browser chat.

Run via `streamlit run harness/ui_streamlit.py` or `python -m harness --ui streamlit`.
"""

from __future__ import annotations

import json

import streamlit as st

from harness.agent import (
    Agent,
    AgentError,
    TextChunk,
    ToolCallResult,
    ToolCallStart,
    TurnComplete,
)
from harness.config import Config
from harness.ollama_client import OllamaClient
from harness.tools import build_registry


st.set_page_config(page_title="ollama-experiments", page_icon="🦙", layout="wide")


def _build_agent(config: Config) -> Agent:
    config.workspace.mkdir(parents=True, exist_ok=True)
    client = OllamaClient(
        host=config.ollama_host,
        model=config.model,
        temperature=config.temperature,
        num_ctx=config.num_ctx,
    )
    registry = build_registry(config.workspace, shell_timeout_sec=config.shell_timeout_sec)
    return Agent(client=client, registry=registry, max_iterations=config.max_tool_iterations)


def _init_state() -> None:
    if "config" not in st.session_state:
        st.session_state.config = Config.from_env()
    if "agent" not in st.session_state:
        st.session_state.agent = _build_agent(st.session_state.config)
    if "visible_messages" not in st.session_state:
        st.session_state.visible_messages = []  # list[{"role", "content", "tool_calls"}]


def _render_history() -> None:
    for msg in st.session_state.visible_messages:
        with st.chat_message(msg["role"]):
            if msg.get("content"):
                st.markdown(msg["content"])
            for tc in msg.get("tool_calls", []):
                with st.expander(f"🔧 {tc['name']}", expanded=False):
                    st.json(tc["args"])
                    st.markdown("**Result:**")
                    st.json(tc["result"])


def _run_turn(user_input: str) -> None:
    agent: Agent = st.session_state.agent
    st.session_state.visible_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        text_placeholder = st.empty()
        buffer = ""
        tool_call_records: dict[str, dict] = {}
        ordered_call_ids: list[str] = []
        expanders: dict[str, "st.delta_generator.DeltaGenerator"] = {}

        try:
            for event in agent.run_turn(user_input):
                if isinstance(event, TextChunk):
                    buffer += event.text
                    text_placeholder.markdown(buffer)
                elif isinstance(event, ToolCallStart):
                    ordered_call_ids.append(event.call_id)
                    record = {"name": event.name, "args": event.args, "result": None}
                    tool_call_records[event.call_id] = record
                    exp = st.expander(f"🔧 {event.name}", expanded=False)
                    expanders[event.call_id] = exp
                    with exp:
                        st.json(event.args)
                        st.markdown("**Result:** _pending..._")
                elif isinstance(event, ToolCallResult):
                    tool_call_records[event.call_id]["result"] = event.result
                    exp = expanders[event.call_id]
                    with exp:
                        st.markdown("**Result:**")
                        if event.error:
                            st.error(event.error)
                        else:
                            st.json(event.result)
                elif isinstance(event, TurnComplete):
                    pass
                elif isinstance(event, AgentError):
                    st.error(event.message)
        except Exception as exc:
            st.error(f"agent error: {exc}")
            return

    st.session_state.visible_messages.append({
        "role": "assistant",
        "content": buffer,
        "tool_calls": [tool_call_records[cid] for cid in ordered_call_ids],
    })


def main() -> None:
    _init_state()
    config: Config = st.session_state.config

    with st.sidebar:
        st.title("🦙 Harness")
        st.caption("Local LLM playground")
        st.markdown(f"**Model:** `{config.model}`")
        st.markdown(f"**Host:** `{config.ollama_host}`")
        st.markdown(f"**Workspace:** `{config.workspace}`")
        st.markdown(f"**Temperature:** `{config.temperature}`")
        st.markdown(f"**num_ctx:** `{config.num_ctx}`")
        st.markdown(f"**Max iterations:** `{config.max_tool_iterations}`")
        st.divider()
        if st.button("Reset conversation", use_container_width=True):
            st.session_state.agent.reset()
            st.session_state.visible_messages = []
            st.rerun()

    st.title("Chat")
    _render_history()

    user_input = st.chat_input("Ask the model anything (it can list files, run shell, search web)")
    if user_input:
        _run_turn(user_input)


main()
```

- [ ] **Step 2: Smoke-check it imports without error**

Run: `uv run python -c "import importlib; importlib.import_module('harness.ui_streamlit'); print('ok')"`
Expected: `ok` (Streamlit will print a warning about running outside the streamlit runtime — that's fine, it still imports.)

Note: if you see `streamlit.runtime.scriptrunner.script_runner.NoSessionContext`, ignore it — the page only does work inside `main()` and we didn't call it.

- [ ] **Step 3: Commit**

```bash
git add harness/ui_streamlit.py
git commit -m "feat(ui): Streamlit chat front-end with live streaming and tool expanders"
```

---

## Task 12: Entry point and dispatcher

**Files:**
- Create: `harness/__main__.py`

- [ ] **Step 1: Implement the entry point**

Create `harness/__main__.py`:

```python
"""`python -m harness` entry point.

Default: starts the terminal REPL.
`--ui streamlit`: re-execs into `streamlit run harness/ui_streamlit.py`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from harness.config import Config


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="harness", description="Local LLM tool-calling harness")
    parser.add_argument("--ui", choices=["repl", "streamlit"], default="repl")
    parser.add_argument("--model", default=None, help="Ollama model name")
    parser.add_argument("--temp", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--ctx", type=int, default=None, help="num_ctx")
    parser.add_argument("--workspace", type=Path, default=None, help="Sandbox root for filesystem and shell tools")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if args.ui == "streamlit":
        # Streamlit needs to own the process. Re-exec into `streamlit run`.
        # Propagate flags via env so the page picks them up through Config.from_env().
        env = os.environ.copy()
        if args.model:
            env["OLLAMA_MODEL"] = args.model
        if args.temp is not None:
            env["TEMPERATURE"] = str(args.temp)
        if args.ctx is not None:
            env["NUM_CTX"] = str(args.ctx)
        if args.workspace is not None:
            env["WORKSPACE_ROOT"] = str(args.workspace)

        page = Path(__file__).parent / "ui_streamlit.py"
        os.execvpe("streamlit", ["streamlit", "run", str(page)], env)

    # REPL path
    config = Config.from_env(
        model=args.model,
        temperature=args.temp,
        num_ctx=args.ctx,
        workspace=args.workspace,
    )

    # Late import so streamlit reexec path doesn't pull in rich unnecessarily.
    from harness.ui_repl import run
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-test arg parsing**

Run: `uv run python -m harness --help`
Expected: prints the argparse help text with `--ui`, `--model`, `--temp`, `--ctx`, `--workspace`.

- [ ] **Step 3: Smoke-test REPL startup (do not send a message)**

Run: `echo "/quit" | uv run python -m harness 2>&1 | head -20`
Expected: prints the welcome panel showing model name, workspace path, and tool list. May print a connection error if Ollama isn't running — that's acceptable for this smoke test. If it prints the panel, the wiring is correct.

- [ ] **Step 4: Commit**

```bash
git add harness/__main__.py
git commit -m "feat: entry point dispatching between REPL and Streamlit"
```

---

## Task 13: End-to-end test (optional, gated) and README polish

**Files:**
- Create: `tests/test_e2e.py`
- Modify: `README.md`

The e2e test only runs when `OLLAMA_E2E=1` and Ollama is serving. CI never runs it; it's there for manual confidence checks.

- [ ] **Step 1: Write the gated e2e test**

Create `tests/test_e2e.py`:

```python
"""End-to-end test against a live Ollama server.

Gated behind OLLAMA_E2E=1 because it requires:
  - `ollama serve` running locally
  - the qwen3-coder-32k:latest model available
"""

import os
from pathlib import Path

import pytest

from harness.agent import Agent, TextChunk, ToolCallResult, ToolCallStart, TurnComplete
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
```

- [ ] **Step 2: Run all unit tests to confirm everything still passes**

Run: `uv run pytest -v`
Expected: all tests pass except `test_e2e.py` which is skipped.

- [ ] **Step 3: Replace README with the polished version**

Overwrite `README.md`:

````markdown
# ollama-experiments

A small Python harness for experimenting with local LLMs via [Ollama](https://ollama.com), using native tool-calling. Ships with filesystem, shell, and web-search tools sandboxed to a workspace directory, and two interchangeable front-ends: a terminal REPL and a Streamlit chat page.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

You'll also need [Ollama](https://ollama.com) installed and a tool-calling-capable model pulled. The default is `qwen3-coder-32k:latest`, built from this Modelfile:

```
FROM qwen3-coder:30b
PARAMETER num_ctx 32768
PARAMETER temperature 0.7
```

Build it with:

```bash
ollama create qwen3-coder-32k -f Modelfile
```

Start the Ollama server in a separate terminal:

```bash
ollama serve
```

## Run

REPL (default):

```bash
uv run python -m harness
```

Streamlit UI:

```bash
uv run python -m harness --ui streamlit
```

Flags: `--model`, `--temp`, `--ctx`, `--workspace`. Env vars: `OLLAMA_HOST`, `OLLAMA_MODEL`, `TEMPERATURE`, `NUM_CTX`, `WORKSPACE_ROOT`, `MAX_TOOL_ITERATIONS`, `SHELL_TIMEOUT_SEC`.

## Tools

| Tool | What it does |
|---|---|
| `list_directory(path)` | List names under a workspace-relative path |
| `read_file(path, max_bytes)` | Read UTF-8 contents, truncated above max_bytes |
| `write_file(path, content, append)` | Write or append to a workspace file |
| `run_shell(command, timeout_sec)` | bash with `cwd=workspace`, stripped env, 30s default timeout |
| `search_web(query, max_results)` | DuckDuckGo, no API key |

All filesystem and shell tools refuse paths that escape `./workspace/`.

## Layout

```
harness/
  agent.py           # Tool-calling loop, yields typed events
  ollama_client.py   # Thin streaming wrapper around ollama-python
  registry.py        # @tool decorator + JSON-schema introspection
  config.py          # Env + CLI configuration
  ui_repl.py         # Terminal front-end (rich)
  ui_streamlit.py    # Browser front-end (streamlit)
  __main__.py        # Dispatcher
  tools/
    filesystem.py    # list_directory, read_file, write_file
    shell.py         # run_shell
    web.py           # search_web
    _sandbox.py      # Path containment helper
```

## Tests

```bash
uv run pytest                # unit tests
OLLAMA_E2E=1 uv run pytest   # also run live integration test
```

## Design doc

`docs/superpowers/specs/2026-05-20-ollama-harness-design.md`
````

- [ ] **Step 4: Final all-tests run**

Run: `uv run pytest -v`
Expected: all non-e2e tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e.py README.md
git commit -m "docs: gated e2e test and polished README"
```

- [ ] **Step 6: Manual smoke check**

After the commit, do one manual check that the whole thing actually talks to a live model:

```bash
ollama serve  # in another terminal, if not already running
uv run python -m harness
# Then type: "What files are in the workspace?"
# Expect: model calls list_directory, replies with the (empty) workspace contents.
```

Then try Streamlit:

```bash
uv run python -m harness --ui streamlit
# Browser opens. Type the same prompt. Expect tool-call expander + streamed reply.
```

---

## Done

At this point you should have:
- All unit tests passing
- REPL working end-to-end against a live model
- Streamlit UI working end-to-end against a live model
- 13 atomic commits, each green
