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
