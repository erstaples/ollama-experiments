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
        console.print()
    elif isinstance(event, AgentError):
        console.print(f"[red]error: {event.message}[/red]")
