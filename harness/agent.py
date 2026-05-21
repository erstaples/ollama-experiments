from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator

from harness.ollama_client import OllamaClient
from harness.registry import ToolRegistry


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

        for _iteration in range(self.max_iterations):
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
        text_parts: list[str] = []
        for chunk in self.client.chat(messages=self.history, tools=[]):
            piece = chunk.get("message", {}).get("content") or ""
            if piece:
                text_parts.append(piece)
                yield TextChunk(text=piece)
        self.history.append({"role": "assistant", "content": "".join(text_parts)})
