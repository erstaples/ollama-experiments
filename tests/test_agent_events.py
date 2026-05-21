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
    def looping_response():
        return [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "loop", "arguments": {}}}],
            },
            "done": True,
        }]

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
    assert client.chat.call_count == 4
