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
    assert kwargs["tools"] is None  # empty list becomes None via `tools or None`
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
