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
