from __future__ import annotations

from typing import Callable


def make_web_tool() -> Callable:
    def search_web(query: str, max_results: int = 5) -> list[dict] | dict:
        """Search the web via DuckDuckGo and return top results.

        Each result is a dict with title, url, and snippet keys. Use this when
        you need information beyond what is in the workspace.
        """
        try:
            from ddgs import DDGS
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
