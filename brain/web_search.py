"""Live web search via DuckDuckGo — no API key required."""

from __future__ import annotations

import os
import time
from typing import Any

import requests

_TIMEOUT = int(os.environ.get("AUREON_SEARCH_TIMEOUT", "8"))
_MAX_RESULTS = int(os.environ.get("AUREON_SEARCH_MAX_RESULTS", "5"))
_RATE_LIMIT_SECONDS = 2.0
_last_search: float = 0.0


def web_search_enabled() -> bool:
    return os.environ.get("AUREON_WEB_SEARCH_ENABLED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def search(query: str, *, max_results: int = _MAX_RESULTS) -> list[dict[str, Any]]:
    """Search DuckDuckGo and return structured results."""
    global _last_search

    if not web_search_enabled():
        return [{"error": "web search disabled", "source": "duckduckgo"}]

    elapsed = time.time() - _last_search
    if elapsed < _RATE_LIMIT_SECONDS:
        time.sleep(_RATE_LIMIT_SECONDS - elapsed)
    _last_search = time.time()

    try:
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_redirect": "1",
                "no_html": "1",
                "skip_disambig": "1",
            },
            timeout=_TIMEOUT,
            headers={"User-Agent": "SOLIA-Aureon/1.0 (sovereign intelligence)"},
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        return [{"error": str(exc), "source": "duckduckgo"}]

    results: list[dict[str, Any]] = []

    abstract = str(data.get("Abstract", "")).strip()
    if abstract:
        results.append({
            "type": "abstract",
            "text": abstract,
            "source": data.get("AbstractSource", ""),
            "url": data.get("AbstractURL", ""),
        })

    for topic in data.get("RelatedTopics", [])[:max_results]:
        if isinstance(topic, dict) and topic.get("Text"):
            results.append({
                "type": "related",
                "text": str(topic.get("Text", "")).strip(),
                "url": topic.get("FirstURL", ""),
                "source": "duckduckgo",
            })

    answer = str(data.get("Answer", "")).strip()
    if answer:
        results.append({
            "type": "instant_answer",
            "text": answer,
            "source": data.get("AnswerType", "duckduckgo"),
        })

    return results[:max_results]


def format_for_context(results: list[dict[str, Any]]) -> str:
    """Convert search results into a context string for the predict brain."""
    if not results:
        return ""
    parts: list[str] = []
    for item in results:
        if item.get("error"):
            continue
        text = str(item.get("text", "")).strip()
        source = str(item.get("source", "web"))
        if text:
            parts.append(f"source {source}: {text[:300]}")
    if not parts:
        return ""
    return "web search results: " + " | ".join(parts)
