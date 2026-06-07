"""Live web search via DuckDuckGo — no API key required."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

_TIMEOUT = int(os.environ.get("AUREON_SEARCH_TIMEOUT", "8"))
_MAX_RESULTS = int(os.environ.get("AUREON_SEARCH_MAX_RESULTS", "5"))
_RATE_LIMIT_SECONDS = 2.0
_last_search: float = 0.0

_LIVE_NEWS_SIGNALS = (
    "news",
    "today",
    "happened",
    "latest",
    "current",
    "right now",
    "this week",
    "going on",
    "breaking",
)
_TECH_SIGNALS = (
    "tech",
    "technology",
    "silicon",
    "startup",
    "software",
    "chip",
    "ai",
    "nvidia",
    "apple",
    "google",
)

TRUSTED_SOURCE_WHITELIST: dict[str, tuple[str, ...]] = {
    "physics": ("arxiv.org", "pubmed.ncbi.nlm.nih.gov", "scholar.google.com"),
    "biology": ("arxiv.org", "pubmed.ncbi.nlm.nih.gov", "scholar.google.com"),
    "mathematics": ("arxiv.org", "pubmed.ncbi.nlm.nih.gov", "scholar.google.com"),
    "science_and_natural_philosophy": (
        "arxiv.org",
        "pubmed.ncbi.nlm.nih.gov",
        "scholar.google.com",
    ),
    "history": ("britannica.com", "history.com", "jstor.org"),
    "geopolitics": ("britannica.com", "history.com", "jstor.org"),
    "governance_and_political_systems": ("britannica.com", "history.com", "jstor.org"),
    "technology_and_engineering": ("arxiv.org", "paperswithcode.com", "cs.stanford.edu"),
    "computer_science": ("arxiv.org", "paperswithcode.com", "cs.stanford.edu"),
    "psychology": ("apa.org", "pubmed.ncbi.nlm.nih.gov"),
    "social_sciences": ("apa.org", "pubmed.ncbi.nlm.nih.gov"),
    "economics": ("imf.org", "worldbank.org", "federalreserve.gov"),
    "spirituality": ("sacred-texts.com", "wisdomlib.org"),
    "religion_and_spirituality": ("sacred-texts.com", "wisdomlib.org"),
    "metaphysics_mysticism_and_occult_sciences": ("sacred-texts.com", "wisdomlib.org"),
    "general": ("wikipedia.org",),
}

_HOST_ALIASES: dict[str, str] = {
    "www.ncbi.nlm.nih.gov": "pubmed.ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov": "pubmed.ncbi.nlm.nih.gov",
    "en.wikipedia.org": "wikipedia.org",
    "www.wikipedia.org": "wikipedia.org",
    "www.britannica.com": "britannica.com",
    "www.history.com": "history.com",
    "www.jstor.org": "jstor.org",
    "www.imf.org": "imf.org",
    "www.worldbank.org": "worldbank.org",
    "www.federalreserve.gov": "federalreserve.gov",
    "www.apa.org": "apa.org",
    "www.sacred-texts.com": "sacred-texts.com",
    "www.wisdomlib.org": "wisdomlib.org",
}


def is_live_news_query(query: str) -> bool:
    """True when the user wants current events, not evergreen reference pages."""
    from brain.response_quality import is_specific_topic_inquiry

    if is_specific_topic_inquiry(query):
        return False
    q = query.strip().lower()
    return any(s in q for s in _LIVE_NEWS_SIGNALS)


def rewrite_topic_inquiry_query(question: str) -> str:
    """Focus search on named projects/programs instead of broad domain words like 'history'."""
    import re

    q = question.strip()
    acronyms = re.findall(r"\b[A-Z][A-Z0-9]+\b", q)
    if acronyms:
        core = " ".join(dict.fromkeys(acronyms))
        return f"{core} program history"
    proper = [w for w in q.split() if w[0].isupper() and len(w) > 2 and w.isalpha()]
    if proper:
        return f"{' '.join(proper[:5])} history"
    cleaned = re.sub(r"(?i)\b(can you )?tell me about (the )?history about (the )?", "", q)
    return cleaned.strip() or question.strip()


def rewrite_live_news_query(question: str) -> str:
    """Turn vague chat questions into news-search queries that return headlines."""
    q = question.strip().lower()
    if any(t in q for t in _TECH_SIGNALS) and any(n in q for n in _LIVE_NEWS_SIGNALS):
        return "technology news today AI startups silicon valley"
    if "stock" in q or "market" in q:
        return "stock market technology news today"
    if any(n in q for n in _LIVE_NEWS_SIGNALS):
        return "latest breaking news headlines today"
    return question.strip()


def web_search_enabled() -> bool:
    return os.environ.get("AUREON_WEB_SEARCH_ENABLED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def trusted_live_retrieval_enabled() -> bool:
    raw = os.environ.get("AUREON_TRUSTED_LIVE_RETRIEVAL", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return True


def trusted_sources_for_domain(domain: str | None) -> tuple[str, ...]:
    key = (domain or "general").strip().lower()
    return TRUSTED_SOURCE_WHITELIST.get(key, TRUSTED_SOURCE_WHITELIST["general"])


def _hostname(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower().split("@")[-1].split(":")[0]
    except Exception:
        return ""
    return _HOST_ALIASES.get(host, host)


def is_trusted_source(url: str, domain: str | None) -> bool:
    host = _hostname(url)
    if not host:
        return False
    allowed = trusted_sources_for_domain(domain)
    return any(host == item or host.endswith(f".{item}") for item in allowed)


def trusted_query(query: str, domain: str | None) -> str:
    sites = " OR ".join(f"site:{host}" for host in trusted_sources_for_domain(domain))
    return f"({query.strip()}) ({sites})"


def _strip_html(raw: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header).*?>.*?</\1>", " ", raw)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    return re.sub(r"\s+", " ", text).strip()


def clean_fetched_text(raw: str, *, limit: int = 6000) -> str:
    text = _strip_html(raw)
    noise = (
        "cookie policy",
        "enable javascript",
        "advertisement",
        "subscribe",
        "sign in",
    )
    chunks = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    useful = [part for part in chunks if not any(n in part.lower() for n in noise)]
    cleaned = " ".join(useful) or text
    return cleaned[:limit].strip()


def fetch_trusted_source(url: str, *, domain: str | None, timeout: int | None = None) -> dict[str, Any] | None:
    if not is_trusted_source(url, domain):
        return None
    try:
        response = requests.get(
            url,
            timeout=timeout or _TIMEOUT,
            headers={"User-Agent": "SOLIA-Aureon/1.0 trusted-retrieval"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.debug("Trusted fetch failed for %s: %s", url, exc)
        return None
    text = clean_fetched_text(response.text)
    if len(text.split()) < 40:
        return None
    return {"url": url, "source": _hostname(url), "text": text}


def trusted_search(
    query: str,
    *,
    domain: str | None,
    max_results: int = 3,
    search_fn: Any | None = None,
    fetch_fn: Any | None = None,
) -> list[dict[str, Any]]:
    """Search only whitelisted sources and return clean fetched text."""
    if not trusted_live_retrieval_enabled():
        return []
    runner = search_fn or _search_ddgs_text
    fetcher = fetch_fn or fetch_trusted_source
    raw_results = runner(trusted_query(query, domain), max_results=max_results * 2)
    docs: list[dict[str, Any]] = []
    for item in raw_results:
        url = str(item.get("url", "")).strip()
        if not url or not is_trusted_source(url, domain):
            continue
        fetched = fetcher(url, domain=domain)
        if fetched:
            title = str(item.get("title") or item.get("text") or fetched["source"]).strip()
            docs.append({**fetched, "title": title[:180]})
        if len(docs) >= max_results:
            break
    return docs


def _search_instant_api(query: str, *, max_results: int) -> list[dict[str, Any]]:
    """DuckDuckGo instant-answer JSON API — good for facts, often empty for news."""
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

    results: list[dict[str, Any]] = []
    abstract = str(data.get("Abstract", "")).strip()
    if abstract:
        results.append({
            "type": "abstract",
            "text": abstract,
            "source": data.get("AbstractSource", "") or "duckduckgo",
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
            "source": data.get("AnswerType", "duckduckgo") or "duckduckgo",
        })
    return results[:max_results]


def _search_ddgs_text(query: str, *, max_results: int) -> list[dict[str, Any]]:
    """Full web text search via ddgs — works for news and live events."""
    try:
        from ddgs import DDGS
    except ImportError:
        logger.debug("ddgs package not installed — text search unavailable")
        return []

    try:
        hits = DDGS().text(query, max_results=max_results)
    except Exception as exc:
        logger.warning("ddgs text search failed: %s", exc)
        return []

    results: list[dict[str, Any]] = []
    for item in hits:
        if not isinstance(item, dict):
            continue
        body = str(item.get("body", "") or item.get("snippet", "")).strip()
        title = str(item.get("title", "")).strip()
        href = str(item.get("href", "") or item.get("url", "")).strip()
        text = body or title
        if not text:
            continue
        if title and title.lower() not in text.lower():
            text = f"{title}: {text}"
        results.append({
            "type": "web",
            "text": text[:500],
            "url": href,
            "source": href.split("/")[2] if href.startswith("http") and "/" in href[8:] else "web",
        })
    return results[:max_results]


def _search_ddgs_news(query: str, *, max_results: int) -> list[dict[str, Any]]:
    """News-index search — returns dated headlines, not evergreen essays."""
    try:
        from ddgs import DDGS
    except ImportError:
        logger.debug("ddgs package not installed — news search unavailable")
        return []

    try:
        hits = DDGS().news(query, max_results=max_results)
    except Exception as exc:
        logger.warning("ddgs news search failed: %s", exc)
        return []

    results: list[dict[str, Any]] = []
    for item in hits:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        body = str(item.get("body", "")).strip()
        href = str(item.get("url", "")).strip()
        source = str(item.get("source", "")).strip()
        if not source and href.startswith("http"):
            parts = href.split("/")
            source = parts[2] if len(parts) > 2 else "web"
        results.append({
            "type": "news",
            "title": title,
            "text": title,
            "body": body[:400],
            "url": href,
            "source": source or "news",
            "date": str(item.get("date", "")).strip(),
        })
    return results[:max_results]


def search(query: str, *, max_results: int = _MAX_RESULTS) -> list[dict[str, Any]]:
    """Search DuckDuckGo and return structured results."""
    global _last_search

    if not web_search_enabled():
        return [{"error": "web search disabled", "source": "duckduckgo"}]

    elapsed = time.time() - _last_search
    if elapsed < _RATE_LIMIT_SECONDS:
        time.sleep(_RATE_LIMIT_SECONDS - elapsed)
    _last_search = time.time()

    effective = rewrite_live_news_query(query) if is_live_news_query(query) else query.strip()

    if is_live_news_query(query):
        try:
            results = _search_ddgs_news(effective, max_results=max_results)
            if results:
                return results
        except Exception as exc:
            logger.debug("News search failed: %s", exc)

        try:
            results = _search_ddgs_text(effective, max_results=max_results)
            if results:
                return results
        except Exception as exc:
            return [{"error": str(exc), "source": "duckduckgo"}]

        return []

    try:
        results = _search_instant_api(effective, max_results=max_results)
        if results:
            return results
    except Exception as exc:
        logger.debug("Instant API search failed: %s", exc)

    try:
        results = _search_ddgs_text(effective, max_results=max_results)
        if results:
            return results
    except Exception as exc:
        return [{"error": str(exc), "source": "duckduckgo"}]

    return []


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
