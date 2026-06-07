"""Trusted live retrieval tests."""

from __future__ import annotations

from brain.web_search import is_trusted_source, trusted_search, trusted_sources_for_domain


def test_trusted_sources_for_domain():
    assert "arxiv.org" in trusted_sources_for_domain("physics")
    assert "pubmed.ncbi.nlm.nih.gov" in trusted_sources_for_domain("psychology")
    assert "imf.org" in trusted_sources_for_domain("economics")


def test_is_trusted_source_blocks_forums():
    assert is_trusted_source("https://arxiv.org/abs/1234.5", "physics") is True
    assert is_trusted_source("https://reddit.com/r/physics/comments/1", "physics") is False


def test_trusted_search_fetches_only_whitelisted_sources(monkeypatch):
    monkeypatch.setenv("AUREON_TRUSTED_LIVE_RETRIEVAL", "1")

    def fake_search(_query: str, *, max_results: int):
        return [
            {"url": "https://reddit.com/r/physics/comments/x", "title": "Forum"},
            {"url": "https://arxiv.org/abs/2401.00001", "title": "Quantum paper"},
        ][:max_results]

    def fake_fetch(url: str, *, domain: str | None):
        return {
            "url": url,
            "source": "arxiv.org",
            "text": "Quantum computing uses qubits, gates, superposition, measurement, and entanglement in physics.",
        }

    docs = trusted_search(
        "quantum computing",
        domain="physics",
        max_results=2,
        search_fn=fake_search,
        fetch_fn=fake_fetch,
    )
    assert len(docs) == 1
    assert docs[0]["url"].startswith("https://arxiv.org")

