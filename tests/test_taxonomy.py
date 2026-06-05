"""Zophiel complete human domain taxonomy tests."""

from __future__ import annotations

from brain.domains.taxonomy import (
    KNOWLEDGE_TAXONOMY,
    taxonomy_stats,
    total_micro_subdomains,
    total_subdomains,
)


def test_zophiel_taxonomy_loaded():
    stats = taxonomy_stats()
    assert stats["domains"] == 30
    assert stats["subdomains"] == 135
    assert stats["micro_subdomains"] == 862
    assert len(KNOWLEDGE_TAXONOMY) == 30
    assert total_subdomains() == 135
    assert total_micro_subdomains() == 862


def test_sample_domain_structure():
    science = KNOWLEDGE_TAXONOMY.get("science_and_natural_philosophy")
    assert science is not None
    assert "physics" in science
    assert "classical_mechanics" in science["physics"]
