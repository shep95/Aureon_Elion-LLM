"""Micro-subdomain helpers — taxonomy is fully defined in knowledge_taxonomy.json."""

from __future__ import annotations

from brain.domains.taxonomy import KNOWLEDGE_TAXONOMY, taxonomy_catalog


def micro_subdomains_for(domain: str, subdomain: str) -> list[str]:
    return list(KNOWLEDGE_TAXONOMY.get(domain, {}).get(subdomain, []))


def build_full_taxonomy() -> dict[str, dict[str, list[str]]]:
    return KNOWLEDGE_TAXONOMY


def topics_for(domain: str, subdomain: str, micro: str) -> list[str]:
    """Leaf topics under a micro-subdomain (for document seeding)."""
    entry = (
        taxonomy_catalog()
        .get(domain, {})
        .get("subdomains", {})
        .get(subdomain, {})
        .get("micro_subdomains", {})
        .get(micro, {})
    )
    return list(entry.get("topics", []))
