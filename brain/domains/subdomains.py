"""Human knowledge domain taxonomy — domains and subdomains (flat index)."""

from __future__ import annotations

from brain.domains.taxonomy import KNOWLEDGE_TAXONOMY

# Derived from Zophiel COMPLETE_HUMAN_DOMAIN_TAXONOMY (30 domains).
SUBDOMAIN_TAXONOMY: dict[str, list[str]] = {
    domain: list(subdomains.keys()) for domain, subdomains in KNOWLEDGE_TAXONOMY.items()
}


def all_domain_slugs() -> list[str]:
    return list(SUBDOMAIN_TAXONOMY.keys())
