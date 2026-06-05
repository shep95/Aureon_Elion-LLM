"""Human knowledge taxonomy — Zophiel complete domain architecture."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_TAXONOMY_PATH = Path(__file__).resolve().parent / "knowledge_taxonomy.json"


@lru_cache(maxsize=1)
def _load_payload() -> dict[str, Any]:
    if not _TAXONOMY_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {_TAXONOMY_PATH.name}. Run: python brain/domains/build_zophiel_taxonomy.py"
        )
    return json.loads(_TAXONOMY_PATH.read_text(encoding="utf-8"))


def taxonomy_stats() -> dict[str, int]:
    return dict(_load_payload().get("stats", {}))


# domain → subdomain → [micro_subdomain, ...]
KNOWLEDGE_TAXONOMY: dict[str, dict[str, list[str]]] = _load_payload()["training_tree"]


def taxonomy_catalog() -> dict[str, Any]:
    """Full metadata including display names and leaf topics."""
    return _load_payload()["domains"]


def all_domain_slugs() -> list[str]:
    return list(KNOWLEDGE_TAXONOMY.keys())


def subdomains_for(domain: str) -> list[str]:
    return list(KNOWLEDGE_TAXONOMY.get(domain, {}).keys())


def micro_subdomains_for(domain: str, subdomain: str) -> list[str]:
    return list(KNOWLEDGE_TAXONOMY.get(domain, {}).get(subdomain, []))


def all_subdomain_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for domain, subs in KNOWLEDGE_TAXONOMY.items():
        for sub in subs:
            pairs.append((domain, sub))
    return pairs


def all_micro_triples() -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    for domain, subs in KNOWLEDGE_TAXONOMY.items():
        for sub, micros in subs.items():
            for micro in micros:
                triples.append((domain, sub, micro))
    return triples


def total_subdomains() -> int:
    return sum(len(subs) for subs in KNOWLEDGE_TAXONOMY.values())


def total_micro_subdomains() -> int:
    return sum(len(micros) for subs in KNOWLEDGE_TAXONOMY.values() for micros in subs.values())


def lookup_names(domain: str, subdomain: str | None = None, micro: str | None = None) -> dict[str, str]:
    """Resolve human-readable names from the Zophiel catalog."""
    catalog = taxonomy_catalog()
    out: dict[str, str] = {}
    domain_entry = catalog.get(domain, {})
    if domain_entry:
        out["domain"] = domain_entry.get("name", domain.replace("_", " ").title())
    if subdomain and domain_entry:
        sub_entry = domain_entry.get("subdomains", {}).get(subdomain, {})
        out["subdomain"] = sub_entry.get("name", subdomain.replace("_", " ").title())
        if micro:
            micro_entry = sub_entry.get("micro_subdomains", {}).get(micro, {})
            out["micro_subdomain"] = micro_entry.get("name", micro.replace("_", " ").title())
    return out
