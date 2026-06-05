"""Brain API helpers."""

from __future__ import annotations

from typing import Any

from brain.cortex import (
    bootstrap_brain,
    brain_status,
    run_domain_cycle,
    run_full_brain,
    run_subdomain_cycle,
)
from brain.domains.taxonomy import KNOWLEDGE_TAXONOMY, total_subdomains


def get_taxonomy() -> dict[str, Any]:
    return {
        "domains": KNOWLEDGE_TAXONOMY,
        "domain_count": len(KNOWLEDGE_TAXONOMY),
        "subdomain_count": total_subdomains(),
    }
