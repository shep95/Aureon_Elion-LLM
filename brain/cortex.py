"""Cortex — coordinates all brain regions across every knowledge domain."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select

from brain.base import AgentContext, AgentResult
from brain.domains.taxonomy import KNOWLEDGE_TAXONOMY, total_subdomains
from brain.regions.collector import CollectorAgent
from brain.regions.evaluator import EvaluatorAgent
from brain.regions.labeler import LabelerAgent
from brain.regions.reward import RewardAgent
from brain.regions.trainer import TrainerAgent
from brain.regions.verifier import VerifierAgent
from db.models import Document, KnowledgeDomain, KnowledgeSubdomain, MicroAgent
from db.seed import seed_knowledge_taxonomy
from db.session import get_session, init_db

REGION_ORDER = (
    ("collector", CollectorAgent),
    ("verifier", VerifierAgent),
    ("labeler", LabelerAgent),
    ("trainer", TrainerAgent),
    ("evaluator", EvaluatorAgent),
    ("reward", RewardAgent),
)


def bootstrap_brain() -> dict[str, int]:
    """Create tables and seed all domains, subdomains, and micro-agents."""
    init_db()
    with get_session() as session:
        return seed_knowledge_taxonomy(session)


def run_subdomain_cycle(
    domain_slug: str,
    subdomain_slug: str,
    epochs: int = 200,
) -> dict[str, Any]:
    """Run all 6 micro-agents for one subdomain — like one specialized brain circuit."""
    bootstrap_brain()
    results: list[dict] = []

    with get_session() as session:
        domain = session.scalar(
            select(KnowledgeDomain).where(KnowledgeDomain.slug == domain_slug)
        )
        if not domain:
            return {"error": f"unknown domain: {domain_slug}"}

        subdomain = session.scalar(
            select(KnowledgeSubdomain).where(
                KnowledgeSubdomain.domain_id == domain.id,
                KnowledgeSubdomain.slug == subdomain_slug,
            )
        )
        if not subdomain:
            return {"error": f"unknown subdomain: {subdomain_slug}"}

        ctx = AgentContext(
            domain_slug=domain_slug,
            subdomain_slug=subdomain_slug,
            domain_id=domain.id,
            subdomain_id=subdomain.id,
            epochs=epochs,
        )

        for region_name, agent_cls in REGION_ORDER:
            agent_row = session.scalar(
                select(MicroAgent).where(
                    MicroAgent.region == region_name,
                    MicroAgent.domain_id == domain.id,
                    MicroAgent.subdomain_id == subdomain.id,
                )
            )
            if not agent_row:
                continue
            agent_impl = agent_cls()
            result = agent_impl.execute(session, agent_row, ctx)
            results.append(
                {
                    "region": region_name,
                    "status": result.status,
                    "metrics": result.metrics,
                    "error": result.error,
                }
            )

    return {
        "domain": domain_slug,
        "subdomain": subdomain_slug,
        "regions": results,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def run_domain_cycle(domain_slug: str, epochs: int = 200) -> dict[str, Any]:
    """Run all subdomains within one knowledge domain."""
    subdomains = KNOWLEDGE_TAXONOMY.get(domain_slug, [])
    cycles = [run_subdomain_cycle(domain_slug, sub, epochs=epochs) for sub in subdomains]
    return {
        "domain": domain_slug,
        "subdomains_processed": len(cycles),
        "cycles": cycles,
    }


def run_full_brain(
    epochs: int = 200,
    domain_limit: int | None = None,
    subdomain_limit: int | None = 1,
) -> dict[str, Any]:
    """
    Train across human knowledge domains and subdomains.

    domain_limit: max top-level domains (None = all 29)
    subdomain_limit: max subdomains per domain (None = all)
    """
    bootstrap_brain()
    started = datetime.now(timezone.utc).isoformat()
    domains = list(KNOWLEDGE_TAXONOMY.keys())
    if domain_limit:
        domains = domains[:domain_limit]

    domain_results: list[dict] = []
    total_cycles = 0

    for domain_slug in domains:
        subs = KNOWLEDGE_TAXONOMY[domain_slug]
        if subdomain_limit:
            subs = subs[:subdomain_limit]
        cycles = []
        for sub in subs:
            cycles.append(run_subdomain_cycle(domain_slug, sub, epochs=epochs))
            total_cycles += 1
        domain_results.append({"domain": domain_slug, "subdomain_cycles": cycles})

    with get_session() as session:
        doc_count = session.scalar(select(func.count()).select_from(Document)) or 0
        agent_count = session.scalar(select(func.count()).select_from(MicroAgent)) or 0

    return {
        "architecture": "micro_algorithms_per_brain_region",
        "regions": [r[0] for r in REGION_ORDER],
        "total_domains": len(KNOWLEDGE_TAXONOMY),
        "total_subdomains": total_subdomains(),
        "domains_processed": len(domains),
        "cycles_executed": total_cycles,
        "documents_in_db": doc_count,
        "micro_agents_registered": agent_count,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "domain_results": domain_results,
    }


def brain_status() -> dict[str, Any]:
    """Snapshot of brain state from PostgreSQL."""
    init_db()
    with get_session() as session:
        domain_count = session.scalar(select(func.count()).select_from(KnowledgeDomain)) or 0
        subdomain_count = session.scalar(select(func.count()).select_from(KnowledgeSubdomain)) or 0
        agent_count = session.scalar(select(func.count()).select_from(MicroAgent)) or 0
        doc_count = session.scalar(select(func.count()).select_from(Document)) or 0
        verified = (
            session.scalar(
                select(func.count()).select_from(Document).where(Document.verified.is_(True))
            )
            or 0
        )

        if domain_count == 0:
            seed_stats = seed_knowledge_taxonomy(session)
            domain_count = seed_stats["domains"]
            subdomain_count = seed_stats["subdomains"]
            agent_count = seed_stats["agents"]

    return {
        "domains": domain_count,
        "subdomains": subdomain_count,
        "micro_agents": agent_count,
        "documents": doc_count,
        "verified_documents": verified,
        "regions": [r[0] for r in REGION_ORDER],
        "taxonomy_domains": list(KNOWLEDGE_TAXONOMY.keys()),
    }
