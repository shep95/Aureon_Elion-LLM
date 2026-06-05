"""Database seeding and repository helpers."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from brain.domains.taxonomy import KNOWLEDGE_TAXONOMY
from db.models import KnowledgeDomain, KnowledgeSubdomain, MicroAgent

BRAIN_REGIONS = ("collector", "verifier", "labeler", "trainer", "evaluator", "reward")


def seed_knowledge_taxonomy(session: Session) -> dict[str, int]:
    """Insert all human knowledge domains, subdomains, and micro-agents."""
    stats = {"domains": 0, "subdomains": 0, "agents": 0}

    for domain_slug, subdomain_slugs in KNOWLEDGE_TAXONOMY.items():
        domain = session.scalar(
            select(KnowledgeDomain).where(KnowledgeDomain.slug == domain_slug)
        )
        if not domain:
            domain = KnowledgeDomain(
                slug=domain_slug,
                name=domain_slug.replace("_", " ").title(),
                description=f"Knowledge domain: {domain_slug}",
            )
            session.add(domain)
            session.flush()
            stats["domains"] += 1

        for sub_slug in subdomain_slugs:
            subdomain = session.scalar(
                select(KnowledgeSubdomain).where(
                    KnowledgeSubdomain.domain_id == domain.id,
                    KnowledgeSubdomain.slug == sub_slug,
                )
            )
            if not subdomain:
                subdomain = KnowledgeSubdomain(
                    domain_id=domain.id,
                    slug=sub_slug,
                    name=sub_slug.replace("_", " ").title(),
                )
                session.add(subdomain)
                session.flush()
                stats["subdomains"] += 1

            for region in BRAIN_REGIONS:
                exists = session.scalar(
                    select(MicroAgent).where(
                        MicroAgent.region == region,
                        MicroAgent.domain_id == domain.id,
                        MicroAgent.subdomain_id == subdomain.id,
                    )
                )
                if not exists:
                    session.add(
                        MicroAgent(
                            region=region,
                            domain_id=domain.id,
                            subdomain_id=subdomain.id,
                            config={"scope": f"{domain_slug}.{sub_slug}"},
                        )
                    )
                    stats["agents"] += 1

        # Domain-level agents (coordinate across subdomains)
        for region in BRAIN_REGIONS:
            exists = session.scalar(
                select(MicroAgent).where(
                    MicroAgent.region == region,
                    MicroAgent.domain_id == domain.id,
                    MicroAgent.subdomain_id.is_(None),
                )
            )
            if not exists:
                session.add(
                    MicroAgent(
                        region=region,
                        domain_id=domain.id,
                        subdomain_id=None,
                        config={"scope": domain_slug, "level": "domain"},
                    )
                )
                stats["agents"] += 1

    session.commit()
    return stats


def get_domain_by_slug(session: Session, slug: str) -> KnowledgeDomain | None:
    return session.scalar(select(KnowledgeDomain).where(KnowledgeDomain.slug == slug))


def get_subdomain(
    session: Session, domain_slug: str, subdomain_slug: str
) -> KnowledgeSubdomain | None:
    domain = get_domain_by_slug(session, domain_slug)
    if not domain:
        return None
    return session.scalar(
        select(KnowledgeSubdomain).where(
            KnowledgeSubdomain.domain_id == domain.id,
            KnowledgeSubdomain.slug == subdomain_slug,
        )
    )
