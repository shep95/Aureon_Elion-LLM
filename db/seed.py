"""Database seeding and repository helpers."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from brain.domains.taxonomy import KNOWLEDGE_TAXONOMY, lookup_names
from brain.graduation import seed_grade_progress
from db.models import (
    KnowledgeDomain,
    KnowledgeMicroSubdomain,
    KnowledgeSubdomain,
    MicroAgent,
)

BRAIN_REGIONS = ("collector", "verifier", "labeler", "trainer", "evaluator", "reward")


def seed_knowledge_taxonomy(session: Session) -> dict[str, int]:
    """Insert domains, subdomains, micro-subdomains, and micro-agents."""
    stats = {"domains": 0, "subdomains": 0, "micro_subdomains": 0, "agents": 0}

    for domain_slug, subdomain_map in KNOWLEDGE_TAXONOMY.items():
        domain = session.scalar(
            select(KnowledgeDomain).where(KnowledgeDomain.slug == domain_slug)
        )
        if not domain:
            names = lookup_names(domain_slug)
            domain = KnowledgeDomain(
                slug=domain_slug,
                name=names.get("domain", domain_slug.replace("_", " ").title()),
                description=f"Knowledge domain: {names.get('domain', domain_slug)}",
            )
            session.add(domain)
            session.flush()
            stats["domains"] += 1

        for sub_slug, micro_slugs in subdomain_map.items():
            subdomain = session.scalar(
                select(KnowledgeSubdomain).where(
                    KnowledgeSubdomain.domain_id == domain.id,
                    KnowledgeSubdomain.slug == sub_slug,
                )
            )
            if not subdomain:
                names = lookup_names(domain_slug, subdomain=sub_slug)
                subdomain = KnowledgeSubdomain(
                    domain_id=domain.id,
                    slug=sub_slug,
                    name=names.get("subdomain", sub_slug.replace("_", " ").title()),
                )
                session.add(subdomain)
                session.flush()
                stats["subdomains"] += 1

            for micro_slug in micro_slugs:
                micro = session.scalar(
                    select(KnowledgeMicroSubdomain).where(
                        KnowledgeMicroSubdomain.subdomain_id == subdomain.id,
                        KnowledgeMicroSubdomain.slug == micro_slug,
                    )
                )
                if not micro:
                    names = lookup_names(domain_slug, subdomain=sub_slug, micro=micro_slug)
                    micro = KnowledgeMicroSubdomain(
                        domain_id=domain.id,
                        subdomain_id=subdomain.id,
                        slug=micro_slug,
                        name=names.get("micro_subdomain", micro_slug.replace("_", " ").title()),
                    )
                    session.add(micro)
                    session.flush()
                    stats["micro_subdomains"] += 1

                for region in BRAIN_REGIONS:
                    exists = session.scalar(
                        select(MicroAgent).where(
                            MicroAgent.region == region,
                            MicroAgent.domain_id == domain.id,
                            MicroAgent.subdomain_id == subdomain.id,
                            MicroAgent.micro_subdomain_id == micro.id,
                        )
                    )
                    if not exists:
                        session.add(
                            MicroAgent(
                                region=region,
                                domain_id=domain.id,
                                subdomain_id=subdomain.id,
                                micro_subdomain_id=micro.id,
                                config={
                                    "scope": f"{domain_slug}.{sub_slug}.{micro_slug}",
                                    "level": "micro_subdomain",
                                },
                            )
                        )
                        stats["agents"] += 1

            # Subdomain coordinator agents (orchestrate micro-subdomains)
            for region in BRAIN_REGIONS:
                exists = session.scalar(
                    select(MicroAgent).where(
                        MicroAgent.region == region,
                        MicroAgent.domain_id == domain.id,
                        MicroAgent.subdomain_id == subdomain.id,
                        MicroAgent.micro_subdomain_id.is_(None),
                    )
                )
                if not exists:
                    session.add(
                        MicroAgent(
                            region=region,
                            domain_id=domain.id,
                            subdomain_id=subdomain.id,
                            micro_subdomain_id=None,
                            config={
                                "scope": f"{domain_slug}.{sub_slug}",
                                "level": "subdomain",
                            },
                        )
                    )
                    stats["agents"] += 1

        # Domain-level coordinator agents
        for region in BRAIN_REGIONS:
            exists = session.scalar(
                select(MicroAgent).where(
                    MicroAgent.region == region,
                    MicroAgent.domain_id == domain.id,
                    MicroAgent.subdomain_id.is_(None),
                    MicroAgent.micro_subdomain_id.is_(None),
                )
            )
            if not exists:
                session.add(
                    MicroAgent(
                        region=region,
                        domain_id=domain.id,
                        subdomain_id=None,
                        micro_subdomain_id=None,
                        config={"scope": domain_slug, "level": "domain"},
                    )
                )
                stats["agents"] += 1

    session.commit()
    grade_rows = seed_grade_progress(session)
    session.commit()
    stats["grade_progress_rows"] = grade_rows
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


def get_micro_subdomain(
    session: Session,
    domain_slug: str,
    subdomain_slug: str,
    micro_slug: str,
) -> KnowledgeMicroSubdomain | None:
    subdomain = get_subdomain(session, domain_slug, subdomain_slug)
    if not subdomain:
        return None
    return session.scalar(
        select(KnowledgeMicroSubdomain).where(
            KnowledgeMicroSubdomain.subdomain_id == subdomain.id,
            KnowledgeMicroSubdomain.slug == micro_slug,
        )
    )
