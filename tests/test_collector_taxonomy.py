"""Collector taxonomy seeding tests."""

from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from brain.base import AgentContext
from brain.grades import get_grade
from brain.regions.collector import CollectorAgent, _topic_seed_text
from db.models import Base, Document, KnowledgeDomain, KnowledgeMicroSubdomain, KnowledgeSubdomain


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _chemistry_ctx(session: Session) -> AgentContext:
    domain = KnowledgeDomain(slug="science_and_natural_philosophy", name="Science")
    session.add(domain)
    session.flush()
    subdomain = KnowledgeSubdomain(
        domain_id=domain.id, slug="chemistry", name="Chemistry"
    )
    session.add(subdomain)
    session.flush()
    micro = KnowledgeMicroSubdomain(
        domain_id=domain.id,
        subdomain_id=subdomain.id,
        slug="physical_chemistry",
        name="Physical Chemistry",
    )
    session.add(micro)
    session.flush()
    grade = get_grade("preschool")
    assert grade is not None
    return AgentContext(
        domain_slug=domain.slug,
        subdomain_slug=subdomain.slug,
        micro_subdomain_slug=micro.slug,
        domain_id=domain.id,
        subdomain_id=subdomain.id,
        micro_subdomain_id=micro.id,
        grade_slug=grade.slug,
        grade=grade,
    )


def test_topic_seed_text_meets_min_length():
    text = _topic_seed_text(
        "Chemical Kinetics",
        {
            "domain": "Science and Natural Philosophy",
            "subdomain": "Chemistry",
            "micro_subdomain": "Physical Chemistry",
        },
    )
    assert len(text) >= 120


def test_collector_ingests_taxonomy_topics():
    agent = CollectorAgent()
    with _session() as session:
        ctx = _chemistry_ctx(session)
        result = agent.run(session, ctx)
        session.commit()

        docs = session.scalars(
            select(Document).where(Document.micro_subdomain_id == ctx.micro_subdomain_id)
        ).all()

    assert result.metrics["collection_attempts"] >= 3
    assert result.metrics["new_documents"] >= 3
    assert len(docs) >= 3
    taxonomy_docs = [d for d in docs if d.source == "taxonomy"]
    assert len(taxonomy_docs) >= 3
    assert all((d.extra or {}).get("topic") for d in taxonomy_docs)


def test_collector_skips_duplicate_topics_on_rerun():
    agent = CollectorAgent()
    with _session() as session:
        ctx = _chemistry_ctx(session)
        first = agent.run(session, ctx)
        session.commit()
        second = agent.run(session, ctx)
        session.commit()

    assert first.metrics["new_documents"] >= 3
    assert second.metrics["new_documents"] == 0
