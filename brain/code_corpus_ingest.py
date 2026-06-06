"""Ingest HumanEval + MBPP into the document database."""

from __future__ import annotations

from sqlalchemy import select

from brain.regions.code_collector import CodeCollector
from db.models import Document, KnowledgeDomain, KnowledgeSubdomain
from db.seed import get_micro_subdomain
from db.session import get_session, init_db


def ingest_code_corpus(*, limit: int = 2000) -> int:
    init_db()
    docs = CodeCollector().collect(limit=limit)
    added = 0

    with get_session() as session:
        domain = session.scalar(
            select(KnowledgeDomain).where(KnowledgeDomain.slug == "technology_and_engineering")
        )
        if not domain:
            return 0

        subdomain = session.scalar(
            select(KnowledgeSubdomain).where(
                KnowledgeSubdomain.domain_id == domain.id,
                KnowledgeSubdomain.slug == "computer_science",
            )
        )
        if not subdomain:
            return 0

        for doc in docs:
            micro_slug = str(doc.metadata.get("micro_subdomain", "python_functions"))
            micro = get_micro_subdomain(session, domain.id, subdomain.id, micro_slug)
            if not micro:
                continue

            digest = doc.content_hash()
            if session.scalar(select(Document).where(Document.content_hash == digest)):
                continue

            session.add(
                Document(
                    domain_id=domain.id,
                    subdomain_id=subdomain.id,
                    micro_subdomain_id=micro.id,
                    source=doc.source,
                    title=doc.title,
                    text=doc.text,
                    url=doc.url,
                    language=doc.language,
                    quality_score=0.85,
                    verified=True,
                    content_hash=digest,
                    extra=doc.metadata,
                )
            )
            added += 1
        session.commit()

    if added:
        from brain.vector_rag import invalidate_rag_index

        invalidate_rag_index()

    return added
