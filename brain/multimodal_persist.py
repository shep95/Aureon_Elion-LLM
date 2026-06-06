"""Persist multimodal collector output to DB — shared by brain cycle and API ingest."""

from __future__ import annotations

from sqlalchemy import select

from db.models import Document, KnowledgeDomain, KnowledgeSubdomain
from db.seed import get_micro_subdomain
from db.session import get_session, init_db
from pipeline.step1_collection.collectors import RawDocument
from pipeline.step1_collection.filters import filter_document


def persist_multimodal_docs(docs: list[RawDocument]) -> dict[str, int]:
    """Persist RawDocument list from MultimodalCollector into documents table."""
    init_db()
    added = 0
    skipped = 0

    domain_slug = "technology_and_engineering"
    subdomain_slug = "computer_science"
    micro_slug = "data_structures"

    with get_session() as session:
        domain = session.scalar(select(KnowledgeDomain).where(KnowledgeDomain.slug == domain_slug))
        if not domain:
            return {"added": 0, "skipped": len(docs)}

        for doc in docs:
            ok, quality, _ = filter_document(doc)
            if not ok:
                skipped += 1
                continue
            digest = doc.content_hash()
            if session.scalar(select(Document).where(Document.content_hash == digest)):
                skipped += 1
                continue
            micro = get_micro_subdomain(session, domain_slug, subdomain_slug, micro_slug)
            if not micro:
                skipped += 1
                continue
            subdomain = session.scalar(
                select(KnowledgeSubdomain).where(
                    KnowledgeSubdomain.domain_id == domain.id,
                    KnowledgeSubdomain.slug == subdomain_slug,
                )
            )
            if not subdomain:
                skipped += 1
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
                    quality_score=quality,
                    verified=False,
                    content_hash=digest,
                    extra=doc.metadata,
                )
            )
            added += 1
        session.commit()

    if added:
        try:
            from brain.vector_rag import invalidate_rag_index

            invalidate_rag_index()
        except Exception:
            pass

    return {"added": added, "skipped": skipped}
