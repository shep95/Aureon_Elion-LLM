"""Human-in-the-loop label review queue."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select

from db.models import Document, DocumentLabel, KnowledgeDomain, KnowledgeSubdomain
from db.session import get_session


def _serialize_row(
    label: DocumentLabel,
    doc: Document,
    domain: KnowledgeDomain | None,
    subdomain: KnowledgeSubdomain | None = None,
) -> dict[str, Any]:
    return {
        "label_id": label.id,
        "document_id": doc.id,
        "title": doc.title,
        "text_preview": (doc.text or "")[:500],
        "source": doc.source,
        "domain_slug": domain.slug if domain else None,
        "domain_name": domain.name if domain else None,
        "subdomain_slug": subdomain.slug if subdomain else None,
        "subdomain_name": subdomain.name if subdomain else None,
        "proposed_label": label.label,
        "confidence": label.confidence,
        "label_source": label.label_source,
        "needs_review": label.needs_review,
        "created_at": label.created_at.isoformat() if label.created_at else None,
    }


def list_pending_review(
    *,
    limit: int = 50,
    domain_slug: str | None = None,
) -> dict[str, Any]:
    with get_session() as session:
        q = (
            select(DocumentLabel, Document, KnowledgeDomain, KnowledgeSubdomain)
            .join(Document, DocumentLabel.document_id == Document.id)
            .join(KnowledgeDomain, DocumentLabel.domain_id == KnowledgeDomain.id)
            .outerjoin(KnowledgeSubdomain, Document.subdomain_id == KnowledgeSubdomain.id)
            .where(DocumentLabel.needs_review.is_(True))
            .order_by(DocumentLabel.created_at.desc())
            .limit(limit)
        )
        if domain_slug:
            q = q.where(KnowledgeDomain.slug == domain_slug)

        rows = session.execute(q).all()
        pending = [_serialize_row(label, doc, domain, subdomain) for label, doc, domain, subdomain in rows]
        total_pending = session.scalar(
            select(func.count()).select_from(DocumentLabel).where(DocumentLabel.needs_review.is_(True))
        ) or 0

    return {"pending": pending, "count": len(pending), "total_pending": total_pending}


def resolve_label(
    label_id: int,
    *,
    label: str | None = None,
    approve: bool = True,
) -> dict[str, Any]:
    with get_session() as session:
        row = session.execute(
            select(DocumentLabel, Document, KnowledgeDomain, KnowledgeSubdomain)
            .join(Document, DocumentLabel.document_id == Document.id)
            .join(KnowledgeDomain, DocumentLabel.domain_id == KnowledgeDomain.id)
            .outerjoin(KnowledgeSubdomain, Document.subdomain_id == KnowledgeSubdomain.id)
            .where(DocumentLabel.id == label_id)
        ).first()
        if not row:
            return {"ok": False, "error": "label_not_found"}

        doc_label, doc, domain, subdomain = row
        if not doc_label.needs_review:
            return {"ok": False, "error": "label_not_in_review_queue"}

        if approve:
            if label:
                doc_label.label = label.strip()[:64]
            doc_label.needs_review = False
            session.commit()
            return {
                "ok": True,
                "action": "approved",
                "label": _serialize_row(doc_label, doc, domain, subdomain),
            }

        session.delete(doc_label)
        session.commit()
        return {"ok": True, "action": "rejected", "label_id": label_id}
