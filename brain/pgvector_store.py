"""Pgvector-style semantic store — JSON embeddings on Document.extra (Tier 4 upgrade path)."""

from __future__ import annotations

import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)


def pgvector_enabled() -> bool:
    return os.environ.get("AUREON_PGVECTOR", "1").strip().lower() not in ("0", "false", "no")


def index_document_embedding(document_id: int, embedding: list[float]) -> None:
    if not pgvector_enabled() or not embedding:
        return
    from sqlalchemy import select

    from db.models import Document
    from db.session import get_session

    with get_session() as session:
        doc = session.get(Document, document_id)
        if not doc:
            return
        extra = dict(doc.extra or {})
        extra["embedding"] = embedding
        extra["embedding_dims"] = len(embedding)
        doc.extra = extra
        session.commit()


def search_similar(text: str, *, top_k: int = 5) -> list[dict[str, Any]]:
    """Cosine similarity over stored embeddings — falls back to empty if disabled."""
    if not pgvector_enabled():
        return []

    from brain.multimodal_processors import text_embedding
    from sqlalchemy import select

    from db.models import Document
    from db.session import get_session

    query_vec = text_embedding(text)
    hits: list[tuple[float, Document]] = []

    with get_session() as session:
        rows = session.scalars(select(Document).limit(5000)).all()
        for doc in rows:
            extra = doc.extra or {}
            emb = extra.get("embedding")
            if not isinstance(emb, list) or len(emb) != len(query_vec):
                continue
            score = _cosine(query_vec, emb)
            if score > 0.05:
                hits.append((score, doc))

    hits.sort(key=lambda h: h[0], reverse=True)
    return [
        {
            "document_id": doc.id,
            "title": doc.title,
            "source": doc.source,
            "score": round(score, 4),
            "modality": (doc.extra or {}).get("modality", "text"),
        }
        for score, doc in hits[:top_k]
    ]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def status() -> dict[str, Any]:
    from sqlalchemy import func, select

    from db.models import Document
    from db.session import get_session

    indexed = 0
    try:
        with get_session() as session:
            total = session.scalar(select(func.count()).select_from(Document)) or 0
            rows = session.scalars(select(Document).limit(1000)).all()
            indexed = sum(1 for d in rows if (d.extra or {}).get("embedding"))
    except Exception:
        total = 0
    return {
        "enabled": pgvector_enabled(),
        "documents_total": total,
        "documents_with_embeddings": indexed,
        "backend": "json_extra_cosine",
        "note": "Native PostgreSQL pgvector extension can replace this layer when enabled.",
    }
