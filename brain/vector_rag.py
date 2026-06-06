"""Vector RAG — TF-IDF retrieval over PostgreSQL corpus with verified citations."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_rebuild_lock = threading.Lock()
_index: VectorRAGIndex | None = None
_rebuild_in_progress = False


@dataclass(frozen=True)
class RagHit:
    document_id: int
    content_hash: str
    title: str
    text: str
    source: str
    score: float

    def citation(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "content_hash": self.content_hash,
            "title": self.title,
            "source": self.source,
            "score": round(self.score, 4),
        }

    def snippet(self, max_chars: int = 800) -> str:
        body = f"{self.title} {self.text}".strip()
        return body[:max_chars]


class VectorRAGIndex:
    """In-memory TF-IDF index rebuilt from documents + seeds."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(
            max_features=int(os.environ.get("AUREON_RAG_MAX_FEATURES", "8192")),
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        self._matrix = None
        self._hits: list[RagHit] = []
        self._built_at = 0.0

    @property
    def document_count(self) -> int:
        return len(self._hits)

    def rebuild(self) -> int:
        from brain.predict_engine import _load_seed_documents
        from sqlalchemy import select

        from db.models import Document
        from db.session import get_session

        limit = int(os.environ.get("AUREON_PREDICT_DOC_LIMIT", "1000000"))
        limit = max(100, min(limit, 1_000_000))

        hits: list[RagHit] = []
        try:
            with get_session() as session:
                rows = session.scalars(select(Document).limit(limit)).all()
                for row in rows:
                    if not row.text:
                        continue
                    hits.append(
                        RagHit(
                            document_id=row.id,
                            content_hash=row.content_hash,
                            title=row.title or "",
                            text=row.text,
                            source=row.source,
                            score=0.0,
                        )
                    )
        except Exception:
            logger.debug("RAG DB load skipped", exc_info=True)

        for i, seed_text in enumerate(_load_seed_documents()):
            hits.append(
                RagHit(
                    document_id=-(i + 1),
                    content_hash=f"seed_{i}",
                    title=seed_text[:80],
                    text=seed_text,
                    source="seeds",
                    score=0.0,
                )
            )

        if not hits:
            self._matrix = None
            self._hits = []
            self._built_at = time.time()
            return 0

        corpus = [h.snippet() for h in hits]
        self._matrix = self._vectorizer.fit_transform(corpus)
        self._hits = hits
        self._built_at = time.time()
        logger.info("Vector RAG index rebuilt — %s documents", len(hits))
        return len(hits)

    def retrieve(self, query: str, *, top_k: int = 8, min_score: float = 0.05) -> list[RagHit]:
        if self._matrix is None or not self._hits:
            return []
        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix)[0]
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        out: list[RagHit] = []
        for idx, score in ranked[:top_k]:
            if score < min_score:
                break
            base = self._hits[idx]
            out.append(
                RagHit(
                    document_id=base.document_id,
                    content_hash=base.content_hash,
                    title=base.title,
                    text=base.text,
                    source=base.source,
                    score=float(score),
                )
            )
        return out


def _schedule_background_rebuild() -> None:
    global _rebuild_in_progress

    def _job() -> None:
        global _index, _rebuild_in_progress
        try:
            fresh = VectorRAGIndex()
            count = fresh.rebuild()
            with _lock:
                _index = fresh
            logger.info("Vector RAG background rebuild complete — %s docs", count)
        except Exception:
            logger.exception("Vector RAG background rebuild failed")
        finally:
            with _rebuild_lock:
                _rebuild_in_progress = False

    with _rebuild_lock:
        if _rebuild_in_progress:
            return
        _rebuild_in_progress = True
    threading.Thread(target=_job, name="aureon-rag-rebuild", daemon=True).start()


def get_rag_index(*, force_rebuild: bool = False) -> VectorRAGIndex:
    """Return the current index; rebuild stale indexes in the background only."""
    global _index
    ttl = float(os.environ.get("AUREON_RAG_TTL_SEC", "300"))
    with _lock:
        if _index is None:
            _index = VectorRAGIndex()
            _index.rebuild()
            return _index

        stale = force_rebuild or (time.time() - _index._built_at) > ttl
        if stale and _index.document_count > 0:
            _schedule_background_rebuild()
            return _index

        if stale and _index.document_count == 0:
            _index.rebuild()
        return _index


def retrieve_with_citations(
    query: str,
    *,
    top_k: int | None = None,
    max_words: int | None = None,
) -> tuple[str, list[RagHit], list[dict[str, Any]]]:
    """Return context text, hits, and citation dicts for chat/predict."""
    if top_k is None:
        top_k = int(os.environ.get("AUREON_RAG_TOP_K", "8"))
    if max_words is None:
        max_words = int(os.environ.get("AUREON_PREDICT_CONTEXT_WORDS", "1000000"))
        max_words = max(50, min(max_words, 1_000_000))

    hits = get_rag_index().retrieve(query, top_k=top_k)
    citations = [h.citation() for h in hits]

    words: list[str] = []
    for hit in hits:
        for word in hit.snippet().lower().split():
            words.append(word)
            if len(words) >= max_words:
                break
        if len(words) >= max_words:
            break

    return " ".join(words[:max_words]), hits, citations


def invalidate_rag_index() -> None:
    global _index
    with _lock:
        if _index is not None:
            _index._built_at = 0.0
