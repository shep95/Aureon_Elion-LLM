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

from app.security import load_json_file_bounded
from pipeline.config import SEEDS_DIR

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
    metadata: dict[str, Any] | None = None

    def citation(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "content_hash": self.content_hash,
            "title": self.title,
            "source": self.source,
            "score": round(self.score, 4),
            "domain": (self.metadata or {}).get("domain"),
            "subdomain": (self.metadata or {}).get("subdomain"),
            "micro_subdomain": (self.metadata or {}).get("micro_subdomain"),
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

    @staticmethod
    def _hit_path(hit: RagHit) -> str | None:
        meta = hit.metadata or {}
        domain = str(meta.get("domain") or "").strip()
        subdomain = str(meta.get("subdomain") or "").strip()
        micro = str(meta.get("micro_subdomain") or "").strip()
        if domain and subdomain and micro:
            return f"{domain}.{subdomain}.{micro}"
        return None

    @classmethod
    def _hit_allowed(
        cls,
        hit: RagHit,
        *,
        domain: str | None,
        paths: set[str] | None,
    ) -> bool:
        if not domain and not paths:
            return True
        meta = hit.metadata or {}
        hit_domain = str(meta.get("domain") or "").strip()
        hit_path = cls._hit_path(hit)
        if paths and hit_path in paths:
            return True
        if domain and hit_domain == domain and not paths:
            return True
        return False

    def rebuild(self) -> int:
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
                            metadata=row.extra or {},
                        )
                    )
        except Exception:
            logger.debug("RAG DB load skipped", exc_info=True)

        seed_index = 0
        for name in ("corpus_seed.json", "corpus_seed_extra.json"):
            path = SEEDS_DIR / name
            if not path.is_file():
                continue
            try:
                payload = load_json_file_bounded(path)
            except Exception:
                logger.debug("RAG seed load skipped for %s", path, exc_info=True)
                continue
            for doc in payload.get("documents", []):
                title = str(doc.get("title", "")).strip()
                body = str(doc.get("text", "")).strip()
                if not title or not body:
                    continue
                hits.append(
                    RagHit(
                        document_id=-(seed_index + 1),
                        content_hash=f"seed_{seed_index}",
                        title=title,
                        text=body,
                        source="seeds",
                        score=0.0,
                        metadata=doc.get("metadata", {}),
                    )
                )
                seed_index += 1

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

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 8,
        min_score: float = 0.05,
        domain: str | None = None,
        paths: list[str] | None = None,
    ) -> list[RagHit]:
        if self._matrix is None or not self._hits:
            return []
        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix)[0]
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        out: list[RagHit] = []
        allowed_paths = set(paths or [])
        for idx, score in ranked:
            if score < min_score:
                break
            base = self._hits[idx]
            if not self._hit_allowed(base, domain=domain, paths=allowed_paths):
                continue
            out.append(
                RagHit(
                    document_id=base.document_id,
                    content_hash=base.content_hash,
                    title=base.title,
                    text=base.text,
                    source=base.source,
                    score=float(score),
                    metadata=base.metadata or {},
                )
            )
            if len(out) >= top_k:
                break
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
    domain: str | None = None,
    paths: list[str] | None = None,
) -> tuple[str, list[RagHit], list[dict[str, Any]]]:
    """Return context text, hits, and citation dicts for chat/predict."""
    if top_k is None:
        top_k = int(os.environ.get("AUREON_RAG_TOP_K", "8"))
    if max_words is None:
        max_words = int(os.environ.get("AUREON_PREDICT_CONTEXT_WORDS", "1000000"))
        max_words = max(50, min(max_words, 1_000_000))

    hits = get_rag_index().retrieve(query, top_k=top_k, domain=domain, paths=paths)
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
