"""Collector region — sensory input: gather domain-specific text."""

from __future__ import annotations

import json
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from brain.base import AgentContext, AgentResult, MicroAgentBase
from db.models import Document
from pipeline.config import SEEDS_DIR
from pipeline.step1_collection.collectors import ArxivCollector, RawDocument
from pipeline.step1_collection.filters import filter_document

DOMAIN_QUERIES: dict[str, str] = {
    "mathematics": "cat:math.AG OR all:algebra OR all:calculus",
    "physics": "cat:physics.gen-ph OR all:quantum mechanics",
    "computer_science": "cat:cs.AI OR cat:cs.LG OR all:machine learning",
    "biology": "all:genetics OR all:molecular biology",
    "linguistics": "all:grammar OR all:phonology OR all:syntax",
    "vedic_sciences": "all:sanskrit OR all:jyotisha",
}


class CollectorAgent(MicroAgentBase):
    region = "collector"

    def run(self, session: Session, ctx: AgentContext) -> AgentResult:
        before = session.scalar(
            select(func.count()).select_from(Document).where(Document.domain_id == ctx.domain_id)
        ) or 0
        attempts = 0

        attempts += self._ingest_seeds(session, ctx)

        query = DOMAIN_QUERIES.get(ctx.domain_slug)
        if query:
            collector = ArxivCollector(query=query)
            for doc in collector.collect(limit=3):
                attempts += 1
                self._persist_doc(session, ctx, doc)

        after = session.scalar(
            select(func.count()).select_from(Document).where(Document.domain_id == ctx.domain_id)
        ) or 0

        return AgentResult(
            region=self.region,
            status="completed",
            metrics={"collection_attempts": attempts, "new_documents": after - before},
        )

    def _ingest_seeds(self, session: Session, ctx: AgentContext) -> int:
        count = 0
        if not SEEDS_DIR.exists():
            return 0
        for path in SEEDS_DIR.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for item in payload.get("documents", []):
                if item.get("metadata", {}).get("domain") != ctx.domain_slug:
                    continue
                doc = RawDocument(
                    doc_id=item.get("doc_id", str(uuid.uuid4())),
                    source="seeds",
                    title=item["title"],
                    text=item["text"],
                    url=item.get("url", ""),
                    metadata=item.get("metadata", {}),
                )
                count += 1
                self._persist_doc(session, ctx, doc)
        return count

    def _persist_doc(self, session: Session, ctx: AgentContext, doc: RawDocument) -> bool:
        ok, quality, _reason = filter_document(doc)
        if not ok:
            return False
        digest = doc.content_hash()
        if session.scalar(select(Document).where(Document.content_hash == digest)):
            return False
        session.add(
            Document(
                domain_id=ctx.domain_id,
                subdomain_id=ctx.subdomain_id,
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
        return True
