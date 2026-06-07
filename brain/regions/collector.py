"""Collector region — sensory input: gather domain-specific text."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from brain.base import AgentContext, AgentResult, MicroAgentBase
from brain.domains.generate_micros import topics_for
from brain.domains.taxonomy import lookup_names
from db.models import Document
from app.security import load_json_file_bounded
from pipeline.config import SEEDS_DIR
from brain.multimodal_collector import MultimodalCollector
from brain.regions.code_collector import CODE_MICRO_SLUGS, CodeCollector
from pipeline.step1_collection.collectors import (
    ArxivCollector,
    GutenbergCollector,
    LocalFileCollector,
    RawDocument,
)
from pipeline.step1_collection.filters import filter_document

DOMAIN_QUERIES: dict[str, str] = {
    "mathematics": "cat:math.AG OR all:algebra OR all:calculus",
    "physics": "cat:physics.gen-ph OR all:quantum mechanics",
    "computer_science": "cat:cs.AI OR cat:cs.LG OR all:machine learning",
    "biology": "all:genetics OR all:molecular biology",
    "linguistics": "all:grammar OR all:phonology OR all:syntax",
    "vedic_sciences": "all:sanskrit OR all:jyotisha",
    "science_and_natural_philosophy": "cat:physics OR cat:math OR all:chemistry OR all:biology",
}

SUBDOMAIN_ARXIV_QUERIES: dict[str, str] = {
    "chemistry": "cat:physics.chem-ph OR all:chemistry",
    "physics": "cat:physics.gen-ph OR all:physics",
    "biology": "all:biology OR all:genetics OR all:ecology",
    "mathematics": "cat:math.* OR all:mathematics",
    "earth_and_environmental_sciences": "all:geology OR all:climate OR all:ecology",
    "astronomy_and_cosmology": "cat:astro-ph OR all:astronomy",
}


def _arxiv_query(ctx: AgentContext) -> str | None:
    if ctx.domain_slug in DOMAIN_QUERIES:
        return DOMAIN_QUERIES[ctx.domain_slug]
    if ctx.subdomain_slug and ctx.subdomain_slug in SUBDOMAIN_ARXIV_QUERIES:
        return SUBDOMAIN_ARXIV_QUERIES[ctx.subdomain_slug]
    if ctx.micro_subdomain_slug:
        term = ctx.micro_subdomain_slug.replace("_", " ")
        return f'all:"{term}"'
    return None


def _topic_seed_text(topic: str, names: dict[str, str]) -> str:
    domain = names.get("domain", "knowledge")
    subdomain = names.get("subdomain", "field")
    micro = names.get("micro_subdomain", "specialty")
    return (
        f"{topic} is a core concept within {micro} ({subdomain}) in the broader domain of {domain}. "
        f"Understanding {topic} requires studying definitions, historical development, key principles, "
        f"methods of analysis, and relationships to neighboring fields. Researchers and practitioners "
        f"apply {topic} to explain phenomena, design experiments, solve practical problems, and extend "
        f"theoretical frameworks. Mastery includes vocabulary, foundational models, common techniques, "
        f"and awareness of open questions and contemporary applications across academic and professional settings."
    )


def _seed_matches_context(metadata: dict[str, Any], ctx: AgentContext) -> bool:
    domain = str(metadata.get("domain") or "").strip()
    if domain and domain not in {ctx.domain_slug, ctx.subdomain_slug, ctx.micro_subdomain_slug}:
        return False

    subdomain = str(metadata.get("subdomain") or "").strip()
    if subdomain and subdomain != ctx.subdomain_slug:
        return False

    micro = str(metadata.get("micro_subdomain") or "").strip()
    if micro and micro != ctx.micro_subdomain_slug:
        return False

    return True


class CollectorAgent(MicroAgentBase):
    region = "collector"

    def run(self, session: Session, ctx: AgentContext) -> AgentResult:
        before = session.scalar(
            select(func.count()).select_from(Document).where(Document.domain_id == ctx.domain_id)
        ) or 0
        attempts = 0

        attempts += self._ingest_taxonomy_topics(session, ctx)
        attempts += self._ingest_seeds(session, ctx)
        attempts += self._ingest_code_corpus(session, ctx)
        attempts += self._ingest_external_sources(session, ctx)

        query = _arxiv_query(ctx)
        if query:
            limit = 3
            if ctx.grade:
                limit = ctx.grade.collection_limit
            collector = ArxivCollector(query=query)
            for doc in collector.collect(limit=limit):
                attempts += 1
                if ctx.grade_slug:
                    doc.metadata["grade"] = ctx.grade_slug
                self._persist_doc(session, ctx, doc)

        after = session.scalar(
            select(func.count()).select_from(Document).where(Document.domain_id == ctx.domain_id)
        ) or 0

        return AgentResult(
            region=self.region,
            status="completed",
            metrics={"collection_attempts": attempts, "new_documents": after - before},
        )

    def _ingest_taxonomy_topics(self, session: Session, ctx: AgentContext) -> int:
        """Seed documents from Zophiel leaf topics for the current micro-subdomain."""
        if not ctx.micro_subdomain_slug:
            return 0

        topics = topics_for(ctx.domain_slug, ctx.subdomain_slug or "", ctx.micro_subdomain_slug)
        if not topics:
            return 0

        limit = 3
        if ctx.grade:
            limit = max(ctx.grade.collection_limit, 3)
        names = lookup_names(ctx.domain_slug, ctx.subdomain_slug, ctx.micro_subdomain_slug)

        count = 0
        for topic in topics[:limit]:
            slug = topic.lower().replace(" ", "_").replace("(", "").replace(")", "")
            doc = RawDocument(
                doc_id=f"taxonomy_{ctx.domain_slug}_{ctx.subdomain_slug}_{ctx.micro_subdomain_slug}_{slug}",
                source="taxonomy",
                title=topic,
                text=_topic_seed_text(topic, names),
                url=f"seed://taxonomy/{ctx.domain_slug}/{ctx.subdomain_slug}/{ctx.micro_subdomain_slug}/{slug}",
                metadata={
                    "domain": ctx.domain_slug,
                    "subdomain": ctx.subdomain_slug,
                    "micro_subdomain": ctx.micro_subdomain_slug,
                    "topic": topic,
                    "source_type": "taxonomy_seed",
                },
            )
            if ctx.grade_slug:
                doc.metadata["grade"] = ctx.grade_slug
            count += 1
            self._persist_doc(session, ctx, doc)
        return count

    def _ingest_code_corpus(self, session: Session, ctx: AgentContext) -> int:
        """Ingest HumanEval + MBPP when training a code-generation micro-topic."""
        if ctx.subdomain_slug != "computer_science":
            return 0
        if ctx.micro_subdomain_slug not in CODE_MICRO_SLUGS:
            return 0

        limit = 2000
        if ctx.grade:
            limit = min(2000, max(ctx.grade.collection_limit * 200, 50))

        count = 0
        for doc in CodeCollector().collect(limit=limit):
            doc.metadata.setdefault("domain", ctx.domain_slug)
            doc.metadata.setdefault("subdomain", ctx.subdomain_slug)
            doc.metadata.setdefault("micro_subdomain", ctx.micro_subdomain_slug)
            doc.metadata.setdefault("code_area", "code_generation")
            if ctx.grade_slug:
                doc.metadata["grade"] = ctx.grade_slug
            count += 1
            self._persist_doc(session, ctx, doc)
        return count

    def _ingest_seeds(self, session: Session, ctx: AgentContext) -> int:
        count = 0
        if not SEEDS_DIR.exists():
            return 0
        for path in SEEDS_DIR.glob("*.json"):
            if not path.is_file():
                continue
            payload = load_json_file_bounded(path)
            for item in payload.get("documents", []):
                metadata = item.get("metadata", {})
                if not _seed_matches_context(metadata, ctx):
                    continue
                doc = RawDocument(
                    doc_id=item.get("doc_id", str(uuid.uuid4())),
                    source="seeds",
                    title=item["title"],
                    text=item["text"],
                    url=item.get("url", ""),
                    metadata=metadata,
                )
                count += 1
                self._persist_doc(session, ctx, doc)
        return count

    def _ingest_external_sources(self, session: Session, ctx: AgentContext) -> int:
        """Pull Gutenberg excerpts and local inbox drops (deduped by content hash)."""
        count = 0
        gutenberg_limit = 1
        if ctx.grade:
            gutenberg_limit = min(ctx.grade.collection_limit, 2)

        for doc in GutenbergCollector().collect(limit=gutenberg_limit):
            count += 1
            if ctx.grade_slug:
                doc.metadata["grade"] = ctx.grade_slug
            doc.metadata.setdefault("domain", ctx.domain_slug)
            self._persist_doc(session, ctx, doc)

        inbox_limit = 10
        if ctx.grade:
            inbox_limit = min(ctx.grade.collection_limit * 2, 25)
        for doc in LocalFileCollector().collect(limit=inbox_limit):
            meta_domain = doc.metadata.get("domain")
            if meta_domain and meta_domain != ctx.domain_slug:
                continue
            count += 1
            if ctx.grade_slug:
                doc.metadata["grade"] = ctx.grade_slug
            doc.metadata.setdefault("domain", ctx.domain_slug)
            self._persist_doc(session, ctx, doc)

        mm_limit = 5
        if ctx.grade:
            mm_limit = min(ctx.grade.collection_limit, 10)
        for doc in MultimodalCollector().collect(limit=mm_limit):
            count += 1
            if ctx.grade_slug:
                doc.metadata["grade"] = ctx.grade_slug
            doc.metadata.setdefault("domain", ctx.domain_slug)
            doc.metadata["modality"] = doc.metadata.get("modality", "text")
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
                micro_subdomain_id=ctx.micro_subdomain_id,
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
