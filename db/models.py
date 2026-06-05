"""PostgreSQL schema — domains, documents, micro-agents, training artifacts."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class KnowledgeDomain(Base):
    """Top-level human knowledge domain (e.g. mathematics, biology)."""

    __tablename__ = "knowledge_domains"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    subdomains: Mapped[list[KnowledgeSubdomain]] = relationship(back_populates="domain")
    micro_agents: Mapped[list[MicroAgent]] = relationship(back_populates="domain")


class KnowledgeSubdomain(Base):
    """Sub-knowledge domain (e.g. algebra under mathematics)."""

    __tablename__ = "knowledge_subdomains"
    __table_args__ = (UniqueConstraint("domain_id", "slug", name="uq_subdomain_domain_slug"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    domain_id: Mapped[int] = mapped_column(ForeignKey("knowledge_domains.id"), index=True)
    slug: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(128))
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    domain: Mapped[KnowledgeDomain] = relationship(back_populates="subdomains")
    micro_agents: Mapped[list[MicroAgent]] = relationship(back_populates="subdomain")


class MicroAgent(Base):
    """
    A specialized micro-algorithm — one brain region for one domain/subdomain.
    Regions: collector, verifier, labeler, trainer, evaluator, reward
    """

    __tablename__ = "micro_agents"
    __table_args__ = (
        UniqueConstraint("region", "domain_id", "subdomain_id", name="uq_agent_region_scope"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    region: Mapped[str] = mapped_column(String(32), index=True)
    domain_id: Mapped[int] = mapped_column(ForeignKey("knowledge_domains.id"), index=True)
    subdomain_id: Mapped[int | None] = mapped_column(
        ForeignKey("knowledge_subdomains.id"), nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(String(32), default="idle")
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    domain: Mapped[KnowledgeDomain] = relationship(back_populates="micro_agents")
    subdomain: Mapped[KnowledgeSubdomain | None] = relationship(back_populates="micro_agents")
    runs: Mapped[list[AgentRun]] = relationship(back_populates="agent")


class AgentRun(Base):
    """Execution record for a single micro-agent cycle."""

    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agent_id: Mapped[int] = mapped_column(ForeignKey("micro_agents.id"), index=True)
    status: Mapped[str] = mapped_column(String(32), default="running")
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    agent: Mapped[MicroAgent] = relationship(back_populates="runs")


class Document(Base):
    """Collected text — raw/clean corpus stored in PostgreSQL."""

    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("content_hash", name="uq_document_hash"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    domain_id: Mapped[int | None] = mapped_column(ForeignKey("knowledge_domains.id"), index=True)
    subdomain_id: Mapped[int | None] = mapped_column(
        ForeignKey("knowledge_subdomains.id"), nullable=True, index=True
    )
    source: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(String(512))
    text: Mapped[str] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    language: Mapped[str] = mapped_column(String(16), default="en")
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    extra: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    labels: Mapped[list[DocumentLabel]] = relationship(back_populates="document")


class DocumentLabel(Base):
    __tablename__ = "document_labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    domain_id: Mapped[int] = mapped_column(ForeignKey("knowledge_domains.id"), index=True)
    subdomain_id: Mapped[int | None] = mapped_column(
        ForeignKey("knowledge_subdomains.id"), nullable=True
    )
    label: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    label_source: Mapped[str] = mapped_column(String(64), default="teacher")
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    document: Mapped[Document] = relationship(back_populates="labels")


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    domain_id: Mapped[int | None] = mapped_column(ForeignKey("knowledge_domains.id"), index=True)
    subdomain_id: Mapped[int | None] = mapped_column(
        ForeignKey("knowledge_subdomains.id"), nullable=True
    )
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    promoted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    domain_id: Mapped[int | None] = mapped_column(ForeignKey("knowledge_domains.id"), index=True)
    benchmark_type: Mapped[str] = mapped_column(String(64))
    score: Mapped[float] = mapped_column(Float)
    passed: Mapped[bool] = mapped_column(Boolean)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class PipelineEvent(Base):
    __tablename__ = "pipeline_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    step: Mapped[str] = mapped_column(String(64))
    event_type: Mapped[str] = mapped_column(String(64))
    domain_id: Mapped[int | None] = mapped_column(ForeignKey("knowledge_domains.id"), nullable=True)
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class PreferencePair(Base):
    """RLHF preference data — preferred vs rejected responses."""

    __tablename__ = "preference_pairs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    domain_id: Mapped[int | None] = mapped_column(ForeignKey("knowledge_domains.id"), nullable=True)
    context: Mapped[str] = mapped_column(Text)
    preferred: Mapped[str] = mapped_column(Text)
    rejected: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
