"""Base micro-agent — each brain region runs independently but shares state via DB."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from db.models import AgentRun, MicroAgent, PipelineEvent


from brain.grades import GradeLevel

@dataclass
class AgentContext:
    domain_slug: str
    subdomain_slug: str | None
    micro_subdomain_slug: str | None
    domain_id: int
    subdomain_id: int | None
    micro_subdomain_id: int | None
    epochs: int = 200
    grade_slug: str | None = None
    grade: GradeLevel | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def scope_slug(self) -> str:
        if self.micro_subdomain_slug:
            return self.micro_subdomain_slug
        if self.subdomain_slug:
            return self.subdomain_slug
        return self.domain_slug


@dataclass
class AgentResult:
    region: str
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class MicroAgentBase(ABC):
    region: str = "base"

    @abstractmethod
    def run(self, session: Session, ctx: AgentContext) -> AgentResult: ...

    def execute(self, session: Session, agent: MicroAgent, ctx: AgentContext) -> AgentResult:
        from app.activity_log import log_region_complete, log_region_start

        source = str(ctx.extra.get("source", "internal"))
        log_region_start(
            self.region,
            domain=ctx.domain_slug,
            subdomain=ctx.subdomain_slug,
            micro_subdomain=ctx.micro_subdomain_slug,
            grade=ctx.grade_slug,
            agent_id=agent.id,
            source=source,
        )

        run = AgentRun(agent_id=agent.id, status="running")
        session.add(run)
        agent.status = "running"
        session.flush()

        try:
            result = self.run(session, ctx)
            run.status = result.status
            run.metrics = result.metrics
            run.error = result.error
            run.finished_at = datetime.now(timezone.utc)
            agent.status = "idle"
            agent.last_run_at = run.finished_at
            session.add(
                PipelineEvent(
                    step=f"brain_{self.region}",
                    event_type=result.status,
                    domain_id=ctx.domain_id,
                    payload={
                        "domain": ctx.domain_slug,
                        "subdomain": ctx.subdomain_slug,
                        "micro_subdomain": ctx.micro_subdomain_slug,
                        "grade": ctx.grade_slug,
                        **result.metrics,
                    },
                )
            )
            log_region_complete(
                self.region,
                domain=ctx.domain_slug,
                subdomain=ctx.subdomain_slug,
                micro_subdomain=ctx.micro_subdomain_slug,
                grade=ctx.grade_slug,
                status=result.status,
                metrics=result.metrics,
                error=result.error,
                source=source,
            )
            return result
        except Exception as exc:
            run.status = "failed"
            run.error = str(exc)
            run.finished_at = datetime.now(timezone.utc)
            agent.status = "error"
            log_region_complete(
                self.region,
                domain=ctx.domain_slug,
                subdomain=ctx.subdomain_slug,
                micro_subdomain=ctx.micro_subdomain_slug,
                grade=ctx.grade_slug,
                status="failed",
                error=str(exc),
                source=source,
            )
            return AgentResult(region=self.region, status="failed", error=str(exc))
