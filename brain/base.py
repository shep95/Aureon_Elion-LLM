"""Base micro-agent — each brain region runs independently but shares state via DB."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from db.models import AgentRun, MicroAgent, PipelineEvent


@dataclass
class AgentContext:
    domain_slug: str
    subdomain_slug: str | None
    domain_id: int
    subdomain_id: int | None
    epochs: int = 200
    extra: dict[str, Any] = field(default_factory=dict)


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
                    payload={"domain": ctx.domain_slug, "subdomain": ctx.subdomain_slug, **result.metrics},
                )
            )
            return result
        except Exception as exc:
            run.status = "failed"
            run.error = str(exc)
            run.finished_at = datetime.now(timezone.utc)
            agent.status = "error"
            return AgentResult(region=self.region, status="failed", error=str(exc))
