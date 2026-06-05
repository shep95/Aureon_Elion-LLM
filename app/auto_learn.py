"""Background automated learning for Railway — kickstart on boot, repeat on interval."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from brain.cortex import bootstrap_brain, run_graduation_ladder
from brain.domains.taxonomy import all_micro_triples
from brain.graduation import current_grade
from db.models import KnowledgeDomain, KnowledgeMicroSubdomain, KnowledgeSubdomain
from db.session import get_session
from fastapi import HTTPException

from app.organism import get_organism
from app.security import exclusive_training_lock
from sqlalchemy import select

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int = 10_000) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, min(maximum, int(raw)))
    except ValueError:
        return default


@dataclass
class AutoLearnConfig:
    enabled: bool = False
    on_startup: bool = True
    interval_sec: int = 3600
    epochs: int = 150
    max_grades_per_cycle: int = 1
    domain_limit: int = 29
    subdomain_limit: int = 1
    micro_limit: int = 1

    @classmethod
    def from_env(cls) -> AutoLearnConfig:
        from app.startup import is_railway

        enabled = _env_bool("AUREON_AUTO_LEARN", default=is_railway())
        return cls(
            enabled=enabled,
            on_startup=_env_bool("AUREON_AUTO_LEARN_ON_STARTUP", default=True),
            interval_sec=_env_int("AUREON_AUTO_LEARN_INTERVAL_SEC", 3600, minimum=300),
            epochs=_env_int("AUREON_AUTO_LEARN_EPOCHS", 150, minimum=50, maximum=500),
            max_grades_per_cycle=_env_int("AUREON_AUTO_LEARN_MAX_GRADES", 1, minimum=1, maximum=7),
            domain_limit=_env_int("AUREON_AUTO_LEARN_DOMAIN_LIMIT", 2, minimum=1, maximum=29),
            subdomain_limit=_env_int("AUREON_AUTO_LEARN_SUBDOMAIN_LIMIT", 1, minimum=1, maximum=20),
            micro_limit=_env_int("AUREON_AUTO_LEARN_MICRO_LIMIT", 1, minimum=1, maximum=10),
        )


@dataclass
class AutoLearnState:
    running: bool = False
    started_at: str | None = None
    last_run_at: str | None = None
    next_run_at: str | None = None
    cycles_completed: int = 0
    last_result: dict[str, Any] = field(default_factory=dict)
    last_error: str | None = None
    current_target: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "started_at": self.started_at,
            "last_run_at": self.last_run_at,
            "next_run_at": self.next_run_at,
            "cycles_completed": self.cycles_completed,
            "last_result": self.last_result,
            "last_error": self.last_error,
            "current_target": self.current_target,
        }


class AutoLearnScheduler:
    """Daemon thread that advances grade-level learning across the taxonomy."""

    def __init__(self, config: AutoLearnConfig | None = None) -> None:
        self.config = config or AutoLearnConfig.from_env()
        self.state = AutoLearnState()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cursor = 0
        self._triples: list[tuple[str, str, str]] = []

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "config": {
                "on_startup": self.config.on_startup,
                "interval_sec": self.config.interval_sec,
                "epochs": self.config.epochs,
                "max_grades_per_cycle": self.config.max_grades_per_cycle,
                "domain_limit": self.config.domain_limit,
                "subdomain_limit": self.config.subdomain_limit,
                "micro_limit": self.config.micro_limit,
            },
            **self.state.to_dict(),
        }

    def start(self) -> None:
        if not self.config.enabled:
            logger.info(
                "Auto-learn disabled (set AUREON_AUTO_LEARN=1 to enable, or deploy on Railway)."
            )
            return
        if self._thread and self._thread.is_alive():
            return
        self._triples = all_micro_triples()
        self.state.running = True
        self.state.started_at = datetime.now(timezone.utc).isoformat()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="aureon-auto-learn", daemon=True)
        self._thread.start()
        logger.info(
            "Auto-learn started — interval=%ss, epochs=%s, max_grades=%s",
            self.config.interval_sec,
            self.config.epochs,
            self.config.max_grades_per_cycle,
        )

    def stop(self) -> None:
        self._stop.set()
        self.state.running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        if self.config.on_startup:
            self._sleep_until_organism_vital()
            self._run_one_cycle()
        while not self._stop.wait(self.config.interval_sec):
            self._sleep_until_organism_vital()
            self._run_one_cycle()

    def _sleep_until_organism_vital(self) -> None:
        for _ in range(12):
            if self._stop.is_set():
                return
            try:
                organism = get_organism()
                organism.pulse()
                if organism.is_vital():
                    return
            except Exception:
                logger.exception("Organism pulse failed during auto-learn wait")
            time.sleep(5)

    def _next_target(self) -> tuple[str, str, str] | None:
        if not self._triples:
            self._triples = all_micro_triples()
        if not self._triples:
            return None
        domain, sub, micro = self._triples[self._cursor % len(self._triples)]
        self._cursor += 1
        return domain, sub, micro

    def _run_one_cycle(self) -> None:
        target = self._next_target()
        if not target:
            self.state.last_error = "no micro-subdomains in taxonomy"
            return

        domain_slug, subdomain_slug, micro_slug = target
        self.state.current_target = {
            "domain": domain_slug,
            "subdomain": subdomain_slug,
            "micro_subdomain": micro_slug,
        }

        try:
            organism = get_organism()
            organism.pulse()
            if not organism.is_vital():
                self.state.last_error = "organism lockdown — auto-learn paused"
                logger.warning("Auto-learn skipped: organism not vital")
                return

            bootstrap_brain()

            with get_session() as session:
                domain = session.scalar(
                    select(KnowledgeDomain).where(KnowledgeDomain.slug == domain_slug)
                )
                if not domain:
                    self.state.last_error = f"unknown domain: {domain_slug}"
                    return
                subdomain = session.scalar(
                    select(KnowledgeSubdomain).where(
                        KnowledgeSubdomain.domain_id == domain.id,
                        KnowledgeSubdomain.slug == subdomain_slug,
                    )
                )
                if not subdomain:
                    self.state.last_error = f"unknown subdomain: {subdomain_slug}"
                    return
                micro = session.scalar(
                    select(KnowledgeMicroSubdomain).where(
                        KnowledgeMicroSubdomain.subdomain_id == subdomain.id,
                        KnowledgeMicroSubdomain.slug == micro_slug,
                    )
                )
                if not micro:
                    self.state.last_error = f"unknown micro_subdomain: {micro_slug}"
                    return
                grade_row = current_grade(session, micro.id)
                grade_slug = grade_row.grade_slug if grade_row else "graduated"

            logger.info(
                "Auto-learn cycle #%s — %s.%s.%s @ grade %s",
                self.state.cycles_completed + 1,
                domain_slug,
                subdomain_slug,
                micro_slug,
                grade_slug,
            )

            with exclusive_training_lock():
                result = run_graduation_ladder(
                    domain_slug,
                    subdomain_slug,
                    micro_slug,
                    epochs=self.config.epochs,
                    max_grades=self.config.max_grades_per_cycle,
                )

            self.state.cycles_completed += 1
            self.state.last_run_at = datetime.now(timezone.utc).isoformat()
            self.state.next_run_at = (
                datetime.now(timezone.utc) + timedelta(seconds=self.config.interval_sec)
            ).isoformat()
            self.state.last_result = {
                "target": self.state.current_target,
                "grade_before": grade_slug,
                "graduation": result.get("ladder", [{}])[-1].get("graduation") if result.get("ladder") else {},
                "steps": result.get("steps_completed", 0),
            }
            self.state.last_error = None
            logger.info("Auto-learn cycle complete: %s", self.state.last_result)

        except HTTPException as exc:
            self.state.last_error = f"training busy: {exc.detail}"
            logger.warning("Auto-learn skipped — %s", exc.detail)
        except Exception as exc:
            self.state.last_error = str(exc)[:500]
            logger.exception("Auto-learn cycle failed")


_scheduler: AutoLearnScheduler | None = None


def get_auto_learn_scheduler() -> AutoLearnScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = AutoLearnScheduler()
    return _scheduler


def start_auto_learn() -> AutoLearnScheduler:
    scheduler = get_auto_learn_scheduler()
    scheduler.start()
    return scheduler


def stop_auto_learn() -> None:
    if _scheduler:
        _scheduler.stop()
