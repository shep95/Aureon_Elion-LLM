"""Background automated learning for Railway — kickstart on boot, repeat on interval."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from brain.cortex import run_batch_graduation_ladder
from brain.domains.taxonomy import KNOWLEDGE_TAXONOMY

from app.organism import get_organism
from app.security import exclusive_training_lock
from fastapi import HTTPException

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


def _env_limit(name: str, default: int | None, *, maximum: int) -> int | None:
    """Parse tier limit. 0 or 'all' = no cap (train entire tier)."""
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("0", "all", "*"):
        return None
    if not raw:
        return default
    try:
        val = int(raw)
        if val <= 0:
            return None
        return min(val, maximum)
    except ValueError:
        return default


@dataclass
class AutoLearnConfig:
    enabled: bool = False
    on_startup: bool = True
    interval_sec: int = 3600
    epochs: int = 150
    max_grades_per_cycle: int = 1
    train_all: bool = False
    domain_limit: int | None = 30
    subdomain_limit: int | None = 8
    micro_limit: int | None = 17

    @classmethod
    def from_env(cls) -> AutoLearnConfig:
        from app.startup import is_railway

        enabled = _env_bool("AUREON_AUTO_LEARN", default=is_railway())
        train_all = _env_bool("AUREON_AUTO_LEARN_ALL", default=False)
        if train_all:
            limits = {"domain_limit": None, "subdomain_limit": None, "micro_limit": None}
        else:
            limits = {
                "domain_limit": _env_limit("AUREON_AUTO_LEARN_DOMAIN_LIMIT", 30, maximum=30),
                "subdomain_limit": _env_limit("AUREON_AUTO_LEARN_SUBDOMAIN_LIMIT", 8, maximum=20),
                "micro_limit": _env_limit("AUREON_AUTO_LEARN_MICRO_LIMIT", 17, maximum=20),
            }
        return cls(
            enabled=enabled,
            on_startup=_env_bool("AUREON_AUTO_LEARN_ON_STARTUP", default=True),
            interval_sec=_env_int("AUREON_AUTO_LEARN_INTERVAL_SEC", 3600, minimum=300),
            epochs=_env_int("AUREON_AUTO_LEARN_EPOCHS", 150, minimum=50, maximum=500),
            max_grades_per_cycle=_env_int("AUREON_AUTO_LEARN_MAX_GRADES", 1, minimum=1, maximum=7),
            train_all=train_all,
            **limits,
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
        self._domain_cursor = 0

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "config": {
                "on_startup": self.config.on_startup,
                "interval_sec": self.config.interval_sec,
                "epochs": self.config.epochs,
                "max_grades_per_cycle": self.config.max_grades_per_cycle,
                "train_all": self.config.train_all,
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
        self.state.running = True
        self.state.started_at = datetime.now(timezone.utc).isoformat()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="aureon-auto-learn", daemon=True)
        self._thread.start()
        logger.info(
            "Auto-learn started — interval=%ss, epochs=%s, max_grades=%s, "
            "domains=%s subs=%s micros=%s train_all=%s",
            self.config.interval_sec,
            self.config.epochs,
            self.config.max_grades_per_cycle,
            self.config.domain_limit if self.config.domain_limit is not None else "all",
            self.config.subdomain_limit if self.config.subdomain_limit is not None else "all",
            self.config.micro_limit if self.config.micro_limit is not None else "all",
            self.config.train_all,
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
                if organism.is_learning_allowed():
                    return
            except Exception:
                logger.exception("Organism pulse failed during auto-learn wait")
            time.sleep(5)

    def _domain_batch(self) -> list[str] | None:
        """When domain_limit < full corpus, rotate through domains across cycles."""
        all_domains = list(KNOWLEDGE_TAXONOMY.keys())
        if self.config.domain_limit is None:
            return None
        if self.config.domain_limit >= len(all_domains):
            return None
        start = self._domain_cursor % len(all_domains)
        batch = [
            all_domains[(start + i) % len(all_domains)] for i in range(self.config.domain_limit)
        ]
        self._domain_cursor += self.config.domain_limit
        return batch

    def _run_one_cycle(self) -> None:
        from app.activity_log import clear_cycle_id, log_ai_activity, new_cycle_id

        domain_slugs = self._domain_batch()
        cycle_id = new_cycle_id("auto")
        limits_label = (
            f"domains={self.config.domain_limit or 'all'} "
            f"subs={self.config.subdomain_limit or 'all'} "
            f"micros={self.config.micro_limit or 'all'}"
        )
        self.state.current_target = {
            "mode": "batch",
            "limits": limits_label,
            "domain_slugs": domain_slugs,
        }

        try:
            organism = get_organism()
            organism.pulse()
            if not organism.is_learning_allowed():
                vitals = organism.get_vitals_report()
                self.state.last_error = "organism lockdown — auto-learn paused"
                logger.warning(
                    "Auto-learn skipped: learning not allowed (vital=%s, lockdown=%s)",
                    vitals.get("vital"),
                    vitals.get("lockdown_reason"),
                )
                log_ai_activity(
                    "auto_learn_skipped",
                    cycle_id=cycle_id,
                    reason="organism lockdown",
                    vitals={
                        o["id"]: o["state"] for o in vitals.get("organs", []) if isinstance(o, dict)
                    },
                )
                return

            log_ai_activity(
                "auto_learn_cycle_start",
                cycle_id=cycle_id,
                source="auto_learn",
                cycle_number=self.state.cycles_completed + 1,
                limits=limits_label,
                train_all=self.config.train_all,
                epochs=self.config.epochs,
                max_grades=self.config.max_grades_per_cycle,
                domain_slugs=domain_slugs,
            )
            logger.info(
                "Auto-learn cycle #%s — batch %s",
                self.state.cycles_completed + 1,
                limits_label,
            )

            with exclusive_training_lock():
                batch = run_batch_graduation_ladder(
                    epochs=self.config.epochs,
                    max_grades=self.config.max_grades_per_cycle,
                    domain_limit=self.config.domain_limit if domain_slugs is None else None,
                    subdomain_limit=self.config.subdomain_limit,
                    micro_subdomain_limit=self.config.micro_limit,
                    domain_slugs=domain_slugs,
                    source="auto_learn",
                )

            if batch["targets_total"] == 0:
                self.state.last_error = "no micro-subdomains matched batch limits"
                log_ai_activity("auto_learn_skipped", cycle_id=cycle_id, reason=self.state.last_error)
                return

            graduations = [r for r in batch["results"] if r.get("graduation", {}).get("passed")]
            last = batch["results"][-1] if batch["results"] else {}

            self.state.cycles_completed += 1
            self.state.last_run_at = datetime.now(timezone.utc).isoformat()
            self.state.next_run_at = (
                datetime.now(timezone.utc) + timedelta(seconds=self.config.interval_sec)
            ).isoformat()
            self.state.last_result = {
                "batch": True,
                "targets_total": batch["targets_total"],
                "targets_processed": batch["targets_processed"],
                "graduations_passed": len(graduations),
                "last_target": last.get("target"),
                "last_graduation": last.get("graduation"),
                "sample_paths": [r["path"] for r in batch["results"][:5]],
            }
            self.state.last_error = None
            log_ai_activity(
                "auto_learn_cycle_complete",
                cycle_id=cycle_id,
                source="auto_learn",
                cycle_number=self.state.cycles_completed,
                result=self.state.last_result,
            )
            logger.info("Auto-learn cycle complete: %s", self.state.last_result)

        except HTTPException as exc:
            self.state.last_error = f"training busy: {exc.detail}"
            logger.warning("Auto-learn skipped — %s", exc.detail)
            log_ai_activity(
                "auto_learn_skipped",
                cycle_id=cycle_id,
                reason=str(exc.detail),
            )
        except Exception as exc:
            self.state.last_error = str(exc)[:500]
            logger.exception("Auto-learn cycle failed")
            log_ai_activity(
                "auto_learn_failed",
                cycle_id=cycle_id,
                error=str(exc)[:500],
            )
        finally:
            clear_cycle_id()


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
