"""Deferred startup — keep Railway health checks fast, bootstrap in background."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def is_railway() -> bool:
    """Detect Railway runtime (multiple env markers + Postgres + PORT)."""
    markers = (
        "RAILWAY_ENVIRONMENT",
        "RAILWAY_ENVIRONMENT_NAME",
        "RAILWAY_SERVICE_ID",
        "RAILWAY_PROJECT_ID",
        "RAILWAY_REPLICA_ID",
        "RAILWAY_PUBLIC_DOMAIN",
    )
    if any(os.environ.get(key, "").strip() for key in markers):
        return True
    db = os.environ.get("DATABASE_URL", "")
    port = os.environ.get("PORT", "")
    return bool(port and db.startswith(("postgres://", "postgresql://")))


@dataclass
class StartupState:
    started: bool = False
    ready: bool = False
    bootstrap_done: bool = False
    auto_learn_started: bool = False
    error: str | None = None
    details: dict = field(default_factory=dict)


_state = StartupState()
_lock = threading.Lock()


def get_startup_state() -> StartupState:
    with _lock:
        return _state


def _warn_production_config() -> None:
    from app.security import api_key_required

    if is_railway() or os.environ.get("AUREON_ENV", "").lower() in ("production", "prod"):
        if not api_key_required():
            logger.warning(
                "AUREON_API_KEY is not set — mutating endpoints are unauthenticated. "
                "Set AUREON_API_KEY on Railway for production."
            )
        if not os.environ.get("AUREON_AUDIT_CHAIN_KEY", "").strip():
            logger.warning(
                "AUREON_AUDIT_CHAIN_KEY is not set — audit chain resets on each deploy. "
                "Generate 32 random bytes as hex and set on Railway."
            )
        db = os.environ.get("DATABASE_URL", "")
        if not db.startswith(("postgres://", "postgresql://")):
            logger.warning(
                "DATABASE_URL is not PostgreSQL — attach Railway Postgres for persistent brain state."
            )


def _preload_olivetti_background() -> None:
    if is_railway() and os.environ.get("AUREON_PRELOAD_OLIVETTI", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        logger.info("Skipping Olivetti preload on Railway (set AUREON_PRELOAD_OLIVETTI=1 to enable).")
        return

    def _job() -> None:
        try:
            from sklearn.datasets import fetch_olivetti_faces

            fetch_olivetti_faces()
            logger.info("Olivetti faces cached for demo endpoints.")
        except Exception:
            logger.exception("Olivetti preload failed (demos may download on first use)")

    threading.Thread(target=_job, name="olivetti-preload", daemon=True).start()


def _deferred_startup() -> None:
    global _state
    logger.info("Aureon deferred startup beginning (Railway=%s)", is_railway())
    try:
        from brain.cortex import bootstrap_brain

        stats = bootstrap_brain()
        with _lock:
            _state.bootstrap_done = True
            _state.details["bootstrap"] = stats
        logger.info("Brain bootstrap complete: %s", stats)

        from app.organism import get_organism

        get_organism().pulse()
        logger.info("Organism pulse complete — vital=%s", get_organism().is_vital())

        _warn_production_config()

        from app.auto_learn import start_auto_learn

        scheduler = start_auto_learn()
        with _lock:
            _state.auto_learn_started = scheduler.config.enabled
            _state.details["auto_learn"] = scheduler.status()

        if scheduler.config.enabled:
            logger.info(
                "Auto-learn ACTIVE — first cycle starts in background, interval=%ss",
                scheduler.config.interval_sec,
            )
        else:
            logger.info(
                "Auto-learn OFF — set AUREON_AUTO_LEARN=1 on Railway to enable automated learning."
            )

        _preload_olivetti_background()

        with _lock:
            _state.ready = True
            _state.error = None
        logger.info("Aureon startup ready.")
    except Exception as exc:
        logger.exception("Deferred startup failed")
        with _lock:
            _state.error = str(exc)[:500]


def start_deferred_startup() -> None:
    global _state
    with _lock:
        if _state.started:
            return
        _state.started = True
    thread = threading.Thread(target=_deferred_startup, name="aureon-startup", daemon=True)
    thread.start()
    logger.info("Deferred startup thread launched — /health is live while bootstrap runs.")
