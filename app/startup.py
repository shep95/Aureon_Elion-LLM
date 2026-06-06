"""Deferred startup — keep Railway health checks fast, bootstrap in background."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

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
    """Log optional improvements only — required vars are auto-provisioned on Railway."""
    from app.railway_env import get_railway_bootstrap_report

    report = get_railway_bootstrap_report()
    if not report.get("railway"):
        return

    if report.get("database") == "sqlite":
        logger.info(
            "Optional: attach Railway Postgres and set DATABASE_URL=${{Postgres.DATABASE_URL}} "
            "for multi-replica persistence (currently using SQLite at %s).",
            report.get("data_dir", "data"),
        )
    if report.get("api_key") in ("generated", "restored"):
        logger.info(
            "Optional: copy AUREON_API_KEY from %s into Railway service variables "
            "if you do not mount a persistent volume on the data directory.",
            report.get("secrets_file"),
        )


def _bind_vault_fingerprint() -> None:
    """Stamp stable vault binding fingerprint into secrets file."""
    if os.environ.get("AUREON_VAULT_BIND_FINGERPRINT", "1").strip().lower() in ("0", "false", "no"):
        return
    try:
        from app.nomad.vault_marrow import seal_vault_fingerprint

        seal_vault_fingerprint()
    except Exception:
        logger.debug("Vault fingerprint bind skipped", exc_info=True)


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
    from app.activity_log import log_ai_activity

    log_ai_activity("startup_begin", railway=is_railway())
    logger.info("Aureon deferred startup beginning (Railway=%s)", is_railway())
    try:
        from brain.cortex import bootstrap_brain

        stats = bootstrap_brain()
        with _lock:
            _state.bootstrap_done = True
            _state.details["bootstrap"] = stats
        log_ai_activity("startup_bootstrap_complete", stats=stats)
        logger.info("Brain bootstrap complete: %s", stats)

        from brain.meta_consciousness import run_meta_inquiry_on_startup

        meta_boot = run_meta_inquiry_on_startup()
        if meta_boot:
            ex = meta_boot[0]
            logger.info("Startup self-inquiry — Q: %s A: %s", ex["question"], ex["answer"])
            with _lock:
                _state.details["meta_consciousness"] = meta_boot

        _bind_vault_fingerprint()

        from app.organism import get_organism

        get_organism().pulse()
        vitals = get_organism().get_vitals_report()
        logger.info(
            "Organism pulse complete — vital=%s learning_allowed=%s",
            vitals.get("vital"),
            vitals.get("learning_allowed"),
        )
        if not vitals.get("learning_allowed"):
            logger.warning(
                "Auto-learn may be blocked — critical organs: %s",
                [
                    o["id"]
                    for o in vitals.get("organs", [])
                    if isinstance(o, dict) and o.get("state") == "critical"
                ],
            )
        with _lock:
            _state.details["organism"] = {
                "vital": vitals.get("vital"),
                "learning_allowed": vitals.get("learning_allowed"),
                "organs": {
                    o["id"]: o.get("state")
                    for o in vitals.get("organs", [])
                    if isinstance(o, dict)
                },
            }

        with _lock:
            from app.railway_env import get_railway_bootstrap_report

            _state.details["railway_bootstrap"] = get_railway_bootstrap_report()

        _warn_production_config()

        from app.auto_learn import start_auto_learn

        scheduler = start_auto_learn()
        with _lock:
            _state.auto_learn_started = scheduler.config.enabled
            _state.details["auto_learn"] = scheduler.status()

        if scheduler.config.enabled:
            log_ai_activity(
                "startup_auto_learn_active",
                interval_sec=scheduler.config.interval_sec,
                status=scheduler.status(),
            )
            logger.info(
                "Auto-learn ACTIVE — first cycle starts in background, %s",
                "continuous 24/7"
                if scheduler.config.continuous
                else f"interval={scheduler.config.interval_sec}s",
            )
        else:
            logger.info(
                "Auto-learn OFF — set AUREON_AUTO_LEARN=1 on Railway to enable automated learning."
            )

        from app.learning_github_sync import start_github_sync_scheduler

        start_github_sync_scheduler()

        from brain.predict_engine import warm_up_predict_brain

        warm_stats = warm_up_predict_brain()
        with _lock:
            _state.details["predict_warmup"] = warm_stats
        logger.info("Predict brain warm-up: %s", warm_stats)

        _preload_olivetti_background()

        with _lock:
            _state.ready = True
            _state.error = None
        log_ai_activity("startup_ready", details=_state.details)
        logger.info("Aureon startup ready.")

        from app.nomad.organism_pulse import start_organism_pulse

        start_organism_pulse()
    except Exception as exc:
        log_ai_activity("startup_failed", error=str(exc)[:500])
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
