"""Structured AI activity logging — every brain action visible in Railway deploy logs."""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("aureon.ai")

_thread_cycle_id = threading.local()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def is_activity_logging_enabled() -> bool:
    """Default ON on Railway so deploy logs capture all AI work."""
    if _env_bool("AUREON_ACTIVITY_LOG", default=False):
        return True
    if os.environ.get("AUREON_ACTIVITY_LOG", "").strip().lower() in ("0", "false", "no", "off"):
        return False
    try:
        from app.startup import is_railway

        return is_railway()
    except ImportError:
        return False


def use_json_format() -> bool:
    if _env_bool("AUREON_LOG_JSON", default=False):
        return True
    if os.environ.get("AUREON_LOG_JSON", "").strip().lower() in ("0", "false", "no", "off"):
        return False
    try:
        from app.startup import is_railway

        return is_railway()
    except ImportError:
        return False


def configure_logging() -> None:
    """Configure root logging for Railway (JSON lines when deployed)."""
    level_name = os.environ.get("AUREON_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    if use_json_format():
        fmt = "%(levelname)s:%(name)s:%(message)s"
    else:
        fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    logging.basicConfig(level=level, format=fmt, force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "aureon.ai"):
        logging.getLogger(name).setLevel(level)


def new_cycle_id(prefix: str = "cyc") -> str:
    """Start a correlated log cycle (auto-learn, grade ladder, API run)."""
    cycle_id = f"{prefix}-{secrets.token_hex(6)}"
    _thread_cycle_id.value = cycle_id
    return cycle_id


def current_cycle_id() -> str | None:
    return getattr(_thread_cycle_id, "value", None)


def clear_cycle_id() -> None:
    if hasattr(_thread_cycle_id, "value"):
        del _thread_cycle_id.value


def _scope_fields(
    *,
    domain: str | None = None,
    subdomain: str | None = None,
    micro_subdomain: str | None = None,
    grade: str | None = None,
) -> dict[str, str]:
    out: dict[str, str] = {}
    if domain:
        out["domain"] = domain
    if subdomain:
        out["subdomain"] = subdomain
    if micro_subdomain:
        out["micro_subdomain"] = micro_subdomain
    if grade:
        out["grade"] = grade
    if domain and subdomain and micro_subdomain:
        out["path"] = f"{domain}.{subdomain}.{micro_subdomain}"
    return out


def log_ai_activity(action: str, **fields: Any) -> None:
    """Emit one structured AI activity line to stdout (Railway log drain)."""
    if not is_activity_logging_enabled():
        return

    payload: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "component": "aureon-ai",
        "action": action,
    }
    cycle_id = fields.pop("cycle_id", None) or current_cycle_id()
    if cycle_id:
        payload["cycle_id"] = cycle_id
    for key, value in fields.items():
        if value is not None:
            payload[key] = value

    if use_json_format():
        logger.info(json.dumps(payload, default=str, separators=(",", ":")))
    else:
        parts = [f"action={action}"]
        for key, value in payload.items():
            if key in ("ts", "component", "action"):
                continue
            if isinstance(value, dict):
                parts.append(f"{key}={json.dumps(value, default=str)}")
            else:
                parts.append(f"{key}={value}")
        logger.info(" | ".join(parts))

    _maybe_audit(action, payload)


def _maybe_audit(action: str, payload: dict[str, Any]) -> None:
    """Mirror high-signal training events to the tamper-evident audit chain."""
    audit_actions = {
        "grade_cycle_start": "training_started",
        "grade_cycle_complete": "training_completed",
        "auto_learn_cycle_start": "training_started",
        "auto_learn_cycle_complete": "training_completed",
    }
    audit_type = audit_actions.get(action)
    if not audit_type:
        return
    try:
        from app.audit import get_audit_log

        detail_parts = [
            payload.get("action", ""),
            payload.get("path") or payload.get("domain") or "",
            payload.get("grade") or "",
            payload.get("status") or "",
        ]
        get_audit_log().record(
            audit_type,  # type: ignore[arg-type]
            correlation_id=str(payload.get("cycle_id") or ""),
            detail=" | ".join(p for p in detail_parts if p),
        )
    except Exception:
        logger.debug("Audit mirror skipped for action=%s", action, exc_info=True)


def log_region_start(
    region: str,
    *,
    domain: str,
    subdomain: str | None,
    micro_subdomain: str | None,
    grade: str | None,
    agent_id: int | None = None,
    source: str = "internal",
) -> None:
    log_ai_activity(
        "region_start",
        region=region,
        source=source,
        agent_id=agent_id,
        **_scope_fields(
            domain=domain,
            subdomain=subdomain,
            micro_subdomain=micro_subdomain,
            grade=grade,
        ),
    )


def log_region_complete(
    region: str,
    *,
    domain: str,
    subdomain: str | None,
    micro_subdomain: str | None,
    grade: str | None,
    status: str,
    metrics: dict[str, Any] | None = None,
    error: str | None = None,
    source: str = "internal",
) -> None:
    log_ai_activity(
        "region_complete",
        region=region,
        status=status,
        source=source,
        metrics=metrics or {},
        error=error,
        **_scope_fields(
            domain=domain,
            subdomain=subdomain,
            micro_subdomain=micro_subdomain,
            grade=grade,
        ),
    )


def log_graduation(
    *,
    domain: str,
    subdomain: str,
    micro_subdomain: str,
    grade: str,
    passed: bool,
    result: dict[str, Any],
    source: str = "internal",
) -> None:
    log_ai_activity(
        "graduation_decision",
        passed=passed,
        source=source,
        result=result,
        **_scope_fields(
            domain=domain,
            subdomain=subdomain,
            micro_subdomain=micro_subdomain,
            grade=grade,
        ),
    )
