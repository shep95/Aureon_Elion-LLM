"""Organism vitals and lockdown guard (nomad sovereign_organism pattern, Aureon-adapted)."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text

from app.audit import get_audit_log
from app.nomad.organ_registry import (
    NOMAD_DOCTRINE,
    ORGAN_DEPENDENCIES,
    ORGAN_META,
    OrganId,
    OrganState,
    dependencies_satisfied,
    organ_activation_order,
)
from app.nomad.supply_spleen import verify_supply_chain
from app.nomad.vault_marrow import check_vault_marrow
from app.rate_limit import get_rate_limiter
from app.replay_guard import get_replay_guard, replay_guard_enabled
from app.security import api_key_required

logger = logging.getLogger(__name__)


def _is_production() -> bool:
    return os.environ.get("RAILWAY_ENVIRONMENT", "").strip() != "" or os.environ.get(
        "AUREON_ENV", ""
    ).strip().lower() in ("production", "prod")


class AureonOrganism:
    """
    Sovereign security organism — fourteen interlocking organs (nomad_cyber_algorithm adapted).

    Partial compromise triggers lockdown on mutating HTTP operations.
    Background auto-learn uses a narrower ``is_learning_allowed()`` gate.
    """

    DOCTRINE = NOMAD_DOCTRINE

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pulse_generation = 0
        self._lockdown_reason: str | None = None
        self._organ_states: dict[OrganId, OrganState] = {}
        self._organ_details: dict[OrganId, str] = {}
        self._last_pulse: dict[OrganId, str] = {}

    def pulse(self) -> None:
        with self._lock:
            self._pulse_generation += 1
            now = datetime.now(timezone.utc).isoformat()
            statuses: dict[OrganId, OrganState] = {}
            details: dict[OrganId, str] = {}

            for organ_id in organ_activation_order():
                if not dependencies_satisfied(organ_id, statuses):
                    statuses[organ_id] = "critical"
                    details[organ_id] = "Dependency organs not vital"
                    continue
                result = self._check_organ(organ_id)
                statuses[organ_id] = result["state"]
                details[organ_id] = result.get("detail", "")

            self._organ_states = statuses
            self._organ_details = details
            self._last_pulse = {oid: now for oid in statuses}

            intrinsic_critical = [
                oid
                for oid, state in statuses.items()
                if state == "critical" and not details[oid].startswith("Dependency organs")
            ]
            non_auth_critical = [oid for oid in intrinsic_critical if oid != "auth_gateway"]
            if non_auth_critical:
                self._lockdown_reason = f"critical organs: {', '.join(non_auth_critical)}"
            else:
                self._lockdown_reason = None

        get_audit_log().record(
            "organism_pulse",
            detail=f"pulse gen={self._pulse_generation} vital={self.is_vital()}",
        )

    def _check_organ(self, organ_id: OrganId) -> dict[str, Any]:
        if organ_id == "crypto_core":
            from app.nomad.crypto_core import verify_crypto_core

            result = verify_crypto_core()
            return {
                "state": "vital" if result["ok"] else "critical",
                "detail": str(result.get("detail", "")),
            }

        if organ_id == "supply_spleen":
            result = verify_supply_chain()
            return {
                "state": "vital" if result["ok"] else "critical",
                "detail": str(result.get("detail", "")),
            }

        if organ_id == "audit_immune":
            chain = get_audit_log().verify_chain()
            if chain["valid"]:
                return {"state": "vital", "detail": f"chain intact ({chain['length']} entries)"}
            if not os.environ.get("AUREON_AUDIT_CHAIN_KEY", "").strip():
                return {
                    "state": "dormant",
                    "detail": "Ephemeral audit key — set AUREON_AUDIT_CHAIN_KEY for durable chain",
                }
            return {"state": "critical", "detail": "; ".join(chain["errors"][:3])}

        if organ_id == "bootstrap_heart":
            from app.startup import is_railway

            if not is_railway():
                return {"state": "dormant", "detail": "Local runtime — bootstrap optional"}
            from app.railway_env import get_railway_bootstrap_report

            report = get_railway_bootstrap_report()
            if not report.get("railway"):
                return {"state": "dormant", "detail": "Bootstrap pending"}
            if report.get("api_key") in ("env", "generated", "restored"):
                return {
                    "state": "vital",
                    "detail": f"Railway bootstrap ok (database={report.get('database')})",
                }
            return {"state": "critical", "detail": "Railway bootstrap incomplete"}

        if organ_id == "auth_gateway":
            if api_key_required():
                return {"state": "vital", "detail": "AUREON_API_KEY configured"}
            if _is_production():
                return {"state": "critical", "detail": "AUREON_API_KEY required in production"}
            return {"state": "dormant", "detail": "Dev mode — mutating endpoints unauthenticated"}

        if organ_id == "replay_guard":
            if not api_key_required():
                return {"state": "dormant", "detail": "Replay guard inactive without API key auth"}
            if replay_guard_enabled():
                snap = get_replay_guard()
                return {
                    "state": "vital",
                    "detail": f"Replay guard armed (nonce cache ≤{snap._options.max_entries})",  # noqa: SLF001
                }
            if _is_production():
                return {"state": "critical", "detail": "Replay guard disabled in production"}
            return {"state": "dormant", "detail": "Replay guard disabled (dev)"}

        if organ_id == "rate_limit_nerves":
            snap = get_rate_limiter().snapshot()
            return {
                "state": "vital",
                "detail": f"limit {snap['max_mutating_per_minute']}/min, active_ips={snap['active_ips']}",
            }

        if organ_id == "database_marrow":
            try:
                from db.session import get_engine

                with get_engine().connect() as conn:
                    conn.execute(text("SELECT 1"))
                url = os.environ.get("DATABASE_URL", "")
                if url.startswith("postgresql"):
                    detail = "PostgreSQL reachable"
                elif url.startswith("sqlite"):
                    detail = "SQLite reachable"
                else:
                    detail = "Database reachable"
                return {"state": "vital", "detail": detail}
            except Exception as exc:
                return {"state": "critical", "detail": str(exc)[:200]}

        if organ_id == "training_lock":
            from app.security import training_lock_available

            if training_lock_available():
                return {"state": "vital", "detail": "Exclusive training lock ready"}
            return {"state": "dormant", "detail": "Training in progress — lock held"}

        if organ_id == "gateway_skin":
            return {"state": "vital", "detail": "SecurityGatewayMiddleware + headers armed"}

        if organ_id == "vault_marrow":
            fp = self._compute_fingerprint_unlocked()
            result = check_vault_marrow(fp)
            return {
                "state": "vital" if result["ok"] else "critical",
                "detail": str(result.get("detail", "")),
            }

        if organ_id == "activity_lungs":
            from app.activity_log import is_activity_logging_enabled

            if is_activity_logging_enabled():
                return {"state": "vital", "detail": "AI activity logging active"}
            return {"state": "dormant", "detail": "Activity logging disabled"}

        if organ_id == "occult_veil":
            from app.nomad.occult_veil import occult_status, occult_veil_enabled

            if not occult_veil_enabled():
                return {"state": "dormant", "detail": "Occult veil disabled"}
            status = occult_status()
            return {
                "state": "vital",
                "detail": f"Planetary epoch {status['planetary_epoch']} active",
            }

        if organ_id == "chaos_entropy":
            from app.nomad.chaos_entropy import chaos_master_key, chaos_status

            status = chaos_status()
            if not status["enabled"]:
                return {"state": "dormant", "detail": "Chaos entropy disabled"}
            if not chaos_master_key():
                return {
                    "state": "dormant",
                    "detail": "Chaos entropy active — no master key (fingerprints optional)",
                }
            return {"state": "vital", "detail": "Chaos padding + fingerprint engine armed"}

        return {"state": "critical", "detail": "Unknown organ"}

    def is_vital(self) -> bool:
        if self._lockdown_reason:
            return False
        if not self._organ_states:
            return True
        return all(state in ("vital", "dormant") for state in self._organ_states.values())

    def is_learning_allowed(self) -> bool:
        if self._lockdown_reason:
            return False
        if not self._organ_states:
            return True
        skip = {
            "auth_gateway",
            "replay_guard",
            "gateway_skin",
            "activity_lungs",
            "training_lock",
            "occult_veil",
            "chaos_entropy",
        }
        for organ_id, state in self._organ_states.items():
            if organ_id in skip:
                continue
            if state == "critical":
                detail = self._organ_details.get(organ_id, "")
                if detail.startswith("Dependency organs"):
                    continue
                return False
        return True

    def require_vital(self, operation: str) -> None:
        if not self.is_vital():
            reason = self._lockdown_reason or "organ not vital"
            raise OrganismLockdownError(
                f"ORGANISM_LOCKDOWN: {operation} blocked — {reason}. {self.DOCTRINE}"
            )

    def enter_lockdown(self, reason: str) -> None:
        with self._lock:
            self._lockdown_reason = reason
        logger.error("Organism lockdown: %s", reason)
        get_audit_log().record("organism_lockdown", detail=reason)

    def _compute_fingerprint_unlocked(self) -> str:
        audit_head = get_audit_log().head_id()
        pulse = str(self._pulse_generation)
        rate = str(get_rate_limiter().snapshot().get("requests_last_window", 0))
        supply = verify_supply_chain().get("hash", "no-supply") or "no-supply"
        payload = f"{audit_head}|{pulse}|{rate}|{supply}|{self._lockdown_reason or 'ok'}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_fingerprint(self) -> str:
        with self._lock:
            return self._compute_fingerprint_unlocked()

    def get_vitals_report(self) -> dict[str, Any]:
        if not self._organ_states:
            self.pulse()
        return {
            "vital": self.is_vital(),
            "learning_allowed": self.is_learning_allowed(),
            "pulse_generation": self._pulse_generation,
            "organism_fingerprint": self.get_fingerprint(),
            "lockdown_reason": self._lockdown_reason,
            "stack": "nomad_cyber_algorithm-adapted",
            "organ_count": len(self._organ_states),
            "organs": [
                {
                    "id": organ_id,
                    "name": ORGAN_META[organ_id]["name"],
                    "state": self._organ_states.get(organ_id, "pending"),
                    "role": ORGAN_META[organ_id]["role"],
                    "depends_on": ORGAN_DEPENDENCIES[organ_id],
                    "last_pulse": self._last_pulse.get(organ_id),
                    "detail": self._organ_details.get(organ_id),
                }
                for organ_id in organ_activation_order()
            ],
            "doctrine": self.DOCTRINE,
        }


class OrganismLockdownError(Exception):
    pass


_organism: AureonOrganism | None = None


def get_organism() -> AureonOrganism:
    global _organism
    if _organism is None:
        _organism = AureonOrganism()
    return _organism
