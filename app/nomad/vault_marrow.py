"""Vault marrow — secrets bound to organism fingerprint (nomad vault_marrow pattern)."""

from __future__ import annotations

import json
import os
from pathlib import Path


def secrets_file_path() -> Path | None:
    from app.railway_env import get_railway_bootstrap_report

    report = get_railway_bootstrap_report()
    path = report.get("secrets_file")
    if path:
        candidate = Path(str(path))
        if candidate.is_file():
            return candidate
    data_dir = os.environ.get("AUREON_DATA_DIR", "data").strip() or "data"
    candidate = Path(data_dir) / "railway-secrets.json"
    return candidate if candidate.is_file() else None


def vault_binding_enabled() -> bool:
    return os.environ.get("AUREON_VAULT_BIND_FINGERPRINT", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _env_secrets_configured() -> bool:
    return bool(os.environ.get("AUREON_API_KEY", "").strip())


def _ensure_secrets_file_from_env() -> Path | None:
    if not _env_secrets_configured():
        return None
    from app.railway_env import sync_env_secrets_to_file

    return sync_env_secrets_to_file()


def check_vault_marrow(organism_fingerprint: str) -> dict[str, str | bool]:
    path = secrets_file_path()
    if not path:
        if _env_secrets_configured():
            path = _ensure_secrets_file_from_env()
        if not path:
            if vault_binding_enabled() and not _env_secrets_configured():
                return {"ok": False, "detail": "Vault binding enabled but secrets file missing"}
            if _env_secrets_configured():
                return {"ok": True, "detail": "Secrets in Railway env (vault file pending sync)"}
            return {"ok": True, "detail": "No persisted secrets vault (optional)"}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {"ok": False, "detail": f"Secrets vault unreadable: {exc}"}

    if not isinstance(payload, dict):
        return {"ok": False, "detail": "Secrets vault invalid format"}

    if vault_binding_enabled():
        bound = str(payload.get("organism_fingerprint", "")).strip()
        if bound and bound != organism_fingerprint:
            # Redeploy or first bind — re-seal vault with current organism fingerprint.
            payload["organism_fingerprint"] = organism_fingerprint
            try:
                path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except OSError as exc:
                return {"ok": False, "detail": f"Could not re-seal vault: {exc}"}
            return {"ok": True, "detail": f"Vault marrow re-sealed to current fingerprint ({path.name})"}

    keys = [k for k in payload if k.startswith("AUREON_")]
    return {"ok": True, "detail": f"Vault marrow sealed ({len(keys)} secrets at {path.name})"}
