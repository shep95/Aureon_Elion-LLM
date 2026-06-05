"""Auto-provision Railway env — secrets, database URL, and data paths before startup."""

from __future__ import annotations

import json
import logging
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_bootstrap_report: dict[str, Any] = {}


def get_railway_bootstrap_report() -> dict[str, Any]:
    return dict(_bootstrap_report)


def _data_dir() -> Path:
    explicit = os.environ.get("AUREON_DATA_DIR", "").strip()
    if explicit:
        path = Path(explicit)
    elif Path("/data").is_dir():
        path = Path("/data")
    else:
        path = Path("data")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _secrets_path(data_dir: Path) -> Path:
    return data_dir / "railway-secrets.json"


def _load_persisted_secrets(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read persisted Railway secrets at %s", path)
        return {}
    if not isinstance(payload, dict):
        return {}
    return {k: str(v) for k, v in payload.items() if k.startswith("AUREON_") and v}


def _persist_secrets(path: Path, values: dict[str, str]) -> None:
    existing = _load_persisted_secrets(path)
    merged = {**existing, **values, "updated_at": datetime.now(timezone.utc).isoformat()}
    path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _resolve_postgres_url() -> str | None:
    for name in ("DATABASE_URL", "DATABASE_PRIVATE_URL", "POSTGRES_URL"):
        url = os.environ.get(name, "").strip()
        if url.startswith(("postgres://", "postgresql://")):
            return url
    host = os.environ.get("PGHOST", "").strip()
    if not host:
        return None
    user = os.environ.get("PGUSER", "postgres").strip()
    password = os.environ.get("PGPASSWORD", "").strip()
    port = os.environ.get("PGPORT", "5432").strip()
    database = os.environ.get("PGDATABASE", "railway").strip()
    if not password:
        return None
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def sync_env_secrets_to_file(data_dir: Path | None = None) -> Path | None:
    """Mirror Railway Variables into the volume secrets file for vault_marrow binding."""
    api_key = os.environ.get("AUREON_API_KEY", "").strip()
    audit_key = os.environ.get("AUREON_AUDIT_CHAIN_KEY", "").strip()
    if not api_key and not audit_key:
        return None

    directory = data_dir or _data_dir()
    secrets_file = _secrets_path(directory)
    values: dict[str, str] = {}
    if api_key:
        values["AUREON_API_KEY"] = api_key
    if audit_key:
        values["AUREON_AUDIT_CHAIN_KEY"] = audit_key
    _persist_secrets(secrets_file, values)
    return secrets_file


def _ensure_api_key(data_dir: Path, report: dict[str, Any]) -> None:
    if os.environ.get("AUREON_API_KEY", "").strip():
        report["api_key"] = "env"
        sync_env_secrets_to_file(data_dir)
        return

    secrets_file = _secrets_path(data_dir)
    persisted = _load_persisted_secrets(secrets_file)
    if persisted.get("AUREON_API_KEY"):
        os.environ["AUREON_API_KEY"] = persisted["AUREON_API_KEY"]
        report["api_key"] = "restored"
        return

    generated = secrets.token_urlsafe(48)
    os.environ["AUREON_API_KEY"] = generated
    _persist_secrets(secrets_file, {"AUREON_API_KEY": generated})
    report["api_key"] = "generated"
    logger.info(
        "Auto-provisioned AUREON_API_KEY (persisted to %s). "
        "Copy to Railway variables to survive redeploys without a volume.",
        secrets_file,
    )


def _ensure_audit_chain_key(data_dir: Path, report: dict[str, Any]) -> None:
    if os.environ.get("AUREON_AUDIT_CHAIN_KEY", "").strip():
        report["audit_chain_key"] = "env"
        sync_env_secrets_to_file(data_dir)
        return

    secrets_file = _secrets_path(data_dir)
    persisted = _load_persisted_secrets(secrets_file)
    if persisted.get("AUREON_AUDIT_CHAIN_KEY"):
        os.environ["AUREON_AUDIT_CHAIN_KEY"] = persisted["AUREON_AUDIT_CHAIN_KEY"]
        report["audit_chain_key"] = "restored"
        return

    generated = secrets.token_hex(32)
    os.environ["AUREON_AUDIT_CHAIN_KEY"] = generated
    _persist_secrets(secrets_file, {"AUREON_AUDIT_CHAIN_KEY": generated})
    report["audit_chain_key"] = "generated"
    logger.info(
        "Auto-provisioned AUREON_AUDIT_CHAIN_KEY (persisted to %s).",
        secrets_file,
    )


def _ensure_database_url(data_dir: Path, report: dict[str, Any]) -> None:
    existing = os.environ.get("DATABASE_URL", "").strip()
    if existing.startswith(("postgres://", "postgresql://", "sqlite:")):
        report["database"] = "postgresql" if existing.startswith("postgres") else "sqlite"
        report["database_source"] = "env"
        return

    postgres = _resolve_postgres_url()
    if postgres:
        os.environ["DATABASE_URL"] = postgres
        report["database"] = "postgresql"
        report["database_source"] = "railway_postgres"
        logger.info("Linked PostgreSQL detected — using Railway DATABASE_URL.")
        return

    db_path = data_dir / "aureon.db"
    sqlite_url = f"sqlite:///{db_path.as_posix()}"
    os.environ["DATABASE_URL"] = sqlite_url
    report["database"] = "sqlite"
    report["database_source"] = "auto_sqlite"
    logger.info(
        "No PostgreSQL linked — using SQLite at %s. "
        "Attach Railway Postgres and set DATABASE_URL=${{Postgres.DATABASE_URL}} for shared persistence.",
        db_path,
    )


def bootstrap_railway_environment() -> dict[str, Any]:
    """
    On Railway, provision missing production env before DB/audit/organism init.

    Idempotent: respects explicit env vars; restores or generates secrets once.
    """
    global _bootstrap_report
    from app.startup import is_railway

    report: dict[str, Any] = {"railway": is_railway(), "actions": []}
    if not is_railway():
        _bootstrap_report = report
        return report

    data_dir = _data_dir()
    os.environ.setdefault("AUREON_DATA_DIR", str(data_dir))
    audit_dir = data_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AUREON_AUDIT_LOG_DIR", str(audit_dir))

    _ensure_api_key(data_dir, report)
    _ensure_audit_chain_key(data_dir, report)
    _ensure_database_url(data_dir, report)

    report["data_dir"] = str(data_dir)
    report["secrets_file"] = str(_secrets_path(data_dir))
    _bootstrap_report = report

    logger.info(
        "Railway bootstrap complete — api_key=%s audit_key=%s database=%s (%s)",
        report.get("api_key"),
        report.get("audit_chain_key"),
        report.get("database"),
        report.get("database_source"),
    )
    return report
