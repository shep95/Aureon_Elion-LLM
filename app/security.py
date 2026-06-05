"""Security utilities — auth, validation, SSRF protection, safe errors."""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
MAX_SUBDOMAIN_LIMIT = 20
MAX_MICRO_SUBDOMAIN_LIMIT = 20
MAX_DOMAIN_LIMIT = 30
MAX_EPOCHS = 500
MAX_JSON_BYTES = 10 * 1024 * 1024
MAX_MODEL_LAYERS = 10
MAX_LAYER_SIZE = 8192
MAX_ARTIFACT_BYTES = 50 * 1024 * 1024

_training_lock = threading.Lock()


def api_key_required() -> bool:
    return bool(os.environ.get("AUREON_API_KEY", "").strip())


def verify_api_key(provided: str | None, *, correlation_id: str | None = None, peer: str | None = None) -> None:
    """Constant-time API key check for mutating endpoints."""
    expected = os.environ.get("AUREON_API_KEY", "").strip()
    if not expected:
        return
    if not provided or not hmac.compare_digest(provided.strip(), expected):
        from app.audit import get_audit_log

        get_audit_log().record(
            "auth_failed",
            correlation_id=correlation_id,
            peer=peer,
            detail="Invalid or missing API key",
        )
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    from app.nomad.client_allowlist import verify_client_allowlist

    try:
        verify_client_allowlist(provided.strip())
    except ValueError as exc:
        from app.audit import get_audit_log

        get_audit_log().record(
            "auth_failed",
            correlation_id=correlation_id,
            peer=peer,
            detail=str(exc),
        )
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def get_api_key_from_request(request: Request) -> str | None:
    return request.headers.get("X-API-Key") or request.headers.get("Authorization", "").removeprefix("Bearer ").strip() or None


def require_mutating_access(request: Request) -> None:
    from app.organism import get_organism

    organism = get_organism()
    organism.require_vital(f"{request.method} {request.url.path}")
    peer = None
    if request.client:
        peer = request.client.host
    verify_api_key(
        get_api_key_from_request(request),
        correlation_id=getattr(request.state, "correlation_id", None),
        peer=peer,
    )


def validate_slug(slug: str, *, label: str = "slug") -> str:
    if not slug or not SLUG_PATTERN.match(slug):
        raise HTTPException(status_code=400, detail=f"Invalid {label}")
    return slug


def clamp_epochs(epochs: int) -> int:
    return max(1, min(epochs, MAX_EPOCHS))


def clamp_domain_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    return max(1, min(limit, MAX_DOMAIN_LIMIT))


def clamp_subdomain_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    return max(1, min(limit, MAX_SUBDOMAIN_LIMIT))


def clamp_micro_subdomain_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    return max(1, min(limit, MAX_MICRO_SUBDOMAIN_LIMIT))


def safe_error_message(exc: Exception, *, log: bool = True) -> str:
    """Never leak stack traces or paths to clients."""
    if log:
        logger.exception("Request failed: %s", exc.__class__.__name__)
    return "An internal error occurred."


def is_safe_webhook_url(url: str) -> bool:
    """Block SSRF to internal networks via alert webhooks."""
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return False
    if parsed.scheme not in ("https", "http"):
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if host in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        return False
    if host.endswith(".local") or host.endswith(".internal"):
        return False
    try:
        addr = ipaddress.ip_address(host)
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
            return False
    except ValueError:
        pass
    return True


def load_json_file_bounded(path: Path, max_bytes: int = MAX_JSON_BYTES) -> Any:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"JSON file too large: {size} bytes")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path_under(base: Path, target: Path) -> Path:
    """Prevent path traversal — target must resolve under base."""
    base_resolved = base.resolve()
    candidate = (base_resolved / target).resolve()
    if base_resolved not in candidate.parents and candidate != base_resolved:
        raise ValueError("Path escapes allowed directory")
    return candidate


def validate_model_payload(payload: dict[str, Any]) -> None:
    layers = payload.get("layer_sizes")
    if not isinstance(layers, list) or not layers or len(layers) > MAX_MODEL_LAYERS:
        raise ValueError("Invalid layer_sizes")
    if any(not isinstance(n, int) or n <= 0 or n > MAX_LAYER_SIZE for n in layers):
        raise ValueError("Layer size out of bounds")
    weights = payload.get("weights")
    biases = payload.get("biases")
    if not isinstance(weights, list) or not isinstance(biases, list):
        raise ValueError("Invalid weights/biases")
    expected = len(layers) - 1
    if len(weights) != expected or len(biases) != expected:
        raise ValueError("Weight layer mismatch")


@contextmanager
def exclusive_training_lock():
    """One expensive training job at a time — mitigates DoS."""
    if not _training_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Training job already in progress")
    try:
        yield
    finally:
        _training_lock.release()


def training_lock_available() -> bool:
    if _training_lock.acquire(blocking=False):
        _training_lock.release()
        return True
    return False
