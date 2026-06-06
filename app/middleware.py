"""HTTP security middleware."""

from __future__ import annotations

import logging
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.audit import get_audit_log
from app.nomad.chaos_veil import apply_chaos_veil
from app.organism import OrganismLockdownError, get_organism
from app.rate_limit import get_rate_limiter
from app.replay_guard import get_replay_guard, replay_guard_enabled
from app.security import MAX_JSON_BYTES, api_key_required

logger = logging.getLogger(__name__)

MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
PUBLIC_PATHS = frozenset({
    "/health",
    "/health/ready",
    "/organism/vitals",
    "/security/doctrine",
    "/security/status",
    "/chat",
    "/api/chat",
    "/api/chat/file",
    "/api/chat/access",
    "/api/chat/learning",
    "/api/chat/timeline",
    "/api/chat/self-inquiry",
    "/docs",
    "/openapi.json",
    "/redoc",
})


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _correlation_id(request: Request) -> str:
    cid = request.headers.get("X-Correlation-ID", "").strip()
    if not cid:
        cid = f"aureon-{uuid.uuid4().hex[:16]}"
    request.state.correlation_id = cid
    return cid


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'"
        )
        cid = getattr(request.state, "correlation_id", None)
        if cid:
            response.headers["X-Correlation-ID"] = cid
        return response


class SecurityGatewayMiddleware(BaseHTTPMiddleware):
    """
    Nomad-inspired gateway layer: body size, correlation IDs, rate limits,
    replay guard, and organism lockdown on mutating routes.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        cid = _correlation_id(request)
        path = request.url.path
        method = request.method.upper()
        ip = _client_ip(request)

        if method in MUTATING_METHODS:
            apply_chaos_veil()
            content_length = request.headers.get("Content-Length")
            if content_length:
                try:
                    if int(content_length) > MAX_JSON_BYTES:
                        get_audit_log().record(
                            "rate_limit_exceeded",
                            correlation_id=cid,
                            peer=ip,
                            detail=f"body too large: {content_length}",
                        )
                        return JSONResponse(
                            status_code=413,
                            content={"detail": "Request body too large"},
                            headers={"X-Correlation-ID": cid},
                        )
                except ValueError:
                    pass

            is_public = path in PUBLIC_PATHS
            if not is_public:
                organism = get_organism()
                if not organism.is_vital():
                    get_audit_log().record(
                        "organism_lockdown",
                        correlation_id=cid,
                        peer=ip,
                        detail=f"{method} {path}",
                    )
                    return JSONResponse(
                        status_code=503,
                        content={
                            "detail": "ORGANISM_LOCKDOWN",
                            "reason": organism.get_vitals_report().get("lockdown_reason"),
                        },
                        headers={"X-Correlation-ID": cid},
                    )

                if not get_rate_limiter().try_acquire(ip):
                    get_audit_log().record(
                        "rate_limit_exceeded",
                        correlation_id=cid,
                        peer=ip,
                        detail=f"{method} {path}",
                    )
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Rate limit exceeded"},
                        headers={"X-Correlation-ID": cid},
                    )

                if api_key_required() and replay_guard_enabled():
                    try:
                        ts_raw = request.headers.get("X-Timestamp", "").strip()
                        nonce = request.headers.get("X-Nonce", "").strip()
                        if not ts_raw or not nonce:
                            return JSONResponse(
                                status_code=400,
                                content={
                                    "detail": "X-Timestamp and X-Nonce headers required when API key auth is enabled"
                                },
                                headers={"X-Correlation-ID": cid},
                            )
                        get_replay_guard().validate(nonce, int(ts_raw), cid)
                    except ValueError as exc:
                        event = "replay_detected" if "Replay" in str(exc) else "auth_failed"
                        get_audit_log().record(
                            event,
                            correlation_id=cid,
                            peer=ip,
                            detail=str(exc),
                        )
                        return JSONResponse(
                            status_code=400 if event == "auth_failed" else 409,
                            content={"detail": str(exc)},
                            headers={"X-Correlation-ID": cid},
                        )

        try:
            response = await call_next(request)
        except OrganismLockdownError as exc:
            get_audit_log().record("organism_lockdown", correlation_id=cid, peer=ip, detail=str(exc))
            return JSONResponse(
                status_code=503,
                content={"detail": str(exc)},
                headers={"X-Correlation-ID": cid},
            )

        if method in MUTATING_METHODS and path not in PUBLIC_PATHS and response.status_code < 400:
            get_audit_log().record(
                "mutating_request",
                correlation_id=cid,
                peer=ip,
                detail=f"{method} {path} -> {response.status_code}",
            )
            from app.nomad.chaos_entropy import chaos_response_headers

            for header, value in chaos_response_headers(cid).items():
                response.headers[header] = value
        return response
