"""Nomad-style security API routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.audit import get_audit_log
from app.nomad.chaos_veil import chaos_jitter_ms, chaos_veil_enabled
from app.nomad.client_allowlist import allowlist_enabled
from app.nomad.chaos_entropy import chaos_status
from app.nomad.occult_veil import occult_status, occult_veil_enabled
from app.nomad.organ_registry import NOMAD_DOCTRINE, ORGAN_META
from app.nomad.organism_pulse import pulse_interval_sec
from app.nomad.shamir import ShamirShare, combine_shares, split_secret
from app.nomad.supply_spleen import compute_requirements_hash
from app.organism import get_organism
from app.security import get_api_key_from_request, require_mutating_access, verify_api_key

router = APIRouter(prefix="/security", tags=["security"])

Mutating = Annotated[None, Depends(require_mutating_access)]


def _require_audit_access(request: Request) -> None:
    verify_api_key(
        get_api_key_from_request(request),
        correlation_id=getattr(request.state, "correlation_id", None),
        peer=request.client.host if request.client else None,
    )


AuditAccess = Annotated[None, Depends(_require_audit_access)]


@router.get("/doctrine")
def security_doctrine() -> dict:
    """Public — nomad sovereign organism doctrine."""
    return {
        "doctrine": NOMAD_DOCTRINE,
        "source": "https://github.com/houseofasher/nomad_cyber_algorithm",
        "adapted_for": "Aureon-LLM",
        "organs": {oid: meta["role"] for oid, meta in ORGAN_META.items()},
    }


@router.get("/status")
def security_status() -> dict:
    """Public — full organism vitals + nomad stack metadata."""
    organism = get_organism()
    organism.pulse()
    report = organism.get_vitals_report()
    chain = get_audit_log().verify_chain()
    return {
        **report,
        "nomad_adaptations": {
            "chaos_veil": chaos_veil_enabled(),
            "chaos_jitter_ms": chaos_jitter_ms(),
            "chaos_entropy": chaos_status(),
            "occult_veil": occult_status(),
            "client_allowlist": allowlist_enabled(),
            "organism_pulse_sec": pulse_interval_sec(),
            "requirements_sha256": compute_requirements_hash(),
        },
        "audit_chain_valid": chain["valid"],
        "audit_entries": chain["length"],
    }


@router.get("/audit")
def security_audit(
    _auth: AuditAccess,
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
) -> dict:
    """Authenticated — tail of tamper-evident audit log."""
    entries = get_audit_log().query(limit=limit)
    chain = get_audit_log().verify_chain()
    return {
        "valid": chain["valid"],
        "length": chain["length"],
        "errors": chain["errors"],
        "entries": entries,
    }


@router.post("/pulse")
def security_force_pulse(_auth: Mutating) -> dict:
    """Authenticated — force organism pulse (ops/debug)."""
    organism = get_organism()
    organism.pulse()
    return organism.get_vitals_report()


@router.get("/occult/epoch")
def security_occult_epoch() -> dict:
    """Public — planetary epoch slot and occult veil metadata."""
    return occult_status()


@router.get("/chaos/status")
def security_chaos_status() -> dict:
    """Public — chaos entropy engine configuration."""
    return chaos_status()


@router.post("/key-ceremony")
def security_key_ceremony(body: dict, _auth: Mutating) -> dict:
    """
    Authenticated — Shamir M-of-N key ceremony (nomad key_ceremony pattern).

    Split: { "action": "split", "secret_hex": "...", "threshold": 3, "shares": 5 }
    Combine: { "action": "combine", "shares": [{"index": 1, "data_hex": "..."}, ...] }
    """
    action = str(body.get("action", "")).strip().lower()
    if action == "split":
        secret_hex = str(body.get("secret_hex", "")).strip()
        if not secret_hex:
            raise HTTPException(status_code=400, detail="secret_hex required")
        try:
            secret = bytes.fromhex(secret_hex)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid secret_hex") from exc
        threshold = int(body.get("threshold", 3))
        shares = int(body.get("shares", 5))
        if len(secret) > 256:
            raise HTTPException(status_code=400, detail="secret too large (max 256 bytes)")
        try:
            parts = split_secret(secret, threshold, shares)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "action": "split",
            "threshold": threshold,
            "shares": shares,
            "parts": [{"index": p.index, "data_hex": p.data.hex()} for p in parts],
        }

    if action == "combine":
        raw_shares = body.get("shares")
        if not isinstance(raw_shares, list) or len(raw_shares) < 2:
            raise HTTPException(status_code=400, detail="shares list required (min 2)")
        try:
            share_objs = [
                ShamirShare(index=int(s["index"]), data=bytes.fromhex(str(s["data_hex"])))
                for s in raw_shares
            ]
            recovered = combine_shares(share_objs)
        except (KeyError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"action": "combine", "secret_hex": recovered.hex(), "share_count": len(share_objs)}

    raise HTTPException(status_code=400, detail="action must be split or combine")
