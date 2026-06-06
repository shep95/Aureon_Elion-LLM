"""Aureon occult veil — planetary epoch anchoring + TCAP temporal entropy (nomad port)."""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Any

PLANETARY_ORBITAL_PERIODS_DAYS = {
    "saturn": 10759.22,
    "jupiter": 4332.59,
    "mars": 686.98,
    "venus": 224.70,
    "mercury": 87.97,
}

PHI = 1.618033988749895
TCAP_ANCHOR = int(PHI * 1_000_000) % 9973


def occult_veil_enabled() -> bool:
    raw = os.environ.get("AUREON_OCCULT_VEIL", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    """RFC 5869 HKDF-Expand using HMAC-SHA256."""
    if not salt:
        salt = b"\x00" * 32
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    okm = b""
    block = b""
    counter = 1
    while len(okm) < length:
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        okm += block
        counter += 1
    return okm[:length]


def planetary_epoch_slot(timestamp_ms: int) -> int:
    day = timestamp_ms / 86_400_000
    return int(day // PLANETARY_ORBITAL_PERIODS_DAYS["saturn"])


def tcap_temporal_hash(timestamp_ms: int, correlation_id: str) -> bytes:
    slot = timestamp_ms // 60_000
    entropy = (slot * TCAP_ANCHOR) ^ ord(correlation_id[0] if correlation_id else "\x00")
    return hmac.new(
        b"aureon-tcap",
        f"{correlation_id}:{slot}:{entropy}".encode("utf-8"),
        hashlib.sha256,
    ).digest()


def derive_occult_veil_key(master_key: bytes, correlation_id: str, timestamp_ms: int) -> bytes:
    epoch = planetary_epoch_slot(timestamp_ms)
    tcap = tcap_temporal_hash(timestamp_ms, correlation_id)
    salt = f"{epoch}".encode("utf-8") + tcap
    info = f"aureon-veil:{correlation_id}".encode("utf-8")
    return hkdf_sha256(master_key, salt, info, 32)


def occult_veil_transform(data: bytes, veil_key: bytes) -> bytes:
    """Symmetric XOR veil — self-inverse for encipher/decipher."""
    out = bytearray(len(data))
    for i, byte in enumerate(data):
        occult_byte = veil_key[i % len(veil_key)] ^ ((i * TCAP_ANCHOR + veil_key[0]) & 0xFF)
        out[i] = byte ^ occult_byte
    return bytes(out)


def occult_status(timestamp_ms: int | None = None) -> dict[str, Any]:
    import time

    ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
    return {
        "enabled": occult_veil_enabled(),
        "planetary_epoch": planetary_epoch_slot(ts),
        "tcap_anchor": TCAP_ANCHOR,
        "orbital_periods_days": PLANETARY_ORBITAL_PERIODS_DAYS,
    }
