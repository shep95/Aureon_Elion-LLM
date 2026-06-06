"""Chaotic entropy engine — per-message padding and fingerprints (nomad port)."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import struct
import time
from typing import Any

from app.nomad.occult_veil import hkdf_sha256


def chaos_entropy_enabled() -> bool:
    raw = os.environ.get("AUREON_CHAOS_ENTROPY", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def chaos_master_key() -> bytes | None:
    """Derive chaos master key from audit chain key or API key."""
    for env_name in ("AUREON_AUDIT_CHAIN_KEY", "AUREON_API_KEY", "AUREON_CHAOS_MASTER_KEY"):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            return hashlib.sha256(raw.encode("utf-8")).digest()
    return None


def derive_message_salt(
    master_key: bytes,
    correlation_id: str,
    sequence: int,
    timestamp_ms: int,
) -> bytes:
    info = f"nomad-chaos-salt:{correlation_id}:{sequence}".encode("utf-8")
    salt_input = f"{timestamp_ms}:{sequence}".encode("utf-8")
    return hkdf_sha256(master_key, salt_input, info, 32)


def derive_pad_length(
    master_key: bytes,
    correlation_id: str,
    sequence: int,
    timestamp_ms: int,
    *,
    minimum: int = 16,
    maximum: int = 272,
) -> int:
    salt = derive_message_salt(master_key, correlation_id, sequence, timestamp_ms)
    span = maximum - minimum + 1
    return minimum + (salt[0] ^ salt[1] ^ salt[2]) % span


def derive_suffix_length(
    master_key: bytes,
    correlation_id: str,
    sequence: int,
    timestamp_ms: int,
) -> int:
    salt = derive_message_salt(master_key, correlation_id, sequence, timestamp_ms)
    return 8 + (salt[3] ^ salt[4]) % 120


def apply_chaotic_padding(
    body: bytes,
    master_key: bytes,
    correlation_id: str,
    sequence: int,
    timestamp_ms: int,
) -> bytes:
    """Prefix + body + suffix — ciphertext length never matches plaintext length."""
    prefix_len = derive_pad_length(master_key, correlation_id, sequence, timestamp_ms)
    suffix_len = derive_suffix_length(master_key, correlation_id, sequence, timestamp_ms)
    prefix = secrets.token_bytes(prefix_len)
    suffix = secrets.token_bytes(suffix_len)
    header = struct.pack(">HH", prefix_len, suffix_len)
    return header + prefix + body + suffix


def strip_chaotic_padding(
    padded: bytes,
    master_key: bytes,
    correlation_id: str,
    sequence: int,
    timestamp_ms: int,
) -> bytes:
    if len(padded) < 4:
        raise ValueError("Chaotic padding header missing")
    prefix_len, suffix_len = struct.unpack(">HH", padded[:4])
    expected_prefix = derive_pad_length(master_key, correlation_id, sequence, timestamp_ms)
    expected_suffix = derive_suffix_length(master_key, correlation_id, sequence, timestamp_ms)
    if prefix_len != expected_prefix or suffix_len != expected_suffix:
        raise ValueError("Chaotic padding length mismatch — possible tamper")
    body_start = 4 + prefix_len
    body_end = len(padded) - suffix_len
    if body_start > body_end:
        raise ValueError("Chaotic padding bounds invalid")
    return padded[body_start:body_end]


def derive_shuffled_order(
    items: list[str],
    master_key: bytes,
    correlation_id: str,
    sequence: int,
    timestamp_ms: int,
    label: str,
) -> list[str]:
    """Fisher-Yates shuffle driven by key material — same inputs yield same order."""
    seed = hmac.new(
        master_key,
        f"chaos-order:{label}:{correlation_id}:{sequence}:{timestamp_ms}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    arr = list(items)
    for i in range(len(arr) - 1, 0, -1):
        j = seed[i % len(seed)] % (i + 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def derive_chaos_fingerprint(
    master_key: bytes,
    correlation_id: str,
    sequence: int,
    timestamp_ms: int,
) -> bytes:
    return hmac.new(
        master_key,
        f"chaos-fingerprint:{correlation_id}:{sequence}:{timestamp_ms}".encode("utf-8"),
        hashlib.sha256,
    ).digest()[:8]


def chaos_response_headers(correlation_id: str, sequence: int = 0) -> dict[str, str]:
    """Headers for authenticated mutating responses — timing + fingerprint."""
    headers: dict[str, str] = {}
    if not chaos_entropy_enabled():
        return headers
    key = chaos_master_key()
    if not key:
        return headers
    ts = int(time.time() * 1000)
    fp = derive_chaos_fingerprint(key, correlation_id, sequence, ts)
    headers["X-Chaos-Fingerprint"] = fp.hex()
    return headers


def chaos_status() -> dict[str, Any]:
    return {
        "enabled": chaos_entropy_enabled(),
        "master_key_configured": chaos_master_key() is not None,
        "pad_range": {"prefix_min": 16, "prefix_max": 272, "suffix_min": 8, "suffix_max": 128},
    }
