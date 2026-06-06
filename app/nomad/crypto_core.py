"""Crypto core self-test — nomad liboqs integrity pattern adapted for Python stdlib."""

from __future__ import annotations

import hashlib
import hmac
import secrets

from app.nomad.occult_veil import hkdf_sha256


def verify_crypto_core() -> dict[str, str | bool]:
    """
    Verify HMAC-SHA256, HKDF, and CSPRNG primitives at organism pulse.
    Mirrors nomad crypto_core organ (liboqs self-test) for Aureon's Python stack.
    """
    errors: list[str] = []

    # RFC 4231 test vector (HMAC-SHA256)
    key = b"Jefe"
    msg = b"what do ya want for nothing?"
    expected = "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
    digest = hmac.new(key, msg, hashlib.sha256).hexdigest()
    if digest != expected:
        errors.append("HMAC-SHA256 known-vector mismatch")

    # SHA-256 empty string
    if hashlib.sha256(b"").hexdigest() != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855":
        errors.append("SHA-256 empty-vector mismatch")

    # HKDF round-trip length
    derived = hkdf_sha256(b"ikm-test", b"salt", b"info", 32)
    if len(derived) != 32 or derived == b"\x00" * 32:
        errors.append("HKDF derivation failed")

    # CSPRNG non-zero entropy
    sample = secrets.token_bytes(32)
    if sample == b"\x00" * 32:
        errors.append("CSPRNG returned zero bytes")

    if errors:
        return {"ok": False, "detail": "; ".join(errors)}
    return {"ok": True, "detail": "Python crypto core self-test passed (HMAC, SHA-256, HKDF, CSPRNG)"}
