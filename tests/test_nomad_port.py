"""Tests for nomad_cyber_algorithm ports into Aureon-LLM."""

from __future__ import annotations

import secrets
import time

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.nomad.chaos_entropy import (
    apply_chaotic_padding,
    chaos_master_key,
    derive_chaos_fingerprint,
    derive_shuffled_order,
    strip_chaotic_padding,
)
from app.nomad.crypto_core import verify_crypto_core
from app.nomad.occult_veil import derive_occult_veil_key, occult_veil_transform, planetary_epoch_slot
from app.nomad.organ_registry import organ_activation_order
from app.nomad.shamir import combine_shares, split_secret
from app.organism import AureonOrganism

client = TestClient(app)


def auth_headers(key: str = "test-secret-key") -> dict[str, str]:
    return {
        "X-API-Key": key,
        "X-Timestamp": str(int(time.time() * 1000)),
        "X-Nonce": secrets.token_hex(16),
        "X-Correlation-ID": f"test-{secrets.token_hex(8)}",
    }


def test_crypto_core_self_test():
    result = verify_crypto_core()
    assert result["ok"] is True


def test_occult_veil_roundtrip():
    master = b"master-key-material-for-veil-test!!"
    ts = 1_700_000_000_000
    cid = "corr-abc"
    key = derive_occult_veil_key(master, cid, ts)
    plain = b"sensitive audit payload"
    cipher = occult_veil_transform(plain, key)
    assert cipher != plain
    assert occult_veil_transform(cipher, key) == plain
    assert planetary_epoch_slot(ts) >= 0


def test_chaos_padding_roundtrip():
    key = b"chaos-master-key-for-padding-test!"
    body = b"hello nomad chaos"
    cid = "req-123"
    seq = 7
    ts = 1_700_000_000_000
    padded = apply_chaotic_padding(body, key, cid, seq, ts)
    assert len(padded) > len(body) + 4
    assert strip_chaotic_padding(padded, key, cid, seq, ts) == body


def test_chaos_fingerprint_deterministic():
    key = b"fp-key"
    fp1 = derive_chaos_fingerprint(key, "cid", 1, 1000)
    fp2 = derive_chaos_fingerprint(key, "cid", 1, 1000)
    fp3 = derive_chaos_fingerprint(key, "cid", 2, 1000)
    assert fp1 == fp2
    assert fp1 != fp3
    assert len(fp1) == 8


def test_chaos_shuffle_stable():
    key = b"shuffle-key"
    items = ["a", "b", "c", "d", "e"]
    a = derive_shuffled_order(items, key, "cid", 0, 1000, "layers")
    b = derive_shuffled_order(items, key, "cid", 0, 1000, "layers")
    c = derive_shuffled_order(items, key, "cid", 1, 1000, "layers")
    assert a == b
    assert sorted(a) == sorted(items)
    assert a != c or len(items) <= 1


def test_shamir_split_combine():
    secret = b"aureon-key-ceremony-secret"
    parts = split_secret(secret, threshold=3, shares=5)
    assert len(parts) == 5
    recovered = combine_shares(parts[:3])
    assert recovered == secret


def test_organ_registry_fourteen_organs():
    assert len(organ_activation_order()) == 14


def test_organism_includes_nomaf_ports():
    org = AureonOrganism()
    org.pulse()
    report = org.get_vitals_report()
    ids = {o["id"] for o in report["organs"]}
    assert "crypto_core" in ids
    assert "occult_veil" in ids
    assert "chaos_entropy" in ids


def test_security_occult_epoch_public():
    r = client.get("/security/occult/epoch")
    assert r.status_code == 200
    assert "planetary_epoch" in r.json()


def test_security_chaos_status_public():
    r = client.get("/security/chaos/status")
    assert r.status_code == 200
    assert "enabled" in r.json()


def test_security_key_ceremony_split(monkeypatch):
    monkeypatch.setenv("AUREON_API_KEY", "ceremony-key")
    secret_hex = secrets.token_bytes(16).hex()
    r = client.post(
        "/security/key-ceremony",
        json={"action": "split", "secret_hex": secret_hex, "threshold": 2, "shares": 3},
        headers=auth_headers("ceremony-key"),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["action"] == "split"
    assert len(body["parts"]) == 3


def test_chaos_fingerprint_header_on_mutating(monkeypatch):
    monkeypatch.setenv("AUREON_API_KEY", "fp-test-key")
    monkeypatch.setenv("AUREON_CHAOS_ENTROPY", "1")
    r = client.post("/security/pulse", headers=auth_headers("fp-test-key"))
    assert r.status_code == 200
    if chaos_master_key():
        assert "X-Chaos-Fingerprint" in r.headers
