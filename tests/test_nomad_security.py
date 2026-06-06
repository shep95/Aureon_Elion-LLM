"""Nomad cyber algorithm security layer tests."""

from __future__ import annotations

import hashlib
import secrets
import time

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.nomad.client_allowlist import is_client_allowed
from app.nomad.organ_registry import organ_activation_order
from app.nomad.supply_spleen import verify_supply_chain
from app.organism import AureonOrganism

client = TestClient(app)


def auth_headers(key: str = "test-secret-key") -> dict[str, str]:
    return {
        "X-API-Key": key,
        "X-Timestamp": str(int(time.time() * 1000)),
        "X-Nonce": secrets.token_hex(16),
        "X-Correlation-ID": f"test-{secrets.token_hex(8)}",
    }


def test_organ_registry_has_fourteen_organs():
    assert len(organ_activation_order()) == 14


def test_supply_spleen_verifies():
    result = verify_supply_chain()
    assert result["ok"] is True
    assert "sha256" in str(result.get("detail", "")).lower() or "hash" in result


def test_client_allowlist_hash():
    key = "my-test-key"
    digest = hashlib.sha256(key.encode()).hexdigest()
    assert is_client_allowed(key) is True
    assert is_client_allowed("other") is True  # allowlist off


def test_client_allowlist_enforced(monkeypatch):
    key = "allowed-key"
    digest = hashlib.sha256(key.encode()).hexdigest()
    monkeypatch.setenv("AUREON_CLIENT_ALLOWLIST", digest)
    assert is_client_allowed(key) is True
    assert is_client_allowed("blocked-key") is False


def test_security_doctrine_public():
    response = client.get("/security/doctrine")
    assert response.status_code == 200
    body = response.json()
    assert "doctrine" in body
    assert body["adapted_for"] == "SOLIA"
    assert len(body["organs"]) == 14


def test_security_status_public():
    response = client.get("/security/status")
    assert response.status_code == 200
    body = response.json()
    assert body["stack"] == "nomad_cyber_algorithm-adapted"
    assert body["organ_count"] == 14
    assert "nomad_adaptations" in body


def test_security_audit_requires_api_key(monkeypatch):
    monkeypatch.setenv("AUREON_API_KEY", "audit-test-key")
    assert client.get("/security/audit").status_code == 401
    response = client.get("/security/audit", headers=auth_headers(key="audit-test-key"))
    assert response.status_code == 200
    assert "entries" in response.json()


def test_organism_fourteen_organs():
    org = AureonOrganism()
    org.pulse()
    report = org.get_vitals_report()
    assert len(report["organs"]) == 14
    assert report["stack"] == "nomad_cyber_algorithm-adapted"
