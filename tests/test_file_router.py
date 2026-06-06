"""File router — PDF/text/image/audio routing (Tier 3–4)."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.main import app
from brain.file_router import ingest_upload, route_bytes
from brain.multimodal_processors import extract_pdf, tier_status

client = TestClient(app)


def test_tier_status_keys():
    status = tier_status()
    assert "pdf" in status
    assert "vision" in status
    assert "audio" in status
    assert "pgvector" in status


def test_route_bytes_text_file():
    data = b"Hello from a text upload.\nSecond line for context."
    result = route_bytes("notes.txt", data, message="Summarize this")
    assert result.modality == "text"
    assert "Hello from a text upload" in result.text
    assert "Summarize this" in result.text
    assert result.content_hash
    assert isinstance(result.metadata.get("embedding"), list)


def test_route_bytes_rejects_oversized(monkeypatch):
    monkeypatch.setattr("brain.file_router.MAX_UPLOAD_BYTES", 32)
    with pytest.raises(ValueError, match="too large"):
        route_bytes("big.txt", b"x" * 64)


def test_ingest_upload_without_persist():
    result = ingest_upload("readme.md", b"# Title\n\nBody content for corpus.", persist=False)
    assert result.modality == "text"
    assert result.document_id is None


def test_extract_pdf_handles_invalid_bytes():
    assert extract_pdf(b"not-a-pdf") == ""


def test_api_chat_file_text_upload():
    payload = {
        "message": "What is in this file?",
        "persist": "false",
    }
    files = {"file": ("sample.txt", io.BytesIO(b"SOLIA multimodal test file content."), "text/plain")}
    r = client.post("/api/chat/file", data=payload, files=files)
    assert r.status_code == 200
    body = r.json()
    assert "reply" in body
    assert body["file"]["modality"] == "text"
    assert "multimodal test" in body["file"]["text_preview"]


def test_api_multimodal_status_includes_tiers():
    r = client.get("/api/brain/multimodal/status")
    assert r.status_code == 200
    data = r.json()
    assert "tiers" in data
    assert "pgvector" in data
