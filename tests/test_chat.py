"""Chat API tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.chat_service import chat, estimate_learning_timeline
from brain.domains.taxonomy import total_micro_subdomains
from app.main import app

client = TestClient(app)


def test_chat_help():
    result = chat("/help")
    assert result["kind"] == "help"
    assert "supervised" in result["reply"].lower()


def test_chat_grades_timeline():
    result = chat("/grades")
    assert result["kind"] == "grades"
    assert "timeline" in result


def test_estimate_timeline():
    t = estimate_learning_timeline(interval_sec=3600, micro_subdomain_count=total_micro_subdomains())
    assert t["grade_count"] == 7
    assert "one_micro_subdomain_all_grades" in t["estimates"]


def test_chat_ui_route():
    r = client.get("/chat")
    assert r.status_code == 200
    assert "Aureon" in r.text


def test_api_chat_post():
    r = client.post("/api/chat", json={"message": "/status"})
    assert r.status_code == 200
    assert r.json()["kind"] == "status"


def test_api_chat_learning():
    r = client.get("/api/chat/learning")
    assert r.status_code == 200
    assert "auto_learn" in r.json()
