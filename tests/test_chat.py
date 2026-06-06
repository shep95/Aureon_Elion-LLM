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
    assert "mind" in result["reply"].lower()


def test_chat_mind_command():
    result = chat("/mind")
    assert result["kind"] == "mind"


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


def test_chat_simple_question():
    result = chat("What is Aureon?")
    assert result["kind"] == "chat"
    assert result.get("simple_qa") is True
    assert len(result["reply"]) <= 160
    assert "supervised" in result["reply"].lower()


def test_chat_prediction_brain(tmp_path, monkeypatch):
    import brain.predict_engine as pe

    monkeypatch.setenv("AUREON_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PIPELINE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AUREON_PREDICT_EPOCHS", "80")
    monkeypatch.setenv("AUREON_PREDICT_MAX_SEQ", "128")
    monkeypatch.setenv("AUREON_PREDICT_D_MODEL", "48")
    monkeypatch.setenv("AUREON_PREDICT_LAYERS", "4")
    monkeypatch.setenv("AUREON_PREDICT_MAX_VOCAB", "2000")
    monkeypatch.setenv("AUREON_PREDICT_TIMEOUT_SEC", "120")
    monkeypatch.setattr(pe, "MODEL_DIR", tmp_path / "models" / "predict_brain")
    monkeypatch.setattr(pe, "_model", None)
    monkeypatch.setattr(pe, "_ready", False)
    result = chat("What is the capital of France?")
    assert result["kind"] == "chat"
    assert result.get("brain_predict") is True
    assert "paris" in result["reply"].lower()
    assert result["prediction"]["pipeline"][0]["name"] == "tokenize"
    assert result["prediction"]["context_window"] >= 128


def test_api_chat_learning():
    r = client.get("/api/chat/learning")
    assert r.status_code == 200
    assert "auto_learn" in r.json()
