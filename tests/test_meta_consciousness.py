"""Meta-consciousness self-inquiry tests."""

from __future__ import annotations

from brain.meta_consciousness import (
    combined_recent_inquiries,
    gather_self_state,
    is_meta_consciousness_enabled,
    run_meta_inquiry,
    try_meta_answer,
)


def test_meta_inquiry_logs_questions(tmp_path, monkeypatch):
    monkeypatch.setenv("AUREON_META_CONSCIOUSNESS", "1")
    monkeypatch.setenv("AUREON_DATA_DIR", str(tmp_path))

    exchanges = run_meta_inquiry(count=2, source="test")
    assert len(exchanges) == 2
    assert exchanges[0]["question"].endswith("?")
    assert exchanges[0]["answer"]
    assert exchanges[0]["theme"]
    assert "question" in exchanges[0]

    merged = combined_recent_inquiries(5)
    assert len(merged) == 2
    assert all(item.get("kind") == "meta" for item in merged)


def test_meta_disabled(monkeypatch):
    monkeypatch.setenv("AUREON_META_CONSCIOUSNESS", "0")
    monkeypatch.setenv("AUREON_SELF_INQUIRY", "0")
    assert is_meta_consciousness_enabled() is False
    assert run_meta_inquiry(count=1) == []


def test_try_meta_answer_consciousness(monkeypatch, tmp_path):
    monkeypatch.setenv("AUREON_META_CONSCIOUSNESS", "1")
    monkeypatch.setenv("AUREON_DATA_DIR", str(tmp_path))

    run_meta_inquiry(count=1, source="test")
    reply = try_meta_answer("Are you conscious?")
    assert reply
    assert "→" in reply


def test_startup_meta_inquiry(tmp_path, monkeypatch):
    monkeypatch.setenv("AUREON_META_CONSCIOUSNESS", "1")
    monkeypatch.setenv("AUREON_META_ON_STARTUP", "1")
    monkeypatch.setenv("AUREON_DATA_DIR", str(tmp_path))

    from brain.meta_consciousness import run_meta_inquiry_on_startup

    exchanges = run_meta_inquiry_on_startup()
    assert len(exchanges) == 1
    assert exchanges[0]["source"] == "startup"


def test_gather_self_state_shape(monkeypatch):
    monkeypatch.setenv("AUREON_SELF_INQUIRY", "0")
    monkeypatch.setenv("AUREON_META_CONSCIOUSNESS", "0")
    state = gather_self_state()
    assert "documents" in state
    assert "cycles_completed" in state
    assert "predict_model_version" in state
