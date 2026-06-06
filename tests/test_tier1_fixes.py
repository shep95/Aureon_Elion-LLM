"""Tier 1 Zophiel routing fixes."""

from __future__ import annotations

from app.chat_service import _resolve_followup, _simple_nl_response, learning_snapshot
from app.session_memory import append_turn, get_history
from brain.predict_engine import _bootstrap_answer


def test_philosophy_bootstrap():
    assert _bootstrap_answer("Who is God to you?")
    assert _bootstrap_answer("What is math?")


def test_simple_nl_god_instant():
    reply = _simple_nl_response("Who is God to you?")
    assert reply and ("deity" in reply.lower() or "god" in reply.lower())


def test_session_memory_and_followup():
    append_turn("s1", user="What is math?", assistant="Numbers and patterns.")
    hist = get_history("s1")
    assert len(hist) == 1
    resolved = _resolve_followup("tell me more about that", "s1")
    assert "math" in resolved.lower()


def test_learning_snapshot_cache(monkeypatch):
    monkeypatch.setenv("AUREON_LEARNING_SNAPSHOT_TTL_SEC", "60")
    a = learning_snapshot(force=True)
    b = learning_snapshot()
    assert a == b
