"""Philosophy and identity routing — no raw classification as reply."""

from __future__ import annotations

from app.chat_service import chat, _simple_chat_reply


def test_god_thoughts_not_classification_label():
    result = chat("What are your thoughts on God?")
    reply = result["reply"].lower()
    assert "philosophy.metaphysics" not in reply
    assert "philosophy of religion" not in reply or "deepest" in reply
    assert len(reply) > 30


def test_do_you_believe_in_god_not_classification():
    result = chat("Do you believe in God?")
    reply = result["reply"].lower()
    assert "philosophy.metaphysics" not in reply
    assert any(w in reply for w in ("believe", "faith", "inquiry", "question", "human"))


def test_who_are_you_not_mechanical_string():
    result = chat("Who are you?")
    reply = result["reply"]
    assert result.get("kind") == "identity"
    assert "supervised ml brain — collect" not in reply.lower()
    assert "aureon" in reply.lower()


def test_simple_chat_no_raw_classification():
    payload = _simple_chat_reply("What is quantum entanglement?", session_id=None)
    reply = payload["reply"].lower()
    assert "philosophy." not in reply or len(reply) > 40
    assert "confidence)" not in reply or "predict" in payload.get("kind", "")


def test_evolve_command():
    result = chat("/evolve improve god routing")
    assert result["kind"] == "self_evolve"
    assert "fork" in result["reply"].lower()
    assert "plan" in result
