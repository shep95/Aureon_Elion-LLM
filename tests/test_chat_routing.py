"""Chat routing — philosophy fast path, code route, and predict timeout."""

from __future__ import annotations

from app.chat_service import _code_payload, _predict_with_timeout, _simple_nl_response, is_code_question


def test_simple_nl_god_and_math():
    god = _simple_nl_response("Who is God to you?")
    assert god
    assert "deity" in god.lower() or "god" in god.lower()
    math = _simple_nl_response("What is math?")
    assert math
    assert "mathematics" in math.lower() or "numbers" in math.lower()


def test_is_code_question():
    assert is_code_question("write a python function to add two numbers")
    assert not is_code_question("what is the capital of france")


def test_code_payload_bootstrap_syntax_valid(monkeypatch):
    import app.chat_service as cs

    monkeypatch.setattr(
        cs,
        "_predict_with_timeout",
        lambda question, session_id=None, seconds=None, force=False: {
            "answer": "def add(a, b): return a + b",
            "confidence": 0.9,
            "model": "stacked_attention_lm",
            "citations": [],
        },
    )
    payload = cs._code_payload("write a python function to add two numbers", session_id="code-test")
    assert payload is not None
    assert payload["kind"] == "code"
    assert payload["code_eval"]["syntax_valid"] is True
    assert "def add" in payload["reply"]


def test_predict_timeout_fallback(monkeypatch):
    monkeypatch.setenv("AUREON_PREDICT_TIMEOUT_SEC", "0.001")

    def slow_predict(_question: str, **_kwargs):
        import time

        time.sleep(2)
        return {"answer": "never"}

    import app.chat_service as cs

    monkeypatch.setattr(cs, "predict_with_steps", slow_predict)
    result = cs._predict_with_timeout("what is quantum gravity")
    assert result is not None
    assert result.get("timed_out") is True
    assert result.get("answer")
