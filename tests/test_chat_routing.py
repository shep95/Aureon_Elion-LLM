"""Chat routing — philosophy fast path, code route, and predict timeout."""

from __future__ import annotations

from app.chat_service import (
    _code_payload,
    _handle_named_entity,
    _predict_with_timeout,
    _simple_nl_response,
    chat,
    is_code_question,
    is_named_entity_question,
)


def test_simple_nl_god_and_math():
    god = _simple_nl_response("Who is God to you?")
    assert god
    assert "deity" in god.lower() or "god" in god.lower()
    math = _simple_nl_response("What is math?")
    assert math
    assert "mathematics" in math.lower() or "numbers" in math.lower()


def test_arithmetic_routes_before_deep_concept():
    result = chat("What is 2+2", session_id="arith-1")
    assert result["reply"] == "4"
    assert result.get("deterministic", {}).get("evaluator") == "deterministic_arithmetic"
    assert result["kind"] == "chat"


def test_is_code_question():
    assert is_code_question("write a python function to add two numbers")
    assert not is_code_question("what is the capital of france")


def test_code_payload_bootstrap_syntax_valid(monkeypatch):
    import app.chat_service as cs
    import brain.code_master as cm

    monkeypatch.setattr(
        cm,
        "generate_master_code",
        lambda question, predict_fn=None: {
            "answer": "def add(a, b): return a + b",
            "code_eval": {"syntax_valid": True, "score": 1.0, "passed_tests": True},
            "method": "retrieval_verified",
            "citations": [],
            "confidence": 0.95,
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
    from brain.system_messages import FALLBACK_CORPUS

    assert FALLBACK_CORPUS in result.get("answer", "")


def test_is_named_entity_question_positive():
    assert is_named_entity_question("who is Adam and Eve") is True
    assert is_named_entity_question("who was John Snow") is True
    assert is_named_entity_question("tell me about Zophiel") is True
    assert is_named_entity_question("who is Nikola Tesla") is True


def test_is_named_entity_question_negative():
    assert is_named_entity_question("what is mathematics") is False
    assert is_named_entity_question("how does backpropagation work") is False
    assert is_named_entity_question("what is the meaning of life") is False


def test_named_entity_bypasses_classifier(monkeypatch):
    classify_called: list[str] = []

    def fake_classify(text: str):
        classify_called.append(text)
        return {"label": "biology.adam_developmental", "confidence": 0.99}

    monkeypatch.setattr("app.chat_service._classify_message", fake_classify)
    monkeypatch.setattr(
        "app.chat_service._predict_with_timeout",
        lambda *a, **kw: {
            "answer": "Adam and Eve are biblical figures from Genesis.",
            "abstained": False,
            "citations": [],
        },
    )
    monkeypatch.setattr(
        "app.chat_service.retrieve_with_citations",
        lambda q, **kw: ("genesis creation story", [], []),
        raising=False,
    )

    result = chat("who is Adam and Eve", session_id="ne-1")
    assert result["kind"] == "named_entity"
    assert "Adam" in result["reply"]
    assert classify_called == []


def test_handle_named_entity_thin_corpus(monkeypatch):
    monkeypatch.setattr(
        "brain.vector_rag.retrieve_with_citations",
        lambda q, **kw: ("", [], []),
    )
    monkeypatch.setattr("brain.web_search.search", lambda q, **kw: [])
    monkeypatch.setattr(
        "app.chat_service._predict_with_timeout",
        lambda *a, **kw: {"answer": "short", "abstained": True},
    )

    result = _handle_named_entity("who is Zophiel", session_id="ne-thin")
    assert result["kind"] == "named_entity_thin"
    assert "named entity" in result["reply"].lower()
