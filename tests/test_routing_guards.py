"""Zophiel routing guard audit — logic rules, not answer patches."""

from __future__ import annotations

from app.chat_service import (
    _disambiguation_payload,
    _format_disambiguation,
    _is_classification_leak,
    _should_disambiguate,
    chat,
    is_deep_concept_question,
    is_named_entity_question,
    is_opinion_request,
    is_own_output,
    is_self_directed,
)
from app.session_memory import append_turn


def test_classification_leak_detected():
    assert _is_classification_leak("philosophy.metaphysics.philosophy_of_religion") is True
    assert _is_classification_leak("biology.genetics.adam_protein") is True
    assert _is_classification_leak("Adam and Eve are biblical figures.") is False


def test_classification_leak_blocked_on_exit(monkeypatch):
    monkeypatch.setattr(
        "app.chat_service._fallback_to_predict",
        lambda *a, **kw: "Recovered answer about the question from predict brain.",
    )
    monkeypatch.setattr(
        "app.chat_service._deterministic_payload",
        lambda text, **kw: {
            "reply": "philosophy.metaphysics.philosophy_of_religion",
            "kind": "chat",
            "session_id": kw.get("session_id"),
        },
    )
    result = chat("2+2", session_id="leak-1")
    assert "philosophy.metaphysics" not in result["reply"]
    assert result["kind"] == "predict_leak_recovered"


def test_is_own_output_detects_assistant_echo():
    sid = "own-output-1"
    prior = (
        "I mapped your question but need more corpus in that domain for a full answer. "
        "Try ingesting more documents."
    )
    append_turn(sid, user="what is X", assistant=prior)
    assert is_own_output(prior, sid) is True
    assert is_own_output("short", sid) is False


def test_self_directed_routes_to_identity(monkeypatch):
    monkeypatch.setattr(
        "brain.identity_handler.handle_identity",
        lambda text, **kw: {"reply": "I am Aureon.", "kind": "identity", "session_id": kw.get("session_id")},
    )
    result = chat("Who are you?", session_id="self-1")
    assert result["kind"] == "identity"
    assert "aureon" in result["reply"].lower()


def test_is_self_directed():
    assert is_self_directed("What are your capabilities?") is True
    assert is_self_directed("What is photosynthesis?") is False


def test_opinion_request_skips_classifier(monkeypatch):
    classify_called: list[str] = []

    def fake_classify(text: str):
        classify_called.append(text)
        return {"label": "philosophy.metaphysics.philosophy_of_religion", "confidence": 0.99}

    monkeypatch.setattr("app.chat_service._classify_message", fake_classify)
    result = chat("What are your thoughts on God?", session_id="opinion-1")
    assert "philosophy.metaphysics" not in result["reply"].lower()
    assert classify_called == []


def test_is_opinion_request():
    assert is_opinion_request("Who is God to you?") is True
    assert is_opinion_request("What is gravity?") is False


def test_named_entity_bypasses_classifier(monkeypatch):
    classify_called: list[str] = []

    monkeypatch.setattr(
        "app.chat_service._classify_message",
        lambda text: classify_called.append(text) or {"label": "biology.adam", "confidence": 0.99},
    )
    monkeypatch.setattr(
        "app.chat_service._predict_with_search_fallback",
        lambda *a, **kw: {"answer": "Adam and Eve are figures from Genesis.", "confidence": 0.8},
    )
    monkeypatch.setattr(
        "brain.vector_rag.retrieve_with_citations",
        lambda q, **kw: ("genesis creation", [], []),
    )
    result = chat("who is Adam and Eve", session_id="ne-2")
    assert result["kind"] == "named_entity"
    assert classify_called == []


def test_deep_concept_not_one_liner(monkeypatch):
    monkeypatch.setattr(
        "app.chat_service._handle_deep_concept",
        lambda text, **kw: {
            "reply": "Entropy measures disorder and information content in thermodynamic systems.",
            "kind": "deep_concept",
            "session_id": kw.get("session_id"),
            "learning": {},
        },
    )
    assert is_deep_concept_question("what is consciousness") is False
    assert is_deep_concept_question("what is god") is False
    assert is_deep_concept_question("what is entropy") is True
    assert is_deep_concept_question("how does backpropagation work") is False
    result = chat("what is entropy", session_id="deep-1")
    assert result["kind"] == "deep_concept"
    assert len(result["reply"]) > 40


def test_should_disambiguate_threshold():
    tight = [{"score": 0.70}, {"score": 0.68}]
    assert _should_disambiguate(tight) is True
    weak = [{"score": 0.50}, {"score": 0.48}]
    assert _should_disambiguate(weak) is False
    far = [{"score": 0.90}, {"score": 0.60}]
    assert _should_disambiguate(far) is False


def test_disambiguation_human_labels_only():
    matches = [
        {"slug": "biology.genetics.dna", "human_label": "DNA Structure", "score": 0.70},
        {"slug": "biology.genetics.rna", "human_label": "RNA Processing", "score": 0.68},
    ]
    payload = _disambiguation_payload("what is genetics", matches)
    assert payload is not None
    options = _format_disambiguation(matches)
    assert all("." not in opt for opt in options)
    assert "DNA Structure" in payload["reply"]


def test_search_fallback_on_low_confidence(monkeypatch):
    calls: list[str] = []

    def fake_predict(text, **kw):
        calls.append(text)
        if len(calls) == 1:
            return {"answer": "unsure", "confidence": 0.1, "abstained": False}
        return {"answer": "Web-grounded answer about quantum gravity.", "confidence": 0.6}

    monkeypatch.setenv("AUREON_WEB_SEARCH_ENABLED", "1")
    monkeypatch.setattr("app.chat_service._predict_with_timeout", fake_predict)
    monkeypatch.setattr(
        "brain.web_search.search",
        lambda q, **kw: [{"text": "Quantum gravity unifies QM and relativity.", "source": "web"}],
    )
    monkeypatch.setattr(
        "brain.web_search.format_for_context",
        lambda results: "Quantum gravity unifies QM and relativity.",
    )
    monkeypatch.setattr(
        "brain.web_search.web_search_enabled",
        lambda: True,
    )

    from app.chat_service import _predict_with_search_fallback

    result = _predict_with_search_fallback("what is quantum gravity", session_id="search-fb")
    assert len(calls) == 1
    assert result.get("search_opinion") is True
    assert "quantum gravity" in result["answer"].lower()
