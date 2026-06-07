"""Analytical brain routing tests."""

from __future__ import annotations

from app.chat_service import chat
from brain.analytical_brain import route_analytical_question


def test_analytical_brain_routes_without_answer_text():
    route = route_analytical_question(
        "What were the real economic causes behind the fall of the Roman Empire - not the textbook version?"
    )
    assert route is not None
    assert route.domain == "history_and_civilization"
    assert route.subject == "economic causes of Rome's fall"
    assert not hasattr(route, "answer")


def test_chat_quantum_artificial_intelligence_question_answers_directly():
    result = chat(
        "Explain Quantum Artificial intelligence to me and how it works",
        session_id="analytical-quantum-ai",
    )
    assert result["kind"] == "analytical"
    assert result["human_understanding"]["subject"] == "quantum artificial intelligence"
    assert result["analytical_route"]["method"] == "analytical_route"
    reply = result["reply"].lower()
    assert "quantum artificial intelligence" in reply
    assert "classical ai" in reply
    assert "compute time limit" not in reply


def test_chat_quantum_computer_question_answers_directly():
    result = chat(
        "what is a quantum computer and how does it work",
        session_id="analytical-quantum-computer",
    )
    assert result["kind"] == "analytical"
    assert result["human_understanding"]["subject"] == "quantum computer"
    assert result["analytical_route"]["method"] == "analytical_route"
    reply = result["reply"].lower()
    assert "qubit" in reply
    assert "superposition" in reply
    assert "compute time limit" not in reply


def test_quantum_computer_abstains_when_relevance_gate_fails(monkeypatch):
    class Hit:
        title = "Silicon Valley"
        text = "Startup funding news from Silicon Valley and India."

    monkeypatch.setattr(
        "app.chat_service._predict_with_search_fallback",
        lambda *a, **kw: {"answer": "Silicon Valley startup funding news.", "confidence": 0.9},
    )
    monkeypatch.setattr(
        "brain.vector_rag.retrieve_with_citations",
        lambda *a, **kw: (
            "",
            [Hit()],
            [{"title": "Silicon Valley", "source": "test", "score": 0.9}],
        ),
    )
    result = chat(
        "what is a quantum computer and how does it work",
        session_id="analytical-relevance-abstain",
    )
    assert result["kind"] == "relevance_abstain"
    assert result["relevance"]["passed"] is False
    assert "Silicon Valley startup funding news" not in result["reply"]


def test_analytical_brain_routes_narcissism_without_answering():
    route = route_analytical_question(
        "What is the difference between narcissism and dark triad personality - "
        "and how do you identify each in conversation?"
    )
    assert route is not None
    assert route.subject == "narcissism vs dark triad"
    assert not hasattr(route, "answer")


def test_analytical_brain_routes_gut_brain_without_answering():
    route = route_analytical_question(
        "What does modern science say about the gut-brain connection and how does it affect mental health?",
    )
    assert route is not None
    assert route.subject == "gut-brain connection"
    assert not hasattr(route, "answer")


def test_analytical_brain_routes_future_cycles_without_answering():
    route = route_analytical_question(
        "Based on historical cycles, what major global shift is most likely to happen in the next 10 years?",
    )
    assert route is not None
    assert route.subject == "major global shift from historical cycles"
    assert not hasattr(route, "answer")

