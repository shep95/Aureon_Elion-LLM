"""Analytical brain routing tests."""

from __future__ import annotations

from app.chat_service import chat
from brain.analytical_brain import answer_analytical_question


def test_analytical_brain_answers_roman_economics():
    answer = answer_analytical_question(
        "What were the real economic causes behind the fall of the Roman Empire - not the textbook version?"
    )
    assert answer is not None
    assert answer.domain == "history_and_civilization"
    assert "fiscal base" in answer.answer.lower()


def test_chat_narcissism_question_not_identity():
    result = chat(
        "What is the difference between narcissism and dark triad personality - "
        "and how do you identify each in conversation?",
        session_id="analytical-narcissism",
    )
    assert result["kind"] == "analytical"
    assert result["human_understanding"]["subject"] == "narcissism vs dark triad"
    assert "I am Aureon" not in result["reply"]
    assert "Machiavellian" in result["reply"]


def test_chat_gut_brain_question_not_modern_ciper():
    result = chat(
        "What does modern science say about the gut-brain connection and how does it affect mental health?",
        session_id="analytical-gut-brain",
    )
    assert result["kind"] == "analytical"
    assert result["human_understanding"]["subject"] == "gut-brain connection"
    assert "modern -" not in result["reply"].lower()
    assert "microbiome" in result["reply"].lower()


def test_chat_future_cycles_question_not_based_ciper():
    result = chat(
        "Based on historical cycles, what major global shift is most likely to happen in the next 10 years?",
        session_id="analytical-future",
    )
    assert result["kind"] == "analytical"
    assert result["human_understanding"]["subject"] == "major global shift from historical cycles"
    assert "rights-based ethics" not in result["reply"].lower()
    assert "legitimacy crisis" in result["reply"].lower()

