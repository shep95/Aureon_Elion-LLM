"""Echo detection and web search routing tests."""

from __future__ import annotations

from app.chat_service import chat, is_search_question
from app.session_memory import append_turn, was_my_output
from brain.system_messages import FALLBACK_CORPUS, is_system_echo


def test_is_system_echo_exact_fallback():
    assert is_system_echo(FALLBACK_CORPUS) is True


def test_is_system_echo_prefix():
    assert is_system_echo(FALLBACK_CORPUS[:50]) is True


def test_is_system_echo_normal_question():
    assert is_system_echo("What is the capital of France?") is False


def test_was_my_output_detects_prior_assistant():
    sid = "echo-test-session"
    long_reply = (
        "Paris is the capital of France and has been for centuries "
        "with rich history and culture."
    )
    append_turn(sid, user="Where is Paris?", assistant=long_reply)
    assert was_my_output(sid, long_reply) is True


def test_was_my_output_ignores_short_strings():
    sid = "echo-short-session"
    append_turn(sid, user="hi", assistant="Hello there, how can I help you today?")
    assert was_my_output(sid, "Hello there") is False


def test_chat_blocks_system_echo():
    result = chat(FALLBACK_CORPUS, session_id="sys-echo-1")
    assert result["kind"] == "echo_detected"
    assert result["reply"] != FALLBACK_CORPUS


def test_chat_blocks_self_echo_from_session():
    sid = "self-echo-1"
    prior = (
        "The mitochondria is the powerhouse of the cell — "
        "it generates ATP through cellular respiration."
    )
    append_turn(sid, user="What are mitochondria?", assistant=prior)
    result = chat(prior, session_id=sid)
    assert result["kind"] == "self_echo_detected"
    assert prior not in result["reply"]


def test_is_search_question_triggers():
    assert is_search_question("What do you think about the latest AI news?") is True
    assert is_search_question("What is photosynthesis?") is False


def test_god_opinion_still_philosophy_with_search_enabled(monkeypatch):
    monkeypatch.setenv("AUREON_WEB_SEARCH_ENABLED", "1")
    result = chat("What do you think about God?", session_id="god-opinion-1")
    assert result["kind"] != "search_opinion"
    assert "god" in result["reply"].lower() or "faith" in result["reply"].lower()


def test_search_and_opine_mocked(monkeypatch):
    monkeypatch.setenv("AUREON_WEB_SEARCH_ENABLED", "1")

    monkeypatch.setattr(
        "brain.web_search.search",
        lambda q, **kw: [{"text": "AI models continue rapid progress in 2026.", "source": "web"}],
    )
    monkeypatch.setattr(
        "app.chat_service._predict_with_timeout",
        lambda *a, **kw: {"answer": "Scale and alignment remain central tensions.", "abstained": False},
    )

    result = chat("What happened with AI stock today?", session_id="search-1")
    assert result["kind"] == "search_opinion"
    assert "2026" in result["reply"] or "progress" in result["reply"].lower()
    assert result.get("sources")
