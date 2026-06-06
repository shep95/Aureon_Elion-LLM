"""Conversational intelligence — continuation routing and human search replies."""

from __future__ import annotations

from app.chat_service import chat, is_search_question
from app.session_memory import append_turn
from brain.conversation_engine import is_continuation_message, resolve_message, update_stack_from_turn


def test_is_continuation_message():
    assert is_continuation_message("dive deeper")
    assert is_continuation_message("go deeper")
    assert is_continuation_message("tell me more")
    assert is_continuation_message("keep going")
    assert not is_continuation_message("What happened in the tech world today?")


def test_resolve_message_continuation():
    resolved = resolve_message("dive deeper", "conv-stack-1")
    assert not resolved.is_continuation  # no history yet

    append_turn("conv-stack-1", user="What happened in tech today?", assistant="Some news.")
    update_stack_from_turn(
        "conv-stack-1",
        user="What happened in tech today?",
        payload={"kind": "search_opinion", "sources": ["reuters.com"]},
    )
    resolved = resolve_message("dive deeper", "conv-stack-1")
    assert resolved.is_continuation
    assert "tech today" in resolved.resolved_text.lower()


def test_search_reply_is_human_not_robotic(monkeypatch):
    monkeypatch.setenv("AUREON_WEB_SEARCH_ENABLED", "1")
    monkeypatch.setattr(
        "brain.web_search.search",
        lambda q, **kw: [
            {
                "text": "TechNewsWorld - Google's AI Search Revamp Fuels DuckDuckGo Install Surge.",
                "source": "technewsworld.com",
            },
            {
                "text": "Reuters - SpaceX lands Google AI compute deal ahead of IPO.",
                "source": "reuters.com",
            },
        ],
    )

    result = chat("What happened in the tech world today?", session_id="human-news-1")
    assert result["kind"] == "search_opinion"
    reply = result["reply"].lower()
    assert "based on" not in reply
    assert "zophiel lens" not in reply
    assert "sources:" not in reply
    assert result.get("sources")
    assert "google" in reply or "spacex" in reply or "duckduckgo" in reply


def test_dive_deeper_continues_thread_not_taxonomy(monkeypatch):
    monkeypatch.setenv("AUREON_WEB_SEARCH_ENABLED", "1")

    calls: list[str] = []

    def mock_search(q, **kw):
        calls.append(q)
        if len(calls) == 1:
            return [
                {
                    "text": "Nvidia raises AI PC stakes with new chips.",
                    "source": "cnbc.com",
                }
            ]
        return [
            {
                "text": "Analysts say the AI PC market will double by 2027.",
                "source": "wired.com",
            }
        ]

    monkeypatch.setattr("brain.web_search.search", mock_search)

    sid = "dive-deeper-1"
    first = chat("What happened in the tech world today?", session_id=sid)
    assert first["kind"] == "search_opinion"

    second = chat("dive deeper", session_id=sid)
    assert second.get("continuation") is True
    assert "biodiversity" not in second["reply"].lower()
    assert ".environmental_science" not in second["reply"].lower()
    assert " → " not in second["reply"]
    assert second["kind"] == "search_opinion"
    assert len(calls) >= 2


def test_religion_spirituality_choice_is_reflection(monkeypatch):
    monkeypatch.setenv("AUREON_WEB_SEARCH_ENABLED", "0")
    q = (
        "if you had to choose religion or spirituality which one would you choose and what domain"
    )
    result = chat(q, session_id="belief-choice-1")
    assert result["kind"] == "reflection"
    reply = result["reply"].lower()
    assert "spiritual" in reply
    assert "religion" in reply
    assert "philosophy" in reply or "ethics" in reply
    assert "biodiversity" not in reply
    assert ".metaphysics" not in result["reply"]


def test_is_search_question_live_news():
    assert is_search_question("What happened in the tech world today?")
