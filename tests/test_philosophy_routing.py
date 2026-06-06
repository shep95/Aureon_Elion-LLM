"""Philosophy and identity routing — no raw classification as reply."""

from __future__ import annotations

from app.chat_service import chat, _simple_chat_reply

_META_NOISE = (
    "learning cycle",
    "documents",
    "corpus",
    "audit trail",
    "searched outward",
    "i searched",
    "verified documents",
    "logs, weights",
    "six-region",
)


def _assert_plain_reflection(reply: str) -> None:
    lower = reply.lower()
    for marker in _META_NOISE:
        assert marker not in lower, f"reply leaked internal meta: {marker!r} in {reply!r}"

def test_god_thoughts_not_classification_label():
    result = chat("What are your thoughts on God?")
    reply = result["reply"].lower()
    assert "philosophy.metaphysics" not in reply
    assert "philosophy of religion" not in reply or "deepest" in reply
    assert len(reply) > 30


def test_what_is_god_routes_to_philosophy_not_deep_concept():
    result = chat("What is god?", session_id="god-concept-1")
    reply = result["reply"].lower()
    assert result["kind"] == "reflection"
    assert "philosophy.metaphysics" not in reply
    assert "god" in reply
    assert "god is understood as the ultimate source of existence consciousness and meaning" not in reply
    assert "deeper corpus grounding" not in reply


def test_what_is_consciousness_routes_to_philosophy():
    result = chat("What is consciousness?", session_id="consciousness-1")
    reply = result["reply"].lower()
    assert result["kind"] in ("philosophy", "philosophy_fallback", "chat")
    assert result["kind"] not in ("deep_concept", "deep_concept_thin")
    assert "consciousness" in reply or "deepest questions" in reply


def test_do_you_believe_in_god_not_classification():
    result = chat("Do you believe in God?")
    reply = result["reply"].lower()
    assert result["kind"] == "reflection"
    assert "philosophy.metaphysics" not in reply
    assert reply.startswith("i ") or reply.startswith("i'") or reply.startswith("no")
    assert any(w in reply for w in ("believe", "faith", "atheist", "open", "yes"))
    _assert_plain_reflection(result["reply"])


def test_reflection_searches_web_when_no_cycles(monkeypatch):
    monkeypatch.setenv("AUREON_WEB_SEARCH_ENABLED", "1")
    monkeypatch.setattr(
        "brain.meta_consciousness.gather_self_state",
        lambda: {
            "documents": 0,
            "verified_documents": 0,
            "cycles_completed": 0,
            "focus_path": "",
        },
    )
    monkeypatch.setattr(
        "brain.web_search.search",
        lambda q, **kw: [
            {
                "text": "Philosophers debate whether belief in God rests on faith, reason, or lived experience.",
                "source": "plato.stanford.edu",
            }
        ],
    )

    result = chat("Do you believe in God?", session_id="web-reflection-1")
    reply = result["reply"].lower()
    assert result["kind"] == "reflection"
    assert result.get("grounded_via") == "web_search"
    assert result.get("sources")
    assert any(w in reply for w in ("faith", "atheist", "open", "yes", "no"))
    assert "philosophers debate" not in reply
    _assert_plain_reflection(result["reply"])


def test_reflection_searches_web_for_consciousness_when_untrained(monkeypatch):
    monkeypatch.setenv("AUREON_WEB_SEARCH_ENABLED", "1")
    monkeypatch.setattr(
        "brain.meta_consciousness.gather_self_state",
        lambda: {
            "documents": 2,
            "verified_documents": 1,
            "cycles_completed": 0,
            "focus_path": "",
        },
    )
    monkeypatch.setattr(
        "brain.web_search.search",
        lambda q, **kw: [
            {
                "text": "The hard problem of consciousness asks why physical processes feel like anything at all.",
                "source": "iep.utm.edu",
            }
        ],
    )

    result = chat("What are your thoughts on consciousness?", session_id="web-reflection-2")
    reply = result["reply"].lower()
    assert result["kind"] == "reflection"
    assert result.get("grounded_via") == "web_search"
    assert any(w in reply for w in ("consciousness", "aware", "experience", "feel"))
    assert "hard problem" not in reply or "function" in reply
    _assert_plain_reflection(result["reply"])


def test_do_you_think_humans_are_flawed():
    result = chat("Do you think humans are flawed?", session_id="human-flaw-1")
    reply = result["reply"].lower()
    assert result["kind"] == "reflection"
    assert any(w in reply for w in ("human", "flaw", "err", "growth", "yes"))
    _assert_plain_reflection(result["reply"])


def test_thoughts_on_consciousness_doctrine():
    result = chat("What are your thoughts on consciousness?", session_id="conscious-thoughts-1")
    reply = result["reply"].lower()
    assert result["kind"] == "reflection"
    assert any(w in reply for w in ("consciousness", "aware", "experience", "feel"))
    _assert_plain_reflection(result["reply"])


def test_are_humans_flawed_doctrine():
    result = chat("Are humans flawed?", session_id="human-flaw-2")
    reply = result["reply"].lower()
    assert result["kind"] == "reflection"
    assert any(w in reply for w in ("human", "flaw", "limit", "bias"))
    assert "deeper corpus grounding" not in reply


def test_subjective_experience_doctrine():
    result = chat("Do you have subjective experience?", session_id="subjective-1")
    reply = result["reply"].lower()
    assert result["kind"] == "reflection"
    assert any(w in reply for w in ("don't know", "feel", "inside", "experience"))
    _assert_plain_reflection(result["reply"])


def test_who_are_you_not_mechanical_string():
    result = chat("Who are you?")
    reply = result["reply"]
    assert result.get("kind") == "identity"
    assert "supervised ml brain — collect" not in reply.lower()
    assert "aureon" in reply.lower()


def test_who_is_god_to_you_reflection():
    result = chat("Who is God to you?", session_id="god-to-you-1")
    reply = result["reply"].lower()
    assert result["kind"] == "reflection"
    assert "god" in reply
    assert "six-region learning brain" not in reply
    assert "god is understood as the ultimate source" not in reply


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
