"""Psychology brain — human response layer tests."""

from __future__ import annotations

from brain.psychology_brain import finalize_chat_payload, interpret_user_question, shape_human_reply


def test_curious_clarifier_keeps_question():
    reply, ctx = shape_human_reply(
        "What type of blood, blood type, plasma, red cells, the iron?",
        payload={"ciper": {"mode": "decompose"}, "kind": "chat"},
        user_message="I can move blood",
    )
    assert reply.endswith("?")
    assert ctx.mode == "curious_clarifier"
    assert "marie_ciper_logic" in ctx.traits


def test_grounded_direct_no_source_prefix():
    reply, ctx = shape_human_reply(
        "Botany is the study of plants.",
        payload={"ciper": {"mode": "answer", "grounded": True}, "kind": "chat"},
        user_message="What is botany?",
    )
    assert "from what i've collected" not in reply.lower()
    assert reply.startswith("Botany")
    assert ctx.mode == "grounded_direct"


def test_interpret_user_question_strips_social_suffix_and_maps_taxonomy():
    understanding = interpret_user_question("explain quantum mechanics to me")
    assert understanding is not None
    assert understanding.intent == "explanation"
    assert understanding.subject == "quantum mechanics"
    assert understanding.normalized_query == "explain quantum mechanics"
    assert "science_and_natural_philosophy.physics.quantum_mechanics" in understanding.taxonomy_paths
    assert "intent_subject_mapping" in understanding.traits


def test_interpret_user_question_strips_how_it_works_suffix():
    understanding = interpret_user_question("Explain Quantum Artificial intelligence to me and how it works")
    assert understanding is not None
    assert understanding.subject == "Quantum Artificial intelligence"
    assert understanding.normalized_query == "explain Quantum Artificial intelligence"


def test_finalize_adds_brains_metadata():
    out = finalize_chat_payload(
        {"reply": "Hello.", "kind": "chat", "simple_qa": True},
        "Hi",
    )
    assert "psychology" in out
    assert "algorithm" in out["brains"]
    assert "psychology" in out["brains"]
