"""Philosophy and personal-belief routing — classification feeds predict, never replaces it."""

from __future__ import annotations

from typing import Any, Callable

PHILOSOPHY_FALLBACK = (
    "This touches one of the deepest questions I engage with. "
    "My corpus is still growing in this domain. "
    "What I can say is that the question you are asking "
    "sits at the intersection of consciousness, existence, and meaning — "
    "domains I take seriously. Ask me again after my next learning cycle "
    "and I will have more to offer."
)

PHILOSOPHY_SIGNALS = (
    "god",
    "belief",
    "believe",
    "soul",
    "consciousness",
    "meaning",
    "existence",
    "divine",
    "spirit",
    "faith",
    "universe",
    "purpose",
    "after death",
    "afterlife",
    "creation",
    "religion",
    "metaphysics",
)

PERSONAL_BELIEF_TRIGGERS = (
    "do you believe",
    "what do you think",
    "what are your thoughts",
    "your opinion",
    "do you feel",
    "what is your view",
    "do you have faith",
    "are you spiritual",
    "what do you believe",
    "your perspective on",
    "how do you feel about",
    "do you think god",
    "do you think there is a god",
)


def is_personal_belief_question(text: str) -> bool:
    q = text.strip().lower()
    return any(t in q for t in PERSONAL_BELIEF_TRIGGERS)


def is_philosophy_question(text: str) -> bool:
    q = text.strip().lower()
    if is_personal_belief_question(text):
        return True
    return any(s in q for s in PHILOSOPHY_SIGNALS)


def philosophy_fallback_if_needed(text: str, result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result and result.get("answer") and len(str(result["answer"])) > 30:
        if not result.get("abstained"):
            return None
    q = text.lower()
    if not any(s in q for s in PHILOSOPHY_SIGNALS):
        return None
    domain = "philosophy.metaphysics.philosophy_of_religion"
    if "god" in q or "faith" in q or "believe" in q:
        domain = "philosophy.metaphysics.philosophy_of_religion"
    elif "consciousness" in q:
        domain = "philosophy.mind.consciousness"
    return {
        "reply": PHILOSOPHY_FALLBACK,
        "kind": "philosophy_fallback",
        "domain": domain,
        "simple_qa": False,
    }


def _belief_enriched_prompt(text: str, classification: dict[str, Any] | None) -> str:
    q = text.strip().lower()
    if "do you believe" in q and "god" in q:
        key = "do you believe in god"
    elif "what are your thoughts" in q and "god" in q:
        key = "what are your thoughts on god"
    elif "who is god" in q or "what is god" in q:
        key = "who is god to you" if "to you" in q else "who is god"
    else:
        key = q.rstrip("?")

    ctx = ""
    if classification:
        ctx = f"domain context {classification['label']} "
    return f"{ctx}question {key} think therefore answer"


def handle_philosophy_question(
    text: str,
    *,
    session_id: str | None,
    predict_fn: Callable[..., dict[str, Any]],
    classify_fn: Callable[[str], dict[str, Any] | None],
    learning_snapshot_fn: Callable[[], dict[str, Any]],
) -> dict[str, Any] | None:
    """Route philosophy/belief through predict — never return raw classification."""
    classification = classify_fn(text)
    enriched = _belief_enriched_prompt(text, classification)
    result = predict_fn(enriched, session_id=session_id, force=True)

    if result and result.get("answer") and not result.get("abstained"):
        reply = str(result["answer"]).strip()
        if len(reply) > 20 and "philosophy." not in reply.lower():
            payload: dict[str, Any] = {
                "reply": reply,
                "kind": "predict",
                "session_id": session_id,
                "learning": learning_snapshot_fn(),
                "brain_predict": True,
                "simple_qa": False,
                "domain": classification["label"] if classification else None,
            }
            if classification:
                payload["classification"] = classification
            if result.get("citations"):
                payload["citations"] = result["citations"][:3]
            payload["prediction"] = {k: v for k, v in result.items() if k != "error"}
            return payload

    fallback = philosophy_fallback_if_needed(text, result)
    if fallback:
        fallback["session_id"] = session_id
        fallback["learning"] = learning_snapshot_fn()
        if classification:
            fallback["classification"] = classification
        return fallback

    return None
