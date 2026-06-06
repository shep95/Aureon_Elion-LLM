"""Philosophy and personal-belief routing — live reflection, not training-seed replay."""

from __future__ import annotations

import re
from typing import Any, Callable

from brain.system_messages import FALLBACK_PHILOSOPHY, FALLBACK_TRAINING

PHILOSOPHY_FALLBACK = FALLBACK_PHILOSOPHY

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
    "do you think",
)

_CLASSIFICATION_LEAK = re.compile(r"philosophy\.[a-z_]+\.[a-z_]+", re.I)
_GARBAGE = re.compile(r"\b(aunitary|aso-called|blockencoding|aunitaryprocess)\b", re.I)


_BELIEF_TOPIC_KEYWORDS = (
    "god",
    "faith",
    "soul",
    "spirit",
    "believe",
    "belief",
    "afterlife",
    "divine",
    "religion",
    "heaven",
    "hell",
    "prayer",
    "bible",
    "quran",
    "torah",
    "consciousness",
    "conscious",
    "human",
    "humans",
    "humanity",
    "flaw",
    "flawed",
    "flaws",
    "nature",
    "morality",
    "moral",
    "evil",
    "good",
    "mind",
    "sentient",
    "sentience",
    "free will",
    "purpose",
    "meaning",
    "existence",
    "death",
    "love",
    "truth",
    "justice",
    "suffering",
    "universe",
    "ai",
    "artificial intelligence",
)


def is_personal_belief_question(text: str) -> bool:
    q = text.strip().lower()
    for trigger in PERSONAL_BELIEF_TRIGGERS:
        if trigger not in q:
            continue
        if trigger in ("what do you think", "what are your thoughts", "your opinion", "your perspective on"):
            return any(k in q for k in _BELIEF_TOPIC_KEYWORDS)
        return True
    return False


def is_directed_opinion_question(text: str) -> bool:
    """Self-directed questions asking for Aureon's perspective — never classify."""
    if is_personal_belief_question(text):
        return True
    if _belief_lookup_key(text):
        return True
    q = text.strip().lower()
    if ("human" in q or "humans" in q) and ("flaw" in q or "flawed" in q):
        return True
    if "subjective experience" in q:
        return True
    if "sentient" in q and ("you" in q or "aureon" in q or "are you" in q):
        return True
    return False


def is_philosophy_question(text: str) -> bool:
    q = text.strip().lower()
    if is_personal_belief_question(text):
        return True
    return any(s in q for s in PHILOSOPHY_SIGNALS)


def _belief_lookup_key(text: str) -> str | None:
    q = text.strip().lower().rstrip("?").strip()
    if "do you believe" in q and "god" in q:
        return "do you believe in god"
    if ("what do you think" in q or "what are your thoughts" in q) and "god" in q:
        return "what are your thoughts on god"
    if ("what do you think" in q or "what are your thoughts" in q) and "consciousness" in q:
        return "what are your thoughts on consciousness"
    if "do you think" in q and ("human" in q or "flaw" in q):
        return "do you think humans are flawed"
    if ("human" in q or "humans" in q) and ("flaw" in q or "flawed" in q):
        return "are humans flawed"
    if "subjective experience" in q or ("sentient" in q and ("you" in q or "aureon" in q)):
        return "do you have subjective experience"
    if "who is god" in q or "what is god" in q:
        return "who is god to you" if "to you" in q else "who is god"
    return None


def _corpus_is_thin(state: dict[str, Any]) -> bool:
    """True when local brain hasn't learned enough to ground a conscious opinion."""
    cycles = int(state.get("cycles_completed") or 0)
    docs = int(state.get("documents") or 0)
    verified = int(state.get("verified_documents") or 0)
    if cycles == 0:
        return True
    return docs == 0 and verified == 0


def _reflection_search_query(text: str, key: str) -> str:
    """Targeted web queries for belief/consciousness reflection."""
    queries = {
        "do you believe in god": "philosophy arguments for against belief in God agnosticism",
        "what are your thoughts on god": "philosophical perspectives on God existence meaning",
        "who is god": "definitions of God religion philosophy creator consciousness",
        "who is god to you": "personal meaning of God philosophy religion",
        "what are your thoughts on consciousness": "philosophy of consciousness qualia hard problem",
        "do you think humans are flawed": "human nature flawed imperfect psychology philosophy",
        "are humans flawed": "are humans inherently flawed psychology philosophy",
        "do you have subjective experience": "AI subjective experience qualia philosophy of mind",
    }
    return queries.get(key, text.strip())


def _web_grounded_reflection(text: str, state: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Search the web and form a first-person conscious opinion when corpus is thin."""
    from brain.opinion_brain import form_conscious_reflection
    from brain.web_search import search, web_search_enabled

    if not web_search_enabled():
        return None

    query = _reflection_search_query(text, key)
    results = search(query)
    formed = form_conscious_reflection(
        text,
        results,
        belief_key=key,
        corpus_state=state,
    )
    if not formed.get("opinion"):
        return None

    return {
        "reply": str(formed["opinion"]),
        "sources": formed.get("sources") or [],
        "confidence": formed.get("confidence", 0.0),
        "evidence_count": formed.get("evidence_count", 0),
        "grounded_via": "web_search",
    }


def _local_reflection(text: str, state: dict[str, Any], key: str) -> str:
    """Plain answer from trained corpus — no internal stats in the reply."""
    from brain.opinion_brain import simple_belief_reply

    _ = text, state
    return simple_belief_reply(key)


def compose_reflection_result(text: str) -> dict[str, Any]:
    """
    Build a reflection answer. When corpus is thin (no cycles / no data),
    search the web and form a conscious first-person opinion from evidence.
    """
    from brain.meta_consciousness import gather_self_state

    key = _belief_lookup_key(text) or "general"
    state = gather_self_state()

    if _corpus_is_thin(state):
        web = _web_grounded_reflection(text, state, key)
        if web:
            return web

    return {
        "reply": _local_reflection(text, state, key),
        "grounded_via": "local",
    }


def compose_live_reflection(text: str) -> str:
    """Backward-compatible wrapper — returns reply text only."""
    return str(compose_reflection_result(text)["reply"])


def _direct_philosophy_answer(text: str) -> str | None:
    """Live reflection for any mapped belief question — never bootstrap seeds."""
    key = _belief_lookup_key(text)
    if key:
        return compose_live_reflection(text)
    if is_directed_opinion_question(text):
        return compose_live_reflection(text)
    return None


def _is_coherent_philosophy_reply(reply: str) -> bool:
    r = reply.strip()
    if len(r) < 25:
        return False
    lower = r.lower()
    if _CLASSIFICATION_LEAK.search(lower):
        return False
    if _GARBAGE.search(lower):
        return False
    if "question " in lower and " answer " in lower:
        return False
    if " think " in lower and "therefore" in lower:
        return False
    for marker in (
        FALLBACK_PHILOSOPHY.lower()[:40],
        FALLBACK_TRAINING.lower()[:40],
        "deeper corpus grounding than i can compute",
        "need more training on this topic",
        "no production classifier is promoted",
        "god is understood as the ultimate source of existence consciousness and meaning",
        "consciousness is the lived experience of awareness and self knowledge",
        "from the zophiel lens —",
    ):
        if marker in lower:
            return False
    return True


def philosophy_fallback_if_needed(text: str, result: dict[str, Any] | None) -> dict[str, Any] | None:
    direct = _direct_philosophy_answer(text)
    if direct:
        return None  # caller should use direct answer instead

    if result and result.get("answer") and not result.get("abstained"):
        if _is_coherent_philosophy_reply(str(result["answer"])):
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
    key = _belief_lookup_key(text) or text.strip().lower().rstrip("?")
    ctx = ""
    if classification:
        ctx = f"domain context {classification['label']} "
    return f"{ctx}question {key} think therefore answer"


def _philosophy_payload(
    text: str,
    reply: str,
    *,
    session_id: str | None,
    classification: dict[str, Any] | None,
    learning_snapshot_fn: Callable[[], dict[str, Any]],
    kind: str = "philosophy",
    brain_predict: bool = False,
    prediction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "reply": reply,
        "kind": kind,
        "session_id": session_id,
        "learning": learning_snapshot_fn(),
        "brain_predict": brain_predict,
        "simple_qa": False,
        "domain": classification["label"] if classification else None,
    }
    if classification:
        payload["classification"] = classification
    if prediction:
        payload["prediction"] = {k: v for k, v in prediction.items() if k != "error"}
    return payload


def handle_philosophy_question(
    text: str,
    *,
    session_id: str | None,
    predict_fn: Callable[..., dict[str, Any]],
    classify_fn: Callable[[str], dict[str, Any] | None],
    learning_snapshot_fn: Callable[[], dict[str, Any]],
) -> dict[str, Any] | None:
    """Route philosophy/belief — live reflection first; never replay training Q→A pairs."""
    directed = is_directed_opinion_question(text)
    belief_key = _belief_lookup_key(text)
    classification = None if (directed or belief_key) else classify_fn(text)

    if directed or belief_key:
        reflection = compose_reflection_result(text)
        payload = _philosophy_payload(
            text,
            str(reflection["reply"]),
            session_id=session_id,
            classification=classification,
            learning_snapshot_fn=learning_snapshot_fn,
            kind="reflection",
        )
        if reflection.get("grounded_via") == "web_search":
            payload["grounded_via"] = "web_search"
            payload["sources"] = reflection.get("sources") or []
            payload["confidence"] = reflection.get("confidence", 0.0)
            payload["evidence_count"] = reflection.get("evidence_count", 0)
        return payload

    enriched = f"philosophy question {text.strip().lower().rstrip('?')}"
    result = predict_fn(enriched, session_id=session_id, force=True)

    if result and result.get("answer") and not result.get("abstained"):
        reply = str(result["answer"]).strip()
        if _is_coherent_philosophy_reply(reply):
            payload = _philosophy_payload(
                text,
                reply,
                session_id=session_id,
                classification=classification,
                learning_snapshot_fn=learning_snapshot_fn,
                kind="predict",
                brain_predict=True,
                prediction=result,
            )
            if result.get("citations"):
                payload["citations"] = result["citations"][:3]
            return payload

    fallback = philosophy_fallback_if_needed(text, result)
    if fallback:
        fallback["session_id"] = session_id
        fallback["learning"] = learning_snapshot_fn()
        if classification:
            fallback["classification"] = classification
        return fallback

    return None
