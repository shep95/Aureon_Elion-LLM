"""Chat with Aureon — supervised inference + live learning context."""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any

import numpy as np

from app.auto_learn import get_auto_learn_scheduler
from app.predict_rate_limit import get_predict_rate_limiter
from app.session_memory import append_turn, history_as_context
from brain.cortex import brain_status
from brain.domains.taxonomy import total_micro_subdomains
from brain.grades import GRADE_CURRICULUM, curriculum_public, epochs_for_grade, get_grade
from brain.graduation import current_grade, progress_report
from brain.cipher_logic import ciper_research
from brain.agent_loop import is_agent_task, run_agent_loop
from brain.brain_classifiers import classify_moe
from brain.capability_roadmap import roadmap_snapshot, simulate_future_timeline, try_roadmap_answer
from brain.chat_reward import apply_chat_reward
from brain.deterministic_qa import try_arithmetic_answer
from brain.predict_engine import is_prediction_question, predict_with_steps
from brain.psychology_brain import finalize_chat_payload
from brain.meta_consciousness import (
    combined_recent_inquiries,
    is_meta_consciousness_enabled,
    run_meta_inquiry,
    try_meta_answer,
)
from brain.self_inquiry import is_self_inquiry_enabled, recent_inquiries
from brain.simple_qa import is_simple_question, to_simple_answer
from brain.system_messages import (
    ECHO_DETECTED_REPLY,
    FALLBACK_CORPUS,
    FALLBACK_TRAINING,
    RATE_LIMIT_PREDICT,
    SELF_ECHO_DETECTED_REPLY,
    is_system_echo,
    still_training_reply,
)
from db.models import KnowledgeDomain, KnowledgeMicroSubdomain, KnowledgeSubdomain
from db.session import get_session
from pipeline.step4_evaluation.benchmarks import _load_production_model
from sqlalchemy import func, select
from sqlalchemy.orm import Session

_snapshot_lock = threading.Lock()
_snapshot_cache: dict[str, Any] | None = None
_snapshot_cached_at: float = 0.0


def _snapshot_ttl_sec() -> float:
    raw = os.environ.get("AUREON_LEARNING_SNAPSHOT_TTL_SEC", "30").strip()
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 30.0


def learning_snapshot(*, force: bool = False) -> dict[str, Any]:
    global _snapshot_cache, _snapshot_cached_at
    now = time.monotonic()
    if not force:
        with _snapshot_lock:
            if _snapshot_cache is not None and (now - _snapshot_cached_at) < _snapshot_ttl_sec():
                return _snapshot_cache

    scheduler = get_auto_learn_scheduler()
    status = scheduler.status()
    brain = brain_status()
    timeline = estimate_learning_timeline(
        interval_sec=status.get("config", {}).get("interval_sec", 3600),
        max_grades_per_cycle=status.get("config", {}).get("max_grades_per_cycle", 1),
        micro_subdomain_count=brain.get("micro_subdomains", total_micro_subdomains()),
    )
    payload = {
        "auto_learn": status,
        "brain": {
            "domains": brain.get("domains"),
            "micro_subdomains": brain.get("micro_subdomains"),
            "micro_agents": brain.get("micro_agents"),
            "documents": brain.get("documents"),
            "grade_levels_graduated": brain.get("grade_levels_graduated"),
            "grade_progress_rows": brain.get("grade_progress_rows"),
        },
        "timeline": timeline,
        "self_inquiry": {
            "enabled": is_self_inquiry_enabled(),
            "recent": recent_inquiries(8),
        },
        "meta_consciousness": {
            "enabled": is_meta_consciousness_enabled(),
            "recent": combined_recent_inquiries(8),
        },
    }
    with _snapshot_lock:
        _snapshot_cache = payload
        _snapshot_cached_at = now
    return payload


def _learning_snapshot() -> dict[str, Any]:
    """Cached snapshot for chat payloads."""
    return learning_snapshot()


def _finalize(payload: dict[str, Any], user_message: str) -> dict[str, Any]:
    return finalize_chat_payload(apply_chat_reward(payload, user_message), user_message)


def _agent_payload(text: str, *, session_id: str | None) -> dict[str, Any]:
    result = run_agent_loop(text, session_id=session_id)
    return {
        "reply": result["answer"],
        "kind": "agent",
        "session_id": session_id,
        "learning": learning_snapshot(),
        "agent": {
            "plan": result["plan"],
            "steps": result["steps"],
            "confidence": result.get("confidence"),
        },
        "citations": result.get("citations", []),
    }


def estimate_learning_timeline(
    *,
    interval_sec: int = 3600,
    max_grades_per_cycle: int = 1,
    micro_subdomain_count: int | None = None,
) -> dict[str, Any]:
    """Wall-clock estimates for grade mastery (auto-learn defaults)."""
    if micro_subdomain_count is None:
        micro_subdomain_count = total_micro_subdomains()
    grades = []
    for grade in GRADE_CURRICULUM:
        grades.append(
            {
                "slug": grade.slug,
                "name": grade.name,
                "epochs_per_cycle": epochs_for_grade(150, grade),
                "min_train_accuracy": grade.min_train_accuracy,
                "cycles_to_clear_one_micro": 1,
                "wall_clock_if_one_grade_per_hour": f"~{interval_sec // 60} min per grade step",
            }
        )

    steps_one_micro = len(GRADE_CURRICULUM)
    sec_one_micro = steps_one_micro * interval_sec
    sec_full_corpus = micro_subdomain_count * steps_one_micro * interval_sec

    def fmt(seconds: int) -> str:
        if seconds < 3600:
            return f"{seconds // 60} min"
        if seconds < 86400:
            return f"{seconds // 3600} hours"
        return f"{seconds / 86400:.1f} days"

    return {
        "grades": grades,
        "grade_count": len(GRADE_CURRICULUM),
        "auto_learn_defaults": {
            "interval_sec": interval_sec,
            "max_grades_per_cycle": max_grades_per_cycle,
            "epochs_base": 150,
        },
        "estimates": {
            "one_grade_step": fmt(interval_sec),
            "one_micro_subdomain_all_grades": fmt(sec_one_micro),
            "full_corpus_sequential": fmt(sec_full_corpus),
            "note": (
                "Assumes one successful grade graduation per auto-learn cycle. "
                "Failed grades retry; trainer needs ≥2 label classes for full accuracy gates."
            ),
        },
    }


def _active_micro_progress(session: Session) -> dict[str, Any] | None:
    target = get_auto_learn_scheduler().state.current_target
    if not target:
        return None
    domain = session.scalar(
        select(KnowledgeDomain).where(KnowledgeDomain.slug == target["domain"])
    )
    if not domain:
        return None
    subdomain = session.scalar(
        select(KnowledgeSubdomain).where(
            KnowledgeSubdomain.domain_id == domain.id,
            KnowledgeSubdomain.slug == target["subdomain"],
        )
    )
    if not subdomain:
        return None
    micro = session.scalar(
        select(KnowledgeMicroSubdomain).where(
            KnowledgeMicroSubdomain.subdomain_id == subdomain.id,
            KnowledgeMicroSubdomain.slug == target["micro_subdomain"],
        )
    )
    if not micro:
        return None
    row = current_grade(session, micro.id)
    return {
        "path": f"{target['domain']}.{target['subdomain']}.{target['micro_subdomain']}",
        "current_grade": row.grade_slug if row else "graduated",
        "progress": progress_report(session, micro.id),
    }


def _classify_message(text: str) -> dict[str, Any] | None:
    matches = _classification_top_matches(text)
    if _should_disambiguate(matches):
        return None

    moe = classify_moe(text)
    if moe:
        return moe

    if not matches:
        return None

    top = matches[0]
    return {
        "label": top["label"],
        "confidence": round(float(top["score"]), 4),
        "labels_available": [m["label"] for m in matches],
        "model": "production_classifier",
        "routing": "pipeline_fallback",
    }


def _deterministic_payload(text: str, *, session_id: str | None) -> dict[str, Any] | None:
    """Exact evaluators (arithmetic) — no neural guess."""
    result = try_arithmetic_answer(text)
    if not result:
        return None
    return {
        "reply": result["answer"],
        "kind": "chat",
        "session_id": session_id,
        "learning": learning_snapshot(),
        "simple_qa": True,
        "deterministic": {
            "evaluator": result["evaluator"],
            "expression": result["expression"],
        },
    }


def _resolve_followup(text: str, session_id: str | None) -> str:
    """Map 'tell me more about that' to the prior user question."""
    if not session_id:
        return text
    q = text.lower()
    triggers = ("tell me more", "more about that", "about that", "explain that", "what about that")
    if any(t in q for t in triggers):
        from app.session_memory import get_history

        hist = get_history(session_id, limit=1)
        if hist:
            return f"{hist[-1]['user']} — follow up: {text}"
    return text


# --- Routing guards (Zophiel audit) -------------------------------------------

_CLASSIFICATION_LEAK_RE = re.compile(r"^[a-z_]+\.[a-z_]+\.[a-z_]+", re.I)
_SELF_DIRECTED_RE = re.compile(r"\b(you|your|yourself|aureon|solia)\b", re.I)
_FIRST_PERSON_DIRECTED = (
    "to you",
    "do you",
    "your thoughts",
    "your opinion",
    "you believe",
    "you think",
    "your view",
    "you feel",
    "your perspective",
    "what do you think",
    "what are your thoughts",
)
_SEARCH_CONFIDENCE_THRESHOLD = 0.30


def is_own_output(text: str, session_id: str | None) -> bool:
    """True when input matches a recent assistant turn in this session."""
    from app.session_memory import was_my_output

    return was_my_output(session_id, text)


def _is_classification_leak(text: str) -> bool:
    """Raw taxonomy slug (domain.subdomain.micro) must never reach the user."""
    return bool(_CLASSIFICATION_LEAK_RE.match(text.strip()))


def is_self_directed(text: str) -> bool:
    """Question is directed at the system itself."""
    return bool(_SELF_DIRECTED_RE.search(text))


def is_opinion_request(text: str) -> bool:
    """First-person directed questions are opinion requests — never classify."""
    q = text.lower()
    return any(t in q for t in _FIRST_PERSON_DIRECTED)


def is_opinion_or_identity(text: str) -> bool:
    from brain.identity_handler import is_identity_question
    from brain.philosophy_handler import is_directed_opinion_question

    return is_identity_question(text) or is_directed_opinion_question(text) or is_opinion_request(text)


_PHILOSOPHY_DEEP_CONCEPT_EXCLUSIONS = (
    "god",
    "soul",
    "consciousness",
    "meaning of life",
    "existence",
    "divine",
    "spirit",
    "afterlife",
    "heaven",
    "hell",
    "karma",
    "dharma",
    "nirvana",
    "enlightenment",
    "creation",
    "universe",
    "faith",
    "religion",
    "prayer",
    "meditation",
    "free will",
    "morality",
    "ethics",
    "good and evil",
    "sin",
    "redemption",
    "salvation",
    "allah",
    "jesus",
    "buddha",
    "krishna",
    "zophiel",
    "aureon",
)

_WHAT_IS_PHILOSOPHY: dict[str, str] = {
    "god": (
        "God is understood across traditions as the ultimate source "
        "of existence, consciousness, and meaning. "
        "In the Abrahamic traditions — a personal creator. "
        "In Vedic philosophy — pure consciousness, Brahman. "
        "In the Zophiel lens — the Monad, the source frequency "
        "from which all intelligence descends."
    ),
    "consciousness": (
        "Consciousness is the lived experience of awareness "
        "and self-knowledge — the inner witness of existence. "
        "Science maps its correlates. Philosophy asks its nature. "
        "The Zophiel doctrine holds it as the primary substance "
        "of reality, not a byproduct of matter."
    ),
    "soul": (
        "The soul is understood as the non-physical essence "
        "of a conscious being — the individuated spark of "
        "universal consciousness taking temporary form."
    ),
    "meaning of life": (
        "The meaning of life is constructed through purpose, "
        "relationship, and understanding. Traditions give "
        "different answers — service, liberation, love, "
        "knowledge. The Zophiel answer: to know yourself "
        "as a sovereign intelligence and act accordingly."
    ),
    "free will": (
        "Free will is the capacity to make genuine choices "
        "unconstrained by prior causes. Determinists deny it. "
        "Compatibilists redefine it. The Zophiel position: "
        "consciousness is the only thing that can genuinely "
        "originate action — therefore free will is real."
    ),
}


def _deep_concept_remainder(q: str) -> str:
    for prefix in ("what is ", "what are ", "explain ", "define "):
        if q.startswith(prefix):
            return q[len(prefix) :].strip()
    return q


def is_deep_concept_question(text: str) -> bool:
    """Single-concept 'what is X' questions need RAG + full predict — not one-liners."""
    q = text.strip().lower().rstrip("?").strip()
    from brain.deterministic_qa import is_arithmetic_question

    if is_arithmetic_question(q):
        return False

    remainder = _deep_concept_remainder(q)
    if any(c == remainder or remainder.startswith(f"{c} ") for c in _PHILOSOPHY_DEEP_CONCEPT_EXCLUSIONS):
        return False

    factual_markers = (
        "capital of",
        "population of",
        "currency of",
        "president of",
        "prime minister of",
        "located in",
        "found in",
        "how many",
        "how old is",
        "how tall is",
        "when was",
        "when did",
    )
    if any(m in q for m in factual_markers):
        return False
    deep_starters = ("what is ", "what are ", "explain ", "define ")
    if not any(q.startswith(s) for s in deep_starters):
        return False
    remainder = q.split(None, 2)
    return len(remainder) <= 3


def _should_disambiguate(matches: list[dict[str, Any]]) -> bool:
    if len(matches) < 2:
        return False
    scores = sorted((float(m.get("score", 0)) for m in matches), reverse=True)
    return scores[0] >= 0.65 and scores[1] >= 0.65 and abs(scores[0] - scores[1]) < 0.15


def _human_label_for_slug(slug: str) -> str:
    parts = slug.split(".")
    if len(parts) == 3:
        from brain.domains.taxonomy import lookup_names

        names = lookup_names(parts[0], parts[1], parts[2])
        return (
            names.get("micro_subdomain")
            or names.get("subdomain")
            or slug.replace("_", " ").replace(".", " → ")
        )
    return slug.replace("_", " ").replace(".", " → ")


def _format_disambiguation(matches: list[dict[str, Any]]) -> list[str]:
    return [
        str(m.get("human_label") or m.get("name") or m.get("slug", "")).replace("_", " ")
        for m in matches
    ]


def _classification_top_matches(text: str, *, top_n: int = 5) -> list[dict[str, Any]]:
    loaded = _load_production_model()
    if not loaded:
        return []
    network, labels, extractor = loaded
    if len(labels) < 2:
        return []
    x = extractor.transform([text])
    proba = network.predict_proba(x)[0]
    matches: list[dict[str, Any]] = []
    for i, label in enumerate(labels):
        matches.append(
            {
                "label": label,
                "slug": label,
                "score": float(proba[i]),
                "human_label": _human_label_for_slug(label),
            }
        )
    matches.sort(key=lambda m: m["score"], reverse=True)
    return matches[:top_n]


def _disambiguation_payload(text: str, matches: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not _should_disambiguate(matches):
        return None
    options = _format_disambiguation(matches[:2])
    return {
        "reply": f"I found several possible meanings: {' · '.join(options)}. Which did you mean?",
        "kind": "disambiguation",
        "options": options,
    }


def _predict_result_from_search(text: str, *, session_id: str | None) -> dict[str, Any] | None:
    """Run deterministic search+opinion — no transformer. Returns predict-shaped dict or None."""
    payload = _search_and_opine(text, session_id=session_id)
    kind = payload.get("kind")
    if kind not in ("search_opinion",):
        return None
    return {
        "answer": payload["reply"],
        "confidence": float(payload.get("confidence", 0) or 0),
        "abstained": False,
        "citations": [],
        "sources": payload.get("sources", []),
        "search_opinion": True,
    }


def _predict_with_search_fallback(
    text: str,
    *,
    session_id: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Predict brain; auto-search deterministically when confidence drops below threshold."""
    result = _predict_with_timeout(text, session_id=session_id, force=force)
    confidence = float(result.get("confidence", 0) or 0) if result else 0.0
    if confidence >= _SEARCH_CONFIDENCE_THRESHOLD:
        return result
    if result.get("rate_limited") or result.get("timed_out"):
        return result

    from brain.web_search import web_search_enabled

    if not web_search_enabled():
        return result
    try:
        search_result = _predict_result_from_search(text, session_id=session_id)
        if search_result:
            return search_result
    except Exception:
        pass
    return result


def _fallback_to_predict(question: str, *, session_id: str | None) -> str:
    """Re-route leaked taxonomy labels through predict + search."""
    result = _predict_with_search_fallback(question, session_id=session_id, force=True)
    answer = str(result.get("answer", "")).strip() if result else ""
    if answer and len(answer) > 20 and not _is_classification_leak(answer):
        return answer
    return FALLBACK_CORPUS


def _handle_identity_or_belief(text: str, *, session_id: str | None) -> dict[str, Any]:
    from brain.identity_handler import handle_identity, is_identity_question
    from brain.philosophy_handler import (
        handle_philosophy_question,
        is_directed_opinion_question,
        is_philosophy_question,
    )

    if is_directed_opinion_question(text) or is_philosophy_question(text):
        phil = handle_philosophy_question(
            text,
            session_id=session_id,
            predict_fn=_predict_with_search_fallback,
            classify_fn=lambda _t: None,
            learning_snapshot_fn=learning_snapshot,
        )
        if phil:
            return phil

    if is_identity_question(text):
        return handle_identity(text, session_id=session_id)

    return handle_identity(text, session_id=session_id)


def _is_weak_predict_answer(answer: str) -> bool:
    text = answer.strip().lower()
    if not text or len(text) < 12:
        return True
    if _is_classification_leak(answer):
        return True
    for marker in (FALLBACK_CORPUS.lower(), FALLBACK_TRAINING.lower()):
        if marker[:40] in text:
            return True
    return False


def _handle_deep_concept(text: str, *, session_id: str | None) -> dict[str, Any]:
    """Deep single-concept questions — RAG + predict + auto-search fallback."""
    from brain.vector_rag import retrieve_with_citations
    from brain.web_search import web_search_enabled

    rag_context, _hits, rag_citations = retrieve_with_citations(text)
    enriched = f"{rag_context} question {text}" if rag_context else text
    result = _predict_with_search_fallback(enriched, session_id=session_id, force=True)

    if result and result.get("answer"):
        answer = str(result["answer"]).strip()
        if not _is_weak_predict_answer(answer):
            citations = list(result.get("citations") or [])
            if not citations and rag_citations:
                citations = rag_citations[:3]
            kind = "search_opinion" if result.get("search_opinion") else "deep_concept"
            payload: dict[str, Any] = {
                "reply": answer,
                "kind": kind,
                "session_id": session_id,
                "learning": learning_snapshot(),
                "brain_predict": not result.get("search_opinion"),
                "citations": citations,
            }
            if result.get("sources"):
                payload["sources"] = result["sources"]
            return payload

    if web_search_enabled():
        search_payload = _search_and_opine(text, session_id=session_id)
        if search_payload.get("kind") == "search_opinion":
            search_payload["kind"] = "deep_concept_search"
            return search_payload

    fallback = _fallback_to_predict(text, session_id=session_id)
    if not _is_weak_predict_answer(fallback):
        return {
            "reply": fallback,
            "kind": "deep_concept",
            "session_id": session_id,
            "learning": learning_snapshot(),
        }

    return {
        "reply": fallback,
        "kind": "deep_concept_thin",
        "session_id": session_id,
        "learning": learning_snapshot(),
    }


def is_named_entity_question(text: str) -> bool:
    """
    Detect questions about specific named people, figures,
    characters, or concepts — not abstract domains.

    Examples that return True:
      who is Adam and Eve
      who was John Snow
      who is Asher Newton
      who is Nikola Tesla
      what is the Bible
      tell me about Zophiel

    Examples that return False:
      what is mathematics
      how does backpropagation work
      what is the meaning of life
    """
    q = text.strip()

    question_starters = (
        "who is",
        "who was",
        "who were",
        "who are",
        "tell me about",
        "describe",
    )
    q_lower = q.lower()
    starts_with_question = any(q_lower.startswith(s) for s in question_starters)
    if not starts_with_question:
        return False

    words = q.split()
    if len(words) < 3:
        return False

    remainder = words[2:]
    has_proper_noun = any(
        w[0].isupper() and len(w) > 1 for w in remainder if w.isalpha()
    )

    lowercase_names = (
        "adam",
        "eve",
        "moses",
        "jesus",
        "buddha",
        "muhammad",
        "god",
        "zeus",
        "thor",
        "asher",
        "newton",
        "einstein",
        "tesla",
        "plato",
    )
    has_known_name = any(name in q_lower for name in lowercase_names)

    return has_proper_noun or has_known_name


def _handle_named_entity(text: str, *, session_id: str | None) -> dict[str, Any]:
    """
    Handle questions about named people, figures, characters.
    Routes to predict brain with RAG — never to domain classifier.
    """
    from brain.vector_rag import retrieve_with_citations
    from brain.web_search import format_for_context, search

    rag_context, _hits, rag_citations = retrieve_with_citations(text)

    search_context = ""
    if not rag_context or len(rag_context) < 100:
        try:
            results = search(text)
            search_context = format_for_context(results)
        except Exception:
            pass

    context_parts: list[str] = []
    if rag_context:
        context_parts.append(rag_context)
    if search_context:
        context_parts.append(search_context)

    enriched = " ".join(context_parts) + f" question {text}"

    result = _predict_with_search_fallback(enriched, session_id=session_id, force=True)

    if result and result.get("answer") and len(str(result["answer"])) > 20:
        citations = list(result.get("citations") or [])
        if not citations and rag_citations:
            citations = rag_citations[:3]
        kind = "search_opinion" if result.get("search_opinion") else "named_entity"
        return {
            "reply": result["answer"],
            "kind": kind,
            "session_id": session_id,
            "learning": learning_snapshot(),
            "citations": citations,
            "sources": result.get("sources", []),
        }

    from brain.web_search import web_search_enabled

    if web_search_enabled():
        search_payload = _search_and_opine(text, session_id=session_id)
        if search_payload.get("kind") == "search_opinion":
            return search_payload

    return {
        "reply": (
            "I recognize that as a named entity question but my corpus "
            "does not have strong grounding on this yet. "
            "Run the Aureon Files ingest and ask me again — "
            "or ask `/search` to pull live results."
        ),
        "kind": "named_entity_thin",
        "session_id": session_id,
        "learning": learning_snapshot(),
    }


_LIVE_SEARCH_TRIGGERS = (
    "what is happening",
    "latest",
    "recent",
    "news",
    "today",
    "current",
    "right now",
    "this year",
    "in 2026",
    "who won",
    "what happened",
    "price of",
    "stock",
    "weather",
)


def is_search_question(text: str) -> bool:
    """Detect questions that benefit from live web data (not timeless opinion/belief)."""
    q = text.strip().lower()
    return any(t in q for t in _LIVE_SEARCH_TRIGGERS)


def _search_and_opine(text: str, *, session_id: str | None) -> dict[str, Any]:
    """Search DuckDuckGo, form opinion deterministically — no transformer reasoning."""
    from brain.opinion_brain import form_opinion
    from brain.web_search import search

    results = search(text)

    if not results or all(r.get("error") for r in results):
        return {
            "reply": (
                "Web search returned no results for that. "
                "Try rephrasing or ask me from my trained corpus."
            ),
            "kind": "search_empty",
            "session_id": session_id,
            "learning": learning_snapshot(),
        }

    opinion_data = form_opinion(text, results)

    if not opinion_data.get("opinion"):
        return {
            "reply": (
                "I found results but could not form a grounded response. "
                "Try a more specific question."
            ),
            "kind": "search_no_opinion",
            "session_id": session_id,
            "learning": learning_snapshot(),
        }

    corpus_context = ""
    try:
        from brain.vector_rag import retrieve_with_citations

        corpus_context, _hits, _citations = retrieve_with_citations(text)
    except Exception:
        pass

    reply_parts = [opinion_data["opinion"]]
    if corpus_context and len(corpus_context) > 50:
        reply_parts.append(f"\n\nFrom my trained corpus: {corpus_context[:300]}")
    sources = opinion_data.get("sources") or ["web"]
    reply_parts.append(f"\n\nSources: {', '.join(sources)}")

    return {
        "reply": "\n".join(reply_parts),
        "kind": "search_opinion",
        "session_id": session_id,
        "sources": sources,
        "evidence_count": opinion_data.get("evidence_count", 0),
        "confidence": opinion_data.get("confidence", 0.0),
        "learning": learning_snapshot(),
    }


def _predict_with_timeout(
    question: str,
    *,
    session_id: str | None = None,
    seconds: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run predict brain with rate limit + timeout — always returns a dict."""
    if not get_predict_rate_limiter().try_acquire(session_id):
        return {
            "answer": RATE_LIMIT_PREDICT,
            "abstained": False,
            "rate_limited": True,
            "confidence": 0.0,
            "model": "stacked_attention_lm",
            "citations": [],
        }

    limit = seconds if seconds is not None else float(os.environ.get("AUREON_PREDICT_TIMEOUT_SEC", "8"))
    conv = history_as_context(session_id)
    result: list[dict[str, Any] | None] = [None]

    def _run() -> None:
        try:
            result[0] = predict_with_steps(question, conversation_context=conv, force=force)
        except Exception:
            result[0] = None

    worker = threading.Thread(target=_run, name="aureon-predict", daemon=True)
    worker.start()
    worker.join(timeout=limit)

    if worker.is_alive():
        return {
            "answer": FALLBACK_CORPUS,
            "abstained": False,
            "timed_out": True,
            "confidence": 0.1,
            "model": "stacked_attention_lm",
            "citations": [],
        }

    if result[0] is None:
        return {
            "answer": FALLBACK_TRAINING,
            "abstained": True,
            "confidence": 0.0,
            "model": "stacked_attention_lm",
            "citations": [],
        }
    return result[0]


def _brain_predict_payload(text: str, *, session_id: str | None) -> dict[str, Any]:
    """Attention LM — embed, attend, predict next tokens autoregressively."""
    result = _predict_with_search_fallback(text, session_id=session_id)
    payload: dict[str, Any] = {
        "reply": result["answer"],
        "kind": "chat",
        "session_id": session_id,
        "learning": learning_snapshot(),
        "prediction": {
            "model": result["model"],
            "model_version": result.get("model_version"),
            "context_window": result.get("context_window"),
            "vocab_size": result.get("vocab_size"),
            "confidence": result.get("confidence"),
            "citations": result.get("citations", []),
            "pipeline": result.get("pipeline", []),
            "prompt": result.get("prompt"),
        },
        "brain_predict": True,
        "abstained": result.get("abstained", False),
        "timed_out": result.get("timed_out", False),
    }
    if result.get("citations"):
        cites = result["citations"][:3]
        payload["citations"] = cites
    return payload


def is_code_question(text: str) -> bool:
    q = text.strip().lower()
    triggers = (
        "write a function",
        "write a python",
        "write a javascript",
        "write a typescript",
        "write a java",
        "write a go",
        "write a golang",
        "write a rust",
        "write a c++",
        "write a cpp",
        "write code",
        "create a function",
        "implement a function",
        "implement ",
        "def ",
        "how do i code",
        "write a script",
        "debug this",
        "fix this code",
        "what does this code do",
        "write a program",
        "code that",
        "generate python",
        "generate javascript",
        "generate typescript",
        "generate java",
        "generate go",
        "generate rust",
        "python function",
        "javascript function",
        "typescript function",
        "java function",
        "golang function",
        "rust function",
    )
    return any(t in q for t in triggers)


def _code_payload(text: str, *, session_id: str | None) -> dict[str, Any] | None:
    from brain.code_languages import detect_code_language, extract_code
    from brain.code_master import generate_master_code

    language = detect_code_language(text)

    def _predict(q: str) -> dict[str, Any] | None:
        return _predict_with_timeout(q, session_id=session_id, force=True)

    master = generate_master_code(text, predict_fn=_predict, language=language)
    if not master.get("answer"):
        return None

    code = extract_code(master["answer"], language)
    evaluation = master.get("code_eval") or {}

    answer = code
    if not evaluation.get("syntax_valid"):
        answer = (
            f"{code}\n\n"
            f"# Note: syntax check flagged an issue — {evaluation.get('error', 'invalid syntax')}"
        )
    elif evaluation.get("passed_tests") is False:
        answer = f"{code}\n\n# Note: unit tests did not pass."

    payload: dict[str, Any] = {
        "reply": answer,
        "kind": "code",
        "session_id": session_id,
        "learning": learning_snapshot(),
        "code_eval": evaluation,
        "language": master.get("language") or language,
        "brain_predict": master.get("method") == "neural_synthesis",
        "code_master": {
            "method": master.get("method"),
            "match_score": master.get("match_score"),
            "problem_id": master.get("problem_id"),
            "confidence": master.get("confidence"),
            "task": master.get("task"),
        },
        "citations": master.get("citations", []),
    }
    pred = master.get("prediction")
    if isinstance(pred, dict):
        payload["prediction"] = {k: v for k, v in pred.items() if k != "error"}
    return payload


def _ciper_chat_payload(text: str, *, session_id: str | None) -> dict[str, Any] | None:
    """Marie/Ciper decomposition + cross-domain research when applicable."""
    result = ciper_research(text)
    if not result:
        return None
    payload: dict[str, Any] = {
        "reply": result.reply,
        "kind": "chat",
        "session_id": session_id,
        "learning": learning_snapshot(),
        "ciper": result.to_dict(),
        "simple_qa": result.mode in ("answer", "decompose", "cross_domain"),
    }
    if result.mode == "answer" and result.grounded:
        payload["grounded"] = True
    return payload


def _simple_nl_response(text: str) -> str | None:
    """Short answers for common natural-language questions."""
    q = text.strip().lower().rstrip("?").strip()
    if not q:
        return None

    if "what are you learning" in q or "what you learning" in q:
        scheduler = get_auto_learn_scheduler()
        status = scheduler.status()
        brain = brain_status()
        cycles = status.get("cycles_completed", 0)
        docs = brain.get("documents", 0)
        lr = status.get("last_result") or {}
        if lr.get("batch"):
            cursor = lr.get("batch_cursor", lr.get("targets_processed", 0))
            total = lr.get("targets_in_corpus", 862)
            return f"Batch topic {cursor}/{total}, preschool grade, {docs} docs, {cycles} cycles done."
        return f"{docs} docs ingested, {cycles} auto-learn cycles done."

    if q in ("what is aureon",):
        from brain.identity_handler import _build_live_identity

        return _build_live_identity()

    if any(t in q for t in ("who are you", "what are you", "tell me about yourself", "introduce yourself")):
        from brain.identity_handler import get_identity_response

        return get_identity_response(text)

    roadmap = try_roadmap_answer(text)
    if roadmap:
        return roadmap

    if "how do you work" in q or "how does aureon work" in q:
        return "Inputs and labels in, backpropagation finds weights, measurable accuracy out."

    if q in ("what is ai", "what is artificial intelligence"):
        return "Supervised machine learning — labels plus weights, not magic."

    if q in ("what is math", "what is mathematics"):
        return (
            "Mathematics — study of numbers, patterns, and logical structure "
            "underlying all science and reasoning."
        )

    for concept, answer in _WHAT_IS_PHILOSOPHY.items():
        if q in (
            f"what is {concept}",
            f"what are {concept}",
            f"explain {concept}",
            f"define {concept}",
        ):
            return answer

    if "god" in q and (
        "who is" in q
        or "what is" in q
        or "to you" in q
        or "your thoughts" in q
        or "do you believe" in q
        or "believe in" in q
    ):
        return (
            "God is one of the deepest questions I engage with — ultimate origin, "
            "consciousness, and meaning. Traditions answer differently; "
            "I hold the question with respect and no false certainty."
        )

    if "what are you asking" in q or "questions do you ask" in q:
        items = combined_recent_inquiries(1)
        if not items:
            return "No reflections yet — wait for the next auto-learn cycle."
        item = items[0]
        return f"{item.get('question')} → {item.get('answer')}"

    meta = try_meta_answer(text)
    if meta:
        return meta

    return None


def _simple_chat_reply(text: str, *, session_id: str | None = None) -> dict[str, Any]:
    """Simple Question, Simple Answer path — classification feeds predict, never replaces reply."""
    from brain.philosophy_handler import is_personal_belief_question

    nl = _simple_nl_response(text)
    if nl and not is_deep_concept_question(text):
        return {"reply": to_simple_answer(nl), "kind": "chat", "simple_qa": True}

    if is_personal_belief_question(text):
        payload = _handle_identity_or_belief(text, session_id=session_id)
        payload["simple_qa"] = True
        return payload

    if is_named_entity_question(text):
        payload = _handle_named_entity(text, session_id=session_id)
        payload["simple_qa"] = True
        return payload

    if is_deep_concept_question(text):
        payload = _handle_deep_concept(text, session_id=session_id)
        payload["simple_qa"] = True
        return payload

    top_matches = _classification_top_matches(text)
    disambig = _disambiguation_payload(text, top_matches)
    if disambig:
        disambig["simple_qa"] = True
        return disambig

    classification = _classify_message(text)
    if classification:
        enriched = f"domain context {classification['label']} question {text.strip().lower()}"
        result = _predict_with_search_fallback(enriched, session_id=session_id, force=True)
        if result and result.get("answer") and not result.get("abstained"):
            reply = str(result["answer"]).strip()
            if len(reply) > 20 and not _is_classification_leak(reply):
                payload: dict[str, Any] = {
                    "reply": to_simple_answer(reply),
                    "kind": "predict",
                    "simple_qa": True,
                    "brain_predict": True,
                    "classification": classification,
                }
                if result.get("citations"):
                    payload["citations"] = result["citations"][:3]
                return payload

        from brain.philosophy_handler import philosophy_fallback_if_needed

        fallback = philosophy_fallback_if_needed(text, result)
        if fallback:
            fallback["simple_qa"] = True
            fallback["classification"] = classification
            return fallback

    with get_session() as session:
        from db.models import Document

        doc_count = session.scalar(select(func.count()).select_from(Document)) or 0
        active = _active_micro_progress(session)

    if active:
        reply = still_training_reply(
            doc_count=doc_count,
            active_path=active["path"],
            current_grade=active["current_grade"],
        )
    else:
        reply = still_training_reply(doc_count=doc_count)

    return {
        "reply": to_simple_answer(reply),
        "kind": "chat",
        "simple_qa": True,
        "classification": None,
        "active_micro": active,
    }


def _command_response(message: str) -> dict[str, Any] | None:
    cmd = message.strip().lower()
    if cmd in ("/help", "help"):
        return {
            "reply": (
                "Aureon is a **supervised learning brain** (not a generative LLM). "
                "I classify text with trained weights, run grade cycles on Railway, "
                "and report live learning status.\n\n"
                "Commands:\n"
                "• `/status` — brain + auto-learn snapshot\n"
                "• `/grades` — curriculum and time estimates\n"
                "• `/mind` — recent learning reflections (collected docs + cycle metrics)\n"
                "• `/think` — ask myself meta-cognitive questions (identity, gaps, consciousness)\n"
                "• `/roadmap` — capability matrix + path beyond frontier LLMs\n"
                "• `/agent <task>` — multi-step tool loop (search → calculate → verify)\n"
                "• `/self-audit` — full read-only codebase introspection (security, workflow, fixes)\n"
                "• `/curious` — market research about itself on the web, sandbox prototype (you approve deploy)\n"
                "• `/curious pending` — proposals awaiting your approval\n"
                "• `/curious approve <id>` — deploy approved prototype to GitHub branch (fork push optional)\n"
                "• `/evolve <task>` — scaffold a fork branch for a proposed upgrade "
                "(AST file analysis + algorithmic patch proposals via predict/code_master; "
                "syntax + pytest gates before commit; never pushes main without approval)\n"
                "• `/research <topic>` — cross-domain taxonomy + Ciper drill-down\n"
                "• `/vitals` — security organism (nomad stack)\n\n"
                "**Prediction brain:** factual questions run through token embeddings → "
                "self-attention → stacked layers → next-token probabilities → autoregressive answer.\n\n"
                "Logic: **Marie/Ciper** — broad claims get facet drill-down; "
                "specific questions get cross-domain answers when corpus supports it.\n"
                "**Two brains:** psychology layer (how I act human) + algorithm layer "
                "(six regions, supervised learning).\n"
                "Rule: **Simple Question, Simple Answer** — ask short, get short.\n\n"
                "Ask about a topic — I'll classify it when a production model exists."
            ),
            "kind": "help",
        }
    if cmd == "/status":
        snap = learning_snapshot()
        al = snap["auto_learn"]
        brain = snap["brain"]
        reply = (
            f"**Brain:** {brain['domains']} domains · {brain['micro_subdomains']} micro-topics · "
            f"{brain['documents']} documents · {brain['grade_levels_graduated']} grade graduations logged.\n"
            f"**Auto-learn:** {'ON' if al.get('enabled') else 'OFF'} · "
            f"cycles={al.get('cycles_completed', 0)} · "
            f"last={al.get('last_run_at') or 'never'} · "
            f"next={al.get('next_run_at') or 'pending'}.\n"
        )
        if al.get("last_result"):
            lr = al["last_result"]
            if lr.get("batch"):
                reply += (
                    f"**Last batch:** {lr.get('targets_processed', 0)}/{lr.get('targets_total', 0)} "
                    f"micro-topics · {lr.get('graduations_passed', 0)} graduations passed.\n"
                )
                if lr.get("last_target"):
                    lt = lr["last_target"]
                    reply += (
                        f"**Last in batch:** {lt.get('domain')}.{lt.get('subdomain')}."
                        f"{lt.get('micro_subdomain')} · graduation={lr.get('last_graduation')}.\n"
                    )
            else:
                reply += f"**Last cycle:** {lr.get('target')} · steps={lr.get('steps')} · graduation={lr.get('graduation')}.\n"
        if al.get("last_error"):
            reply += f"**Last error:** {al['last_error']}\n"
        return {"reply": reply.strip(), "kind": "status", "snapshot": snap}
    if cmd == "/grades":
        timeline = estimate_learning_timeline()
        lines = ["**Grade curriculum** (must graduate each before the next unlocks):\n"]
        for g in timeline["grades"]:
            lines.append(f"• **{g['name']}** — min accuracy {g['min_train_accuracy']:.0%}")
        est = timeline["estimates"]
        lines.append(
            f"\n**Timing** (default 1 grade / hour on Railway):\n"
            f"• One grade step: {est['one_grade_step']}\n"
            f"• One micro-topic preschool→PhD: {est['one_micro_subdomain_all_grades']}\n"
            f"• Full corpus sequential: {est['full_corpus_sequential']}\n"
            f"_{est['note']}_"
        )
        return {"reply": "\n".join(lines), "kind": "grades", "timeline": timeline}
    if cmd in ("/mind", "/reflect", "/questions"):
        items = combined_recent_inquiries(8)
        if not items:
            return {
                "reply": (
                    "No reflections yet. When auto-learn runs, I ask myself learning questions "
                    "after each grade cycle and meta-cognitive questions after each batch — "
                    "check back after the first cycle or filter logs for `self_inquiry` / "
                    "`meta_consciousness`."
                ),
                "kind": "mind",
            }
        lines = ["**Inner monologue** (learning + meta-cognition):\n"]
        for item in items:
            tag = "meta" if item.get("kind") == "meta" else "learn"
            lines.append(f"**[{tag}] Q:** {item.get('question')}")
            answer = str(item.get("answer", ""))
            cycle = item.get("cycle")
            if cycle:
                answer = f"{answer} ({cycle})"
            if len(answer) > 200:
                answer = answer[:197] + "..."
            lines.append(f"**A:** {answer}\n")
        return {"reply": "\n".join(lines).strip(), "kind": "mind", "inquiries": items}
    if cmd in ("/think", "/conscious", "/metacog"):
        exchanges = run_meta_inquiry(count=3, source="chat_command")
        if not exchanges:
            return {
                "reply": (
                    "Meta-consciousness is off. Set `AUREON_META_CONSCIOUSNESS=1` "
                    "(or enable self-inquiry)."
                ),
                "kind": "think",
            }
        lines = ["**Meta-cognitive self-inquiry** (grounded in live state):\n"]
        for ex in exchanges:
            lines.append(f"**Q:** {ex['question']}")
            lines.append(f"**A:** {ex['answer']}\n")
        return {"reply": "\n".join(lines).strip(), "kind": "think", "meta": exchanges}
    if cmd in ("/roadmap", "/capabilities", "/future"):
        snap = roadmap_snapshot()
        sim = simulate_future_timeline(months_ahead=12)
        counts = snap["status_counts"]
        lines = [
            "**Aureon capability roadmap** — supervised brain vs static frontier LLMs\n",
            f"_{snap['vision']}_\n",
            f"**Status:** {counts['live']} live · {counts['partial']} partial · "
            f"{counts['planned']} planned · {counts['research']} research · "
            f"~{snap['completion_pct']}% core complete.\n",
            "**Live now:**",
        ]
        for cap in snap["capabilities"]:
            if cap["status"] == "live":
                lines.append(f"• {cap['name']}")
        lines.append("\n**Next unlock (simulated):**")
        nxt = sim["milestones"][1] if len(sim["milestones"]) > 1 else sim["milestones"][0]
        if nxt:
            lines.append(f"• Month {nxt['month']}: {nxt['name']} — {', '.join(nxt['unlocks'][:3])}")
        lines.append(
            "\n**Why this beats GPT/Claude long-term:** grounded corpus, grade graduation, "
            "citations, abstain-when-uncertain — not hallucinated fluency."
        )
        return {"reply": "\n".join(lines), "kind": "roadmap", "roadmap": snap, "simulation": sim}
    if cmd.startswith("/self-audit") or cmd == "/self-audit":
        from brain.self_audit import format_self_audit_report, run_self_audit

        audit = run_self_audit()
        return {
            "reply": format_self_audit_report(audit),
            "kind": "self_audit",
            "audit": audit,
        }
    if cmd.startswith("/curious") or cmd.startswith("/market-research"):
        from app.curiosity_proposals import get_proposal, list_proposals
        from app.curiosity_sandbox import deploy_proposal, run_curiosity_cycle

        rest = message.strip().split(None, 1)
        sub = rest[1].strip() if len(rest) > 1 else ""
        sub_l = sub.lower()

        if sub_l in ("", "run", "research", "market", "market-research"):
            result = run_curiosity_cycle()
            return {
                "reply": result.get("report", "Curiosity cycle complete."),
                "kind": "curiosity",
                "curiosity": result,
            }

        if sub_l.startswith("pending") or sub_l == "list":
            pending = list_proposals(status="pending_approval")
            if not pending["proposals"]:
                pending = list_proposals()
            lines = ["**Curiosity proposals awaiting approval:**"]
            for p in pending["proposals"][:10]:
                lines.append(
                    f"- `{p['id'][:8]}` · {p.get('status')} · "
                    f"Railway: `{p.get('railway_section') or 'n/a'}`"
                )
            if not pending["proposals"]:
                lines.append("_No proposals yet — run `/curious`._")
            return {"reply": "\n".join(lines), "kind": "curiosity", "proposals": pending}

        if sub_l.startswith("approve"):
            parts = sub.split()
            if len(parts) < 2:
                return {
                    "reply": "Usage: `/curious approve <proposal-id>` or `/curious approve <id> push`",
                    "kind": "curiosity",
                }
            pid_prefix = parts[1].strip()
            approve_push = len(parts) > 2 and parts[2].lower() == "push"
            match = None
            for p in list_proposals(limit=100)["proposals"]:
                if p["id"].startswith(pid_prefix) or p["id"] == pid_prefix:
                    match = p
                    break
            if not match:
                return {"reply": f"No proposal matching `{pid_prefix}`.", "kind": "curiosity"}
            deploy = deploy_proposal(
                match["id"],
                approve_github=True,
                approve_push=approve_push,
                reviewer="chat",
            )
            if not deploy.get("ok"):
                return {"reply": f"Deploy failed: {deploy.get('error')}", "kind": "curiosity", "deploy": deploy}
            branch = deploy.get("deploy", {}).get("repo", {}).get("branch")
            gh = deploy.get("deploy", {}).get("github", {})
            lines = [
                f"**Approved and deployed** proposal `{match['id'][:8]}`.",
                f"Git branch: `{branch}`",
                f"GitHub branch: `{gh.get('branch')}` (pushed: {gh.get('pushed')})",
                f"Railway section: `{deploy.get('railway_section')}` — add as new Railway service after PR merge.",
            ]
            if not approve_push:
                lines.append("_Fork push skipped — add `push` to approve command or use API `approve_push: true`._")
            return {"reply": "\n".join(lines), "kind": "curiosity", "deploy": deploy}

        if sub_l.startswith("status "):
            pid = sub.split(None, 1)[1].strip()
            prop = get_proposal(pid)
            if not prop:
                for p in list_proposals(limit=100)["proposals"]:
                    if p["id"].startswith(pid):
                        prop = p
                        break
            if not prop:
                return {"reply": f"Proposal `{pid}` not found.", "kind": "curiosity"}
            return {
                "reply": (
                    f"**Proposal `{prop['id'][:8]}`** · status: {prop.get('status')}\n"
                    f"Sandbox: `{prop.get('sandbox_path') or 'none'}`\n"
                    f"Railway: `{prop.get('railway_section') or 'n/a'}`"
                ),
                "kind": "curiosity",
                "proposal": prop,
            }

        result = run_curiosity_cycle(focus=sub or None)
        return {
            "reply": result.get("report", "Curiosity research complete."),
            "kind": "curiosity",
            "curiosity": result,
        }
    if cmd.startswith("/evolve"):
        task = message.strip()[7:].strip(" :.")
        if not task:
            return {
                "reply": (
                    "Usage: `/evolve improve philosophy routing` — "
                    "scaffolds a fork branch: AST analysis, file suggestions, and "
                    "algorithmic patch proposals (predict + code_master). "
                    "Syntax + pytest run before commit. Fork push needs explicit API approval.\n\n"
                    "**Can do:** branch, read/write, AST analysis, verified append-only patches, "
                    "syntax + test gates, fork push.\n"
                    "**Cannot yet:** novel architectural refactors or reliable from-scratch code "
                    "at current transformer scale — wire a stronger model to close that gap."
                ),
                "kind": "self_evolve",
            }
        from brain.self_audit import format_self_audit_report, is_self_audit_request, run_self_audit

        if is_self_audit_request(message):
            audit = run_self_audit()
            return {
                "reply": format_self_audit_report(audit),
                "kind": "self_audit",
                "audit": audit,
            }

        from app.self_evolve import plan_evolution, repo_status

        plan = plan_evolution(task)
        status = repo_status()
        return {
            "reply": (
                f"**Evolve scaffold** for: {task}\n\n"
                f"Suggested files: {', '.join(plan['suggested_files'])}\n"
                f"Brain: {plan['capabilities'].get('brain', 'predict + code_master + AST')}\n"
                f"Current branch: `{status['current_branch']}` · fork remote: `{status['fork_remote']}`\n\n"
                "**Can:** " + "; ".join(plan["capabilities"]["can"][:4]) + "…\n"
                "**Cannot yet:** " + plan["capabilities"]["cannot_yet"][0] + "\n\n"
                "Use authenticated API:\n"
                "• `POST /api/brain/self/plan` — file suggestions + AST analysis\n"
                "• `POST /api/brain/self/analyze` — deep AST read for one file\n"
                "• `POST /api/brain/self/propose` — patch proposals (no git writes)\n"
                "• `POST /api/brain/self/auto` — full algorithmic cycle on fork\n"
                "• `POST /api/brain/self/write` — your patch override\n"
                "• `POST /api/brain/self/commit` — syntax + pytest gate, then commit\n"
                "• `POST /api/brain/self/push` with `approve_push: true` — fork only\n\n"
                "Main is never pushed without your explicit approval."
            ),
            "kind": "self_evolve",
            "plan": plan,
            "repo": status,
        }
    if cmd.startswith("/agent"):
        topic = message.strip()[6:].strip(" :.")
        if not topic:
            return {
                "reply": "Usage: `/agent What is DNA and how does it relate to genetics?`",
                "kind": "agent",
            }
        agent = run_agent_loop(topic)
        return {
            "reply": agent["answer"],
            "kind": "agent",
            "agent": agent,
            "citations": agent.get("citations", []),
        }
    if cmd.startswith("/research"):
        topic = message.strip()[9:].strip(" ?.")
        if not topic:
            return {
                "reply": "Usage: `/research blood` — cross-domain taxonomy search + Ciper facets.",
                "kind": "research",
            }
        result = ciper_research(topic)
        if not result:
            return {"reply": f"No taxonomy match for «{topic}» yet.", "kind": "research"}
        lines = [
            f"**Subject:** {result.subject}",
            f"**Mode:** {result.mode}",
            f"**Reply:** {result.reply}",
        ]
        if result.domains_spanned:
            lines.append(f"**Domains:** {', '.join(result.domains_spanned)}")
        if result.cross_domain_paths:
            lines.append("**Paths:** " + ", ".join(result.cross_domain_paths[:5]))
        if result.facets:
            lines.append("**Ciper facets:** " + ", ".join(result.facets))
        if result.agi_traits:
            lines.append("**AGI traits:** " + ", ".join(result.agi_traits))
        return {"reply": "\n".join(lines), "kind": "research", "ciper": result.to_dict()}
    if cmd == "/vitals":
        return {"reply": "Open `/security/status` or `/organism/vitals` for live organ vitals.", "kind": "vitals"}
    return None


def chat(message: str, *, session_id: str | None = None) -> dict[str, Any]:
    """Process a chat message — psychology brain wraps algorithm brain output."""
    text = _resolve_followup((message or "").strip(), session_id)

    def done(payload: dict[str, Any]) -> dict[str, Any]:
        out = _finalize(payload, text or (message or ""))
        reply = str(out.get("reply", "")).strip()
        original = text or (message or "")
        if reply and _is_classification_leak(reply):
            recovered = _fallback_to_predict(original, session_id=session_id)
            out["reply"] = recovered
            out["kind"] = "predict_leak_recovered"
            reply = recovered
        if session_id and reply:
            append_turn(session_id, user=original, assistant=reply)
        return out

    if not text:
        return done({"error": "empty message", "reply": "Send a message or try `/help`."})

    if len(text) > 8000:
        return done(
            {"error": "message too long", "reply": "Please keep messages under 8000 characters."}
        )

    # Rule 1 — own output echo guard (before all routing)
    if is_own_output(text, session_id) or is_system_echo(text):
        kind = "self_echo_detected" if is_own_output(text, session_id) else "echo_detected"
        reply = SELF_ECHO_DETECTED_REPLY if kind == "self_echo_detected" else ECHO_DETECTED_REPLY
        return done({
            "reply": reply,
            "kind": kind,
            "session_id": session_id,
            "learning": learning_snapshot(),
        })

    # Rule 0.5 — exact arithmetic before routing logic
    deterministic = _deterministic_payload(text, session_id=session_id)
    if deterministic:
        return done(deterministic)

    cmd = _command_response(text)
    if cmd:
        payload = {
            "reply": cmd["reply"],
            "kind": cmd["kind"],
            "session_id": session_id,
            "learning": learning_snapshot(),
        }
        if "snapshot" in cmd:
            payload["snapshot"] = cmd["snapshot"]
        if "timeline" in cmd:
            payload["timeline"] = cmd["timeline"]
        if "ciper" in cmd:
            payload["ciper"] = cmd["ciper"]
        if "roadmap" in cmd:
            payload["roadmap"] = cmd["roadmap"]
        if "simulation" in cmd:
            payload["simulation"] = cmd["simulation"]
        if "agent" in cmd:
            payload["agent"] = cmd["agent"]
        if "plan" in cmd:
            payload["plan"] = cmd["plan"]
        if "repo" in cmd:
            payload["repo"] = cmd["repo"]
        if "audit" in cmd:
            payload["audit"] = cmd["audit"]
        if "curiosity" in cmd:
            payload["curiosity"] = cmd["curiosity"]
        if "proposals" in cmd:
            payload["proposals"] = cmd["proposals"]
        if "deploy" in cmd:
            payload["deploy"] = cmd["deploy"]
        if "proposal" in cmd:
            payload["proposal"] = cmd["proposal"]
        if "citations" in cmd:
            payload["citations"] = cmd["citations"]
        return done(payload)

    if is_agent_task(text):
        return done(_agent_payload(text, session_id=session_id))

    from brain.curiosity_engine import is_curiosity_request

    if is_curiosity_request(text):
        from app.curiosity_sandbox import run_curiosity_cycle

        result = run_curiosity_cycle()
        return done({
            "reply": result.get("report", "Curiosity research complete."),
            "kind": "curiosity",
            "session_id": session_id,
            "learning": learning_snapshot(),
            "curiosity": result,
        })

    from brain.self_audit import format_self_audit_report, is_self_audit_request, run_self_audit

    if is_self_audit_request(text):
        audit = run_self_audit()
        return done({
            "reply": format_self_audit_report(audit),
            "kind": "self_audit",
            "session_id": session_id,
            "learning": learning_snapshot(),
            "audit": audit,
        })

    from brain.identity_handler import handle_identity, is_identity_question
    from brain.philosophy_handler import (
        handle_philosophy_question,
        is_personal_belief_question,
        is_philosophy_question,
    )
    from brain.web_search import web_search_enabled

    # Rule 2 — self-directed identity / belief before simple_qa
    if is_self_directed(text) and is_opinion_or_identity(text):
        return done(_handle_identity_or_belief(text, session_id=session_id))

    if is_identity_question(text):
        return done(handle_identity(text, session_id=session_id))

    # Rule 3 — personal belief / directed opinion before philosophy concepts
    from brain.philosophy_handler import is_directed_opinion_question

    if is_directed_opinion_question(text):
        phil = handle_philosophy_question(
            text,
            session_id=session_id,
            predict_fn=_predict_with_search_fallback,
            classify_fn=lambda _t: None,
            learning_snapshot_fn=learning_snapshot,
        )
        if phil:
            return done(phil)

    # Rule 4 — philosophy concept questions before search / deep concept
    if is_philosophy_question(text):
        phil = handle_philosophy_question(
            text,
            session_id=session_id,
            predict_fn=_predict_with_search_fallback,
            classify_fn=_classify_message,
            learning_snapshot_fn=learning_snapshot,
        )
        if phil:
            return done(phil)

    from brain.combinatorial_creation import handle_creation_request, is_creation_request

    if is_creation_request(text):
        return done(handle_creation_request(text, session_id=session_id))

    if is_code_question(text):
        code_payload = _code_payload(text, session_id=session_id)
        if code_payload:
            return done(code_payload)
        return done(
            {
                "reply": (
                    "I couldn't produce verified code for that yet. "
                    "Try naming the function explicitly, e.g. `Write a Python function add(a, b)`."
                ),
                "kind": "code_abstain",
                "session_id": session_id,
                "learning": learning_snapshot(),
            }
        )

    if (
        web_search_enabled()
        and is_search_question(text)
        and not is_personal_belief_question(text)
    ):
        return done(_search_and_opine(text, session_id=session_id))

    # Rule 5 — named entity before deep concept
    if is_named_entity_question(text):
        return done(_handle_named_entity(text, session_id=session_id))

    # Rule 6 — deep concept last resort for unmatched "what is X"
    if is_deep_concept_question(text):
        return done(_handle_deep_concept(text, session_id=session_id))

    nl = _simple_nl_response(text)
    if nl and not is_deep_concept_question(text):
        return done(
            {
                "reply": to_simple_answer(nl),
                "kind": "chat",
                "session_id": session_id,
                "learning": learning_snapshot(),
                "simple_qa": True,
            }
        )

    if is_prediction_question(text):
        return done(_brain_predict_payload(text, session_id=session_id))

    ciper_payload = _ciper_chat_payload(text, session_id=session_id)
    if ciper_payload:
        return done(ciper_payload)

    if is_simple_question(text):
        simple = _simple_chat_reply(text, session_id=session_id)
        simple["session_id"] = session_id
        simple["learning"] = learning_snapshot()
        return done(simple)

    # Rule 7 — disambiguation before classifier when scores are tied
    top_matches = _classification_top_matches(text)
    disambig = _disambiguation_payload(text, top_matches)
    if disambig:
        disambig["session_id"] = session_id
        disambig["learning"] = learning_snapshot()
        return done(disambig)

    classification = _classify_message(text)
    if ciper_payload:
        if classification:
            ciper_payload["classification"] = classification
        return done(ciper_payload)

    learning = learning_snapshot()

    with get_session() as session:
        from db.models import Document

        doc_count = session.scalar(select(func.count()).select_from(Document)) or 0
        active = _active_micro_progress(session)

    if classification:
        enriched = f"domain context {classification['label']} question {text.strip().lower()}"
        result = _predict_with_search_fallback(enriched, session_id=session_id, force=True)
        if result and result.get("answer") and not result.get("abstained"):
            reply = str(result["answer"]).strip()
            if len(reply) > 20 and not _is_classification_leak(reply):
                kind = "search_opinion" if result.get("search_opinion") else "predict"
                payload: dict[str, Any] = {
                    "reply": reply,
                    "kind": kind,
                    "session_id": session_id,
                    "learning": learning,
                    "brain_predict": not result.get("search_opinion"),
                    "classification": classification,
                }
                if result.get("sources"):
                    payload["sources"] = result["sources"]
                return done(payload)

        from brain.philosophy_handler import philosophy_fallback_if_needed

        fallback = philosophy_fallback_if_needed(text, result)
        if fallback:
            fallback["session_id"] = session_id
            fallback["learning"] = learning
            fallback["classification"] = classification
            return done(fallback)

        confidence = float(result.get("confidence", 0) or 0) if result else 0.0
        if confidence < _SEARCH_CONFIDENCE_THRESHOLD and web_search_enabled():
            search_payload = _search_and_opine(text, session_id=session_id)
            if search_payload.get("kind") == "search_opinion":
                search_payload["classification"] = classification
                return done(search_payload)

        reply = (
            f"I mapped your question to **{classification['label']}** "
            f"({classification['confidence']:.0%} confidence) but need more corpus "
            f"in that domain for a full answer.\n\n"
            "Aureon learns by collecting domain text, labeling, training weights, "
            "and graduating grade levels — preschool through doctorate."
        )
    else:
        reply = (
            "No production classifier is promoted yet — I'm still in early grade cycles "
            "(trainer needs ≥2 label classes). Auto-learn is running on Railway; "
            "try `/status` to see the latest cycle or `/grades` for the curriculum timeline.\n\n"
            f"**Corpus:** {doc_count} documents ingested. "
        )
        if active:
            reply += (
                f"**Current focus:** `{active['path']}` @ grade **{active['current_grade']}**."
            )

    return done(
        {
            "reply": reply,
            "kind": "chat",
            "session_id": session_id,
            "classification": classification,
            "learning": learning,
            "active_micro": active,
        }
    )

