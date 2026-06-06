"""Conversational intelligence — context stack, intent resolution, continuation routing.

Maps human conversation habits into explicit layers:
  1. Context stack (working memory)
  2. Intent resolver (continuation vs new request)
  3. Semantic weight (meaning relative to active topic)
  4. Response selection (handled by callers — search, reflection, predict)
  5. Momentum (depth level, pacing)
  6. Self-correction (audit before output)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

_CONTINUATION_PHRASES = (
    "dive deeper",
    "go deeper",
    "tell me more",
    "more about that",
    "about that",
    "explain that",
    "what about that",
    "keep going",
    "continue",
    "what else",
    "expand on",
    "elaborate",
    "more detail",
    "more on that",
    "go on",
    "and then",
    "say more",
    "more on this",
    "keep talking",
)

_CONTINUATION_WORDS = frozenset(
    {"more", "deeper", "continue", "go", "dive", "elaborate", "expand", "else"}
)

_CLASSIFICATION_LEAK_RE = re.compile(r"^[a-z_]+\.[a-z_]+\.[a-z_]+", re.I)
_SOURCES_SUFFIX_RE = re.compile(r"\n\nSources:.*$", re.DOTALL | re.IGNORECASE)
_ROBOTIC_MARKERS = (
    "based on ",
    "from the zophiel lens",
    "additional context from search suggests",
)


@dataclass(frozen=True)
class ConversationStack:
    active_topic: str
    topic_kind: str
    depth_level: int
    last_user_message: str
    last_kind: str


@dataclass(frozen=True)
class ResolvedMessage:
    original_text: str
    resolved_text: str
    is_continuation: bool
    depth_delta: int


def is_continuation_message(text: str) -> bool:
    """Layer 2 — short vague messages that continue the prior thread."""
    q = text.strip().lower().rstrip("?").strip()
    if not q or len(q) > 80:
        return False
    if any(phrase in q for phrase in _CONTINUATION_PHRASES):
        return True
    words = q.split()
    if len(words) <= 4 and any(w in _CONTINUATION_WORDS for w in words):
        return True
    return False


def resolve_message(text: str, session_id: str | None) -> ResolvedMessage:
    """Layer 2 + 3 — resolve referential follow-ups to the active topic."""
    original = text.strip()
    if not session_id or not is_continuation_message(original):
        return ResolvedMessage(original, original, False, 0)

    from app.session_memory import get_conversation_stack, get_history

    stack = get_conversation_stack(session_id)
    hist = get_history(session_id, limit=1)
    if not hist and not stack:
        return ResolvedMessage(original, original, False, 0)

    topic = (stack or {}).get("active_topic") or hist[-1]["user"]
    kind = (stack or {}).get("topic_kind") or "unknown"
    depth = int((stack or {}).get("depth_level") or 0) + 1

    resolved = f"{topic} — continuation depth {depth}: {original}"
    return ResolvedMessage(original, resolved, True, 1)


def _topic_kind_from_payload(payload: dict[str, Any], user_message: str) -> str:
    kind = str(payload.get("kind") or "chat")
    q = user_message.lower()
    if kind == "search_opinion" or payload.get("sources"):
        if any(t in q for t in ("tech", "technology", "ai ", " silicon")):
            return "live_news"
        return "search_opinion"
    if kind in ("reflection", "philosophy", "predict") and any(
        s in q for s in ("god", "belief", "spirit", "religion", "consciousness")
    ):
        return "reflection"
    if kind == "predict":
        return "predict"
    return kind


def update_stack_from_turn(
    session_id: str | None,
    *,
    user: str,
    payload: dict[str, Any],
) -> None:
    """Layer 1 — persist working memory after each assistant turn."""
    from app.session_memory import set_conversation_stack

    if not session_id:
        return

    user_text = user.strip()
    if not user_text:
        return

    kind = str(payload.get("kind") or "chat")
    continuation = is_continuation_message(user_text)

    if continuation:
        from app.session_memory import get_conversation_stack

        prev = get_conversation_stack(session_id) or {}
        active_topic = str(prev.get("active_topic") or user_text)
        topic_kind = str(prev.get("topic_kind") or _topic_kind_from_payload(payload, user_text))
        depth = int(prev.get("depth_level") or 0) + 1
    else:
        active_topic = user_text
        topic_kind = _topic_kind_from_payload(payload, user_text)
        depth = 0

    set_conversation_stack(
        session_id,
        active_topic=active_topic[:500],
        topic_kind=topic_kind,
        depth_level=depth,
        last_user_message=user_text[:500],
        last_kind=kind,
    )


def _continuation_search_query(stack: ConversationStack | None, hist: list[dict[str, str]]) -> str:
    topic = (stack.active_topic if stack else "") or (hist[-1]["user"] if hist else "")
    depth = stack.depth_level if stack else 1
    if depth <= 1:
        return f"{topic} latest news analysis details"
    return f"{topic} in depth background implications expert analysis"


def audit_response(reply: str, *, is_continuation: bool, active_topic: str) -> str | None:
    """Layer 6 — reject taxonomy leaks and robotic scaffolding."""
    r = reply.strip()
    if not r:
        return "I lost the thread on that — can you say which part you want expanded?"
    if _CLASSIFICATION_LEAK_RE.match(r):
        return None
    lower = r.lower()
    if is_continuation and any(m in lower for m in _ROBOTIC_MARKERS):
        return None
    if is_continuation and active_topic and active_topic.lower()[:20] not in lower:
        if " → " in r and "(" in r:
            return None
    return r


def try_continuation_response(
    text: str,
    *,
    session_id: str | None,
    search_and_opine_fn: Callable[..., dict[str, Any]],
    predict_fn: Callable[..., dict[str, Any] | None],
    philosophy_fn: Callable[..., dict[str, Any] | None] | None = None,
) -> dict[str, Any] | None:
    """
    Layer 4/5 — route continuation messages before taxonomy/classifier.
    Returns a chat payload or None when not a continuation.
    """
    resolved = resolve_message(text, session_id)
    if not resolved.is_continuation or not session_id:
        return None

    from app.session_memory import get_conversation_stack, get_history

    hist = get_history(session_id)
    if not hist:
        return None

    raw_stack = get_conversation_stack(session_id)
    stack = None
    if raw_stack:
        stack = ConversationStack(
            active_topic=str(raw_stack.get("active_topic") or hist[-1]["user"]),
            topic_kind=str(raw_stack.get("topic_kind") or "unknown"),
            depth_level=int(raw_stack.get("depth_level") or 0),
            last_user_message=str(raw_stack.get("last_user_message") or hist[-1]["user"]),
            last_kind=str(raw_stack.get("last_kind") or "chat"),
        )

    topic_kind = stack.topic_kind if stack else "unknown"
    active_topic = stack.active_topic if stack else hist[-1]["user"]
    depth = (stack.depth_level if stack else 0) + 1

    payload: dict[str, Any] | None = None

    if topic_kind in ("search_opinion", "live_news", "unknown"):
        from app.chat_service import is_search_question

        last_user = hist[-1]["user"]
        if topic_kind in ("search_opinion", "live_news") or is_search_question(last_user):
            query = _continuation_search_query(stack, hist)
            payload = search_and_opine_fn(
                query,
                session_id=session_id,
                depth=depth,
                continuation=True,
            )

    if payload is None and topic_kind == "reflection" and philosophy_fn is not None:
        enriched = f"{active_topic} — go deeper: {text}"
        payload = philosophy_fn(enriched, session_id=session_id)

    if payload is None:
        query = _continuation_search_query(stack, hist)
        from brain.web_search import web_search_enabled

        if web_search_enabled():
            payload = search_and_opine_fn(
                query,
                session_id=session_id,
                depth=depth,
                continuation=True,
            )
        else:
            result = predict_fn(query, session_id=session_id, force=True)
            if result and result.get("answer"):
                payload = {
                    "reply": str(result["answer"]),
                    "kind": "predict",
                    "session_id": session_id,
                    "brain_predict": True,
                }

    if not payload:
        return None

    reply = str(payload.get("reply", "")).strip()
    reply = _SOURCES_SUFFIX_RE.sub("", reply).strip()
    audited = audit_response(reply, is_continuation=True, active_topic=active_topic)
    if audited is None and topic_kind in ("search_opinion", "live_news"):
        query = _continuation_search_query(stack, hist)
        payload = search_and_opine_fn(
            query,
            session_id=session_id,
            depth=depth,
            continuation=True,
        )
        reply = str(payload.get("reply", "")).strip()
        reply = _SOURCES_SUFFIX_RE.sub("", reply).strip()
        audited = audit_response(reply, is_continuation=True, active_topic=active_topic)

    if audited is None:
        audited = (
            f"Still on {active_topic[:120]} — I couldn't pull a clean deeper layer. "
            "Try naming one headline or angle you want unpacked."
        )

    payload["reply"] = audited
    payload["continuation"] = True
    payload["conversation_depth"] = depth
    payload["active_topic"] = active_topic
    return payload
