"""In-memory session store — last N turns for conversational context."""

from __future__ import annotations

import os
import re
import threading
from typing import Any

_lock = threading.Lock()
_sessions: dict[str, list[dict[str, str]]] = {}
_stacks: dict[str, dict[str, Any]] = {}

_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _max_turns() -> int:
    raw = os.environ.get("AUREON_SESSION_MAX_TURNS", "10").strip()
    try:
        return max(1, min(int(raw), 50))
    except ValueError:
        return 10


def _max_sessions() -> int:
    raw = os.environ.get("AUREON_SESSION_MAX_SESSIONS", "5000").strip()
    try:
        return max(100, min(int(raw), 100_000))
    except ValueError:
        return 5000


def _sanitize_session_id(session_id: str | None) -> str | None:
    if not session_id:
        return None
    sid = session_id.strip()
    if not _SESSION_ID_RE.match(sid):
        return None
    return sid


def append_turn(session_id: str | None, *, user: str, assistant: str) -> None:
    sid = _sanitize_session_id(session_id)
    if not sid or not user.strip():
        return
    user_text = user.strip()[:4000]
    assistant_text = assistant.strip()[:8000]
    with _lock:
        if sid not in _sessions and len(_sessions) >= _max_sessions():
            oldest = next(iter(_sessions))
            del _sessions[oldest]
        turns = _sessions.setdefault(sid, [])
        turns.append({"user": user_text, "assistant": assistant_text})
        if len(turns) > _max_turns():
            _sessions[sid] = turns[-_max_turns() :]


def get_history(session_id: str | None, *, limit: int | None = None) -> list[dict[str, str]]:
    sid = _sanitize_session_id(session_id)
    if not sid:
        return []
    cap = limit if limit is not None else _max_turns()
    with _lock:
        return list(_sessions.get(sid, [])[-cap:])


def was_my_output(session_id: str | None, text: str) -> bool:
    """Check if input matches something the assistant recently said in this session."""
    history = get_history(session_id)
    text_stripped = text.strip().lower()
    if len(text_stripped) <= 30:
        return False
    for turn in history:
        assistant_text = turn.get("assistant", "").strip().lower()
        if not assistant_text:
            continue
        if text_stripped == assistant_text:
            return True
        if text_stripped in assistant_text or assistant_text in text_stripped:
            return True
    return False


def history_as_context(session_id: str | None) -> str:
    """Compact prior turns for predict/RAG prompts."""
    turns = get_history(session_id)
    if not turns:
        return ""
    parts: list[str] = []
    for turn in turns:
        parts.append(f"user said {turn['user'][:500]}")
        parts.append(f"assistant said {turn['assistant'][:500]}")
    return "conversation " + " ".join(parts) + " "


def session_count() -> int:
    with _lock:
        return len(_sessions)


def get_conversation_stack(session_id: str | None) -> dict[str, Any] | None:
    """Working memory for conversational intelligence (active topic, depth, kind)."""
    sid = _sanitize_session_id(session_id)
    if not sid:
        return None
    with _lock:
        stack = _stacks.get(sid)
        return dict(stack) if stack else None


def set_conversation_stack(session_id: str | None, **fields: Any) -> None:
    sid = _sanitize_session_id(session_id)
    if not sid:
        return
    with _lock:
        if sid not in _stacks and len(_stacks) >= _max_sessions():
            oldest = next(iter(_stacks))
            del _stacks[oldest]
        current = _stacks.setdefault(sid, {})
        for key, value in fields.items():
            if value is not None:
                current[key] = value
