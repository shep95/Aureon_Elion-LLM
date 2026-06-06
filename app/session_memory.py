"""In-memory session store — last N turns for conversational context."""

from __future__ import annotations

import os
import threading
from typing import Any

_lock = threading.Lock()
_sessions: dict[str, list[dict[str, str]]] = {}


def _max_turns() -> int:
    raw = os.environ.get("AUREON_SESSION_MAX_TURNS", "10").strip()
    try:
        return max(1, min(int(raw), 50))
    except ValueError:
        return 10


def append_turn(session_id: str | None, *, user: str, assistant: str) -> None:
    if not session_id or not user.strip():
        return
    with _lock:
        turns = _sessions.setdefault(session_id, [])
        turns.append({"user": user.strip(), "assistant": assistant.strip()})
        if len(turns) > _max_turns():
            _sessions[session_id] = turns[-_max_turns() :]


def get_history(session_id: str | None, *, limit: int | None = None) -> list[dict[str, str]]:
    if not session_id:
        return []
    cap = limit if limit is not None else _max_turns()
    with _lock:
        return list(_sessions.get(session_id, [])[-cap:])


def history_as_context(session_id: str | None) -> str:
    """Compact prior turns for predict/RAG prompts."""
    turns = get_history(session_id)
    if not turns:
        return ""
    parts: list[str] = []
    for turn in turns:
        parts.append(f"user said {turn['user']}")
        parts.append(f"assistant said {turn['assistant']}")
    return "conversation " + " ".join(parts) + " "


def session_count() -> int:
    with _lock:
        return len(_sessions)
