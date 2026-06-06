"""Per-session rate limit for expensive predict-brain calls."""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict


class SessionRateLimiter:
    """Sliding-window counter keyed by session_id."""

    def __init__(self, *, max_per_minute: int | None = None, window_sec: float = 60.0) -> None:
        raw = os.environ.get("AUREON_PREDICT_RATE_LIMIT_PER_MINUTE", "10").strip()
        default = 10
        try:
            default = max(1, int(raw))
        except ValueError:
            pass
        self._max = max_per_minute if max_per_minute is not None else default
        self._window_sec = window_sec
        self._lock = threading.Lock()
        self._timestamps: dict[str, list[float]] = defaultdict(list)

    def try_acquire(self, session_id: str | None) -> bool:
        key = (session_id or "anonymous").strip() or "anonymous"
        now = time.monotonic()
        cutoff = now - self._window_sec
        with self._lock:
            recent = [t for t in self._timestamps[key] if t >= cutoff]
            if len(recent) >= self._max:
                self._timestamps[key] = recent
                return False
            recent.append(now)
            self._timestamps[key] = recent
            return True


_limiter: SessionRateLimiter | None = None


def get_predict_rate_limiter() -> SessionRateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = SessionRateLimiter()
    return _limiter
