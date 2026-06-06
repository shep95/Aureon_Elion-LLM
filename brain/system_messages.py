"""Single source of truth for all system-generated messages and echo detection."""

from __future__ import annotations

FALLBACK_CORPUS = (
    "This question needs deeper corpus grounding than I can compute in time. "
    "Try again after auto-learn finishes, or ask `/mind` for what I know now."
)

FALLBACK_TIMEOUT = (
    "I hit my compute time limit on that question — Railway is often busy during auto-learn. "
    "I tried web search too but could not ground a full answer. "
    "Try again in a moment, ask something shorter, or check `/status`."
)

FALLBACK_TRAINING = (
    "I need more training on this topic — ask me again after the next auto-learn cycle."
)

FALLBACK_PHILOSOPHY = (
    "This touches one of the deepest questions I engage with. "
    "My corpus is still growing in this domain. "
    "What I can say is that the question you are asking "
    "sits at the intersection of consciousness, existence, and meaning — "
    "domains I take seriously. Ask me again after my next learning cycle "
    "and I will have more to offer."
)

FALLBACK_CLASSIFICATION_LEAK = (
    "I classified that question but my predict brain needs more "
    "training to generate a full answer. Ask me again after the "
    "next auto-learn cycle."
)

RATE_LIMIT_PREDICT = "Too many predict requests — wait a minute and try again."

ECHO_DETECTED_REPLY = (
    "It looks like you sent back one of my system messages. "
    "I cannot process my own fallback text as a question. "
    "Try asking something directly and I will do my best to answer."
)

SELF_ECHO_DETECTED_REPLY = (
    "You sent back something I said. "
    "What would you like to know or explore?"
)

ECHO_PREFIXES = (
    "This question needs deeper corpus grounding",
    "I hit my compute time limit",
    "I need more training on this topic",
    "Still training —",
    "Still training -",
    "Supervised ML brain —",
    "Supervised ML brain -",
    "Existence of God maps to",
    "philosophy.metaphysics",
    "This touches one of the deepest questions",
    "I classified that question but my predict brain",
    "Too many predict requests",
    "I mapped your question to **",
    "No production classifier is promoted yet",
)

ALL_SYSTEM_MESSAGES: frozenset[str] = frozenset({
    FALLBACK_CORPUS,
    FALLBACK_TIMEOUT,
    FALLBACK_TRAINING,
    FALLBACK_PHILOSOPHY,
    FALLBACK_CLASSIFICATION_LEAK,
    RATE_LIMIT_PREDICT,
})


def is_system_echo(text: str) -> bool:
    """Detect when input matches a known system fallback (exact or prefix)."""
    stripped = text.strip()
    if stripped in ALL_SYSTEM_MESSAGES:
        return True
    return any(stripped.startswith(prefix) for prefix in ECHO_PREFIXES)


def still_training_reply(
    *,
    doc_count: int,
    active_path: str | None = None,
    current_grade: str | None = None,
) -> str:
    if active_path and current_grade:
        return f"Still training — {doc_count} docs, focus {active_path} @ {current_grade}."
    return f"Still training — {doc_count} docs, no promoted classifier yet."
