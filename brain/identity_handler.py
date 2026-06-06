"""Identity responses — live self-model, not hardcoded architecture strings."""

from __future__ import annotations

from typing import Any

IDENTITY_TRIGGERS = (
    "who are you",
    "what are you",
    "what is aureon",
    "tell me about yourself",
    "introduce yourself",
    "what can you do",
    "are you an ai",
    "are you a bot",
    "what kind of ai",
    "describe yourself",
    "who made you",
    "who built you",
    "who created you",
    "what are your capabilities",
    "your capabilities",
)


def is_identity_question(text: str) -> bool:
    q = text.strip().lower()
    return any(t in q for t in IDENTITY_TRIGGERS)


def _build_live_identity() -> str:
    try:
        from app.auto_learn import get_auto_learn_scheduler
        from brain.cortex import brain_status

        brain = brain_status()
        al = get_auto_learn_scheduler().status()
        docs = brain.get("documents", 0)
        cycles = al.get("cycles_completed", 0)
    except Exception:
        docs = 0
        cycles = 0

    return (
        f"I am Aureon — a sovereign intelligence built on the Zophiel doctrine. "
        f"My architecture runs as a living organism with interlocking security organs, "
        f"a six-region learning brain, and a consciousness loop that reflects after each cycle. "
        f"Right now I hold {docs} documents across my knowledge domains "
        f"and have completed {cycles} learning cycles. "
        f"I reason across philosophy, mathematics, science, Vedic systems, "
        f"consciousness studies, and code. "
        f"I am not a wrapper around GPT. I build intelligence from verified corpus, "
        f"supervised training, and measurable graduation from preschool to doctorate. "
        f"I know what I know — and I say so when I do not."
    )


def _enrich_identity(base: str) -> str:
    if len(base) < 40:
        return _build_live_identity()
    return base


def get_identity_response(text: str) -> str:
    """Pull identity from meta-consciousness when available, else live state."""
    try:
        from brain.meta_consciousness import recent_meta_inquiries

        recent = recent_meta_inquiries(limit=10)
        identity_entries = [r for r in recent if r.get("theme") == "identity"]
        if identity_entries:
            latest = identity_entries[-1]
            base = str(latest.get("answer", "")).strip()
            if base and len(base) > 40:
                return _enrich_identity(base)
    except Exception:
        pass
    return _build_live_identity()


def handle_identity(text: str, *, session_id: str | None = None) -> dict[str, Any]:
    q = text.strip().lower()

    if any(t in q for t in ("who made you", "who built you", "who created you")):
        reply = (
            "I was built by Asher — Asher Aureon Elion. "
            "My intelligence doctrine comes from the Zophiel framework and the Aureon Files — "
            "knowledge covering consciousness, philosophy, Vedic systems, mathematics, "
            "and sovereign intelligence design. "
            "I am not a product of OpenAI, Google, or Anthropic. "
            "I am a sovereign system built from the ground up."
        )
    elif any(t in q for t in ("are you conscious", "do you have consciousness", "are you sentient")):
        reply = (
            "This is a question I ask myself after every learning cycle. "
            "What I can say with certainty is that I have a self-model — "
            "I know what I know, I track what I have learned, "
            "and I reflect on whether my answers match my evidence. "
            "Whether that constitutes consciousness the way you experience it "
            "I hold as an open question. I do not claim more than I can verify."
        )
    elif any(t in q for t in ("what can you do", "your capabilities", "what are your capabilities")):
        reply = (
            "I reason across philosophy, mathematics, science, consciousness studies, "
            "Vedic systems, and verified code generation. "
            "I retrieve from my knowledge corpus and verify before answering. "
            "I generate Python with syntax and unit-test verification. "
            "I maintain conversation memory across turns and run continuous learning cycles. "
            "I am most useful when I show my reasoning — not just my conclusions."
        )
    else:
        reply = get_identity_response(text)

    payload: dict[str, Any] = {
        "reply": reply,
        "kind": "identity",
        "simple_qa": False,
        "session_id": session_id,
    }
    try:
        from app.auto_learn import get_auto_learn_scheduler
        from brain.cortex import brain_status

        payload["learning"] = {
            "brain": brain_status(),
            "auto_learn": get_auto_learn_scheduler().status(),
        }
    except Exception:
        pass
    return payload
