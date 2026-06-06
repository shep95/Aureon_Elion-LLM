"""Psychology brain — how Aureon *acts human* when it responds.

This layer sits ON TOP of the algorithm brain (collector → reward). It does not
train weights or collect data. It shapes tone, pacing, and conversational
pattern using doctrine from the Aureon Files psychology corpus:

- Human psychology Brain.pdf
- Human Emotions.pdf
- Text Human Patterns.pdf
- HUMAN PATTERN RECOGNITION & BIO-LINGUISTICS.pdf
- You need this form of logic in your.txt (Marie/Ciper facet logic)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from brain.simple_qa import is_simple_question, to_simple_answer

# Aureon Files — human response / psychology corpus (local path, separate repo)
PSYCHOLOGY_CORPUS_SOURCES: tuple[str, ...] = (
    "Human psychology Brain.pdf",
    "Human Emotions.pdf",
    "Text Human Patterns.pdf",
    "HUMAN PATTERN RECOGNITION & BIO-LINGUISTICS.pdf",
    "You need this form of logic in your.txt",
)

# Aureon Files — main algorithm / knowledge brain
ALGORITHM_CORPUS_SOURCES: tuple[str, ...] = (
    "Aureon Brain.pdf",
    "Zophiel Brain LLM.pdf",
    "Zophiel Brain LLM (1).pdf",
    "consciousness-ontology-brain.pdf",
    "How To Create AI.txt",
)

_DISTRESS_RE = re.compile(
    r"\b(suicid|kill myself|self[- ]harm|want to die|end my life|hopeless)\b",
    re.IGNORECASE,
)


@dataclass
class PsychologyContext:
    mode: str
    register: str
    corpus: str
    traits: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "register": self.register,
            "corpus": self.corpus,
            "traits": self.traits,
            "sources": list(PSYCHOLOGY_CORPUS_SOURCES),
        }


def _response_mode(payload: dict[str, Any], user_message: str) -> str:
    if _DISTRESS_RE.search(user_message):
        return "crisis_honest"
    ciper = payload.get("ciper") or {}
    if ciper.get("mode") == "decompose":
        return "curious_clarifier"
    if ciper.get("mode") == "answer" and ciper.get("grounded"):
        return "grounded_direct"
    if ciper.get("mode") == "cross_domain":
        return "cross_domain_curiosity"
    if payload.get("simple_qa"):
        return "simple_direct"
    if payload.get("classification"):
        return "classified_direct"
    kind = payload.get("kind", "chat")
    if kind in ("status", "grades", "help", "research", "mind", "think"):
        return "technical_report"
    return "conversational"


def _match_user_length(reply: str, user_message: str) -> str:
    """Bio-linguistics: don't answer a 4-word question with a paragraph."""
    if not is_simple_question(user_message):
        return reply
    return to_simple_answer(reply, max_len=160)


def _shape_curious_clarifier(reply: str) -> str:
    """Ciper/Marie — human clarifier, not a lecture."""
    cleaned = reply.strip()
    if cleaned.lower().startswith("what type of"):
        return cleaned
    if not cleaned.endswith("?"):
        cleaned += "?"
    return cleaned


def _shape_grounded_direct(reply: str) -> str:
    lower = reply.lower()
    if lower.startswith(("from what", "here's", "from my")):
        return reply
    return f"From what I've collected: {reply.lstrip()}"


def _shape_crisis_honest(_reply: str) -> str:
    return (
        "I'm a supervised learning system, not a crisis counselor. "
        "If you're in danger, contact local emergency services or a crisis line now."
    )


def shape_human_reply(
    reply: str,
    *,
    payload: dict[str, Any],
    user_message: str,
) -> tuple[str, PsychologyContext]:
    """Apply psychology brain — how to sound human, not what to know."""
    mode = _response_mode(payload, user_message)
    traits: list[str] = ["text_human_patterns", "bio_linguistic_pacing"]

    if mode == "crisis_honest":
        shaped = _shape_crisis_honest(reply)
        traits.extend(["honest_limits", "no_eliza_theater"])
        register = "crisis"
    elif mode == "technical_report":
        shaped = reply
        register = "technical"
        traits.append("full_detail_on_request")
    elif mode == "curious_clarifier":
        shaped = _shape_curious_clarifier(reply)
        register = "conversational"
        traits.extend(["marie_ciper_logic", "social_clarification"])
    elif mode == "grounded_direct":
        shaped = _shape_grounded_direct(reply)
        register = "conversational"
        traits.extend(["honest_grounding", "simple_question_simple_answer"])
    elif mode == "cross_domain_curiosity":
        shaped = reply
        if not shaped.lower().startswith(("that spans", "that touches")):
            shaped = f"That spans a few domains — {shaped[0].lower()}{shaped[1:]}" if shaped else shaped
        register = "conversational"
        traits.extend(["cross_domain_curiosity", "agi_style_linking"])
    elif mode == "simple_direct":
        shaped = _match_user_length(reply, user_message)
        register = "conversational"
        traits.append("simple_question_simple_answer")
    elif mode == "classified_direct":
        shaped = _match_user_length(reply, user_message)
        register = "conversational"
        traits.extend(["pattern_recognition", "measurable_confidence"])
    else:
        shaped = reply
        register = "conversational"
        traits.append("conversational_default")

    ctx = PsychologyContext(
        mode=mode,
        register=register,
        corpus="psychology_brain",
        traits=traits,
    )
    return shaped, ctx


def finalize_chat_payload(payload: dict[str, Any], user_message: str) -> dict[str, Any]:
    """Last step before chat response — psychology brain wraps algorithm output."""
    reply = str(payload.get("reply", ""))
    shaped, psych = shape_human_reply(reply, payload=payload, user_message=user_message)
    payload["reply"] = shaped
    payload["psychology"] = psych.to_dict()
    payload["brains"] = {
        "psychology": "response_layer — how Aureon acts human",
        "algorithm": "six_regions — collector, verifier, labeler, trainer, evaluator, reward",
    }
    return payload
