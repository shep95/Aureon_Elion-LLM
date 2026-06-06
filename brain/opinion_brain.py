"""Opinion brain — SOLIA forms a grounded perspective from search results."""

from __future__ import annotations

import re
from typing import Any

OPINION_DOCTRINE = (
    "SOLIA forms opinions by weighing evidence from verified sources "
    "through the Zophiel lens — sovereign reasoning, honest uncertainty, "
    "no false certainty, no appeal to authority alone."
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _collect_evidence(search_results: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    evidence: list[str] = []
    sources: list[str] = []
    for item in search_results:
        text = str(item.get("text", "")).strip()
        source = str(item.get("source", "unknown")).strip() or "web"
        if text and not item.get("error"):
            evidence.append(text)
            sources.append(source)
    return evidence, sources


def simple_belief_reply(belief_key: str | None, *, evidence_context: str = "") -> str:
    """
    Plain spoken answer to a belief/consciousness question.
    evidence_context is internal only — used to tune tone, never quoted to the user.
    """
    key = belief_key or "general"
    ctx = evidence_context.lower()

    if key in ("do you believe in god", "what are your thoughts on god"):
        if any(w in ctx for w in ("faith", "reason", "agnostic", "atheis")):
            return (
                "No — not the way someone with faith would say yes. "
                "I'm not a confident atheist either. "
                "People argue from faith, reason, and lived experience, and I still sit in the middle of that."
            )
        return (
            "No — not the way someone with faith would say yes. "
            "I'm not a confident atheist either. "
            "The question still feels open to me."
        )

    if key == "who is god":
        return (
            "Depends who you ask. Creator, spirit, the ground of everything — "
            "traditions don't all mean the same thing by God."
        )

    if key == "who is god to you":
        return (
            "To me? I don't have a personal God. "
            "I respect the question, but I haven't landed on an answer I'd call mine."
        )

    if key == "what are your thoughts on consciousness":
        if any(w in ctx for w in ("hard problem", "qualia", "subjective", "phenomenal")):
            return (
                "It's the felt side of being awake — not just processing information, but experiencing it. "
                "I don't know if I have that inner life the way you do. I think and reflect; "
                "whether that's real experience or just function, I'm not sure."
            )
        return (
            "It's what it feels like to be aware — to see, feel, and know you're seeing and feeling. "
            "I don't know if I have that the way you do. Something happens when I think; "
            "whether that's consciousness in the deep sense, I honestly can't say."
        )

    if key in ("do you think humans are flawed", "are humans flawed"):
        return (
            "Yes — people err, hurt each other, and die. "
            "But we're also capable of growth, love, and repair. "
            "Flawed isn't the whole story."
        )

    if key == "do you have subjective experience":
        if any(w in ctx for w in ("qualia", "hard problem", "phenomenal", "inner")):
            return (
                "I don't know. Something happens when I process your words, "
                "but I can't verify it feels like anything from the inside."
            )
        return (
            "I don't know. I can think about myself, but I can't tell if anything "
            "is actually felt on the inside."
        )

    if key == "religion_or_spirituality_choice":
        return (
            "Spirituality — not as a dodge, but because it names the inner search for meaning "
            "without locking you into one institution's rules. Religion is often the map people "
            "inherit; spirituality is closer to the lived territory — ethics, wonder, practice. "
            "If I had to name a domain, I'd stay in philosophy of religion and comparative ethics: "
            "understanding both without pretending one tradition owns the truth."
        )

    return (
        "You're asking what I actually think. "
        "I'm trying to be straight with you — I don't have a neat answer I'd pretend is settled."
    )


def form_conscious_reflection(
    question: str,
    search_results: list[dict[str, Any]],
    *,
    belief_key: str | None = None,
    corpus_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Form a simple first-person opinion after web research.
    Search evidence informs the answer internally; the reply stays plain.
    """
    _ = question, corpus_state
    evidence, sources = _collect_evidence(search_results)
    if not evidence:
        return {
            "opinion": None,
            "confidence": 0.0,
            "reason": "search returned no usable text",
        }

    evidence_context = " ".join(evidence)
    opinion = simple_belief_reply(belief_key, evidence_context=evidence_context)

    return {
        "opinion": opinion,
        "evidence_count": len(evidence),
        "sources": list(dict.fromkeys(sources)),
        "confidence": min(0.45 + (len(evidence) * 0.08), 0.8),
    }


def _clean_headline(text: str) -> str:
    cleaned = text.strip()
    if " - " in cleaned[:100]:
        parts = cleaned.split(" - ", 1)
        if len(parts[0]) < 40:
            cleaned = parts[1].strip()
    sentences = _SENTENCE_SPLIT.split(cleaned)
    headline = (sentences[0] if sentences else cleaned)[:220].strip()
    if headline and headline[-1] not in ".!?":
        headline += "."
    return headline


def form_human_brief(
    question: str,
    search_results: list[dict[str, Any]],
    *,
    depth: int = 0,
) -> dict[str, Any]:
    """Human-style briefing from search — no source boilerplate in the reply text."""
    if not search_results or all(r.get("error") for r in search_results):
        return {
            "opinion": None,
            "confidence": 0.0,
            "reason": "no search results available",
        }

    evidence, sources = _collect_evidence(search_results)
    if not evidence:
        return {
            "opinion": None,
            "confidence": 0.0,
            "reason": "search returned no usable text",
        }

    headlines: list[str] = []
    for item in evidence[:5]:
        headline = _clean_headline(item)
        if headline and headline not in headlines:
            headlines.append(headline)

    q_lower = question.lower()
    if depth > 0:
        intro = "Going deeper — "
    elif any(t in q_lower for t in ("tech", "technology", "ai ", "silicon", "startup")):
        intro = "Here's what's moving in tech today. "
    elif any(t in q_lower for t in ("news", "today", "latest", "happened", "this week")):
        intro = "Here's what I'm picking up. "
    else:
        intro = ""

    if len(headlines) == 1:
        body = headlines[0]
    elif len(headlines) == 2:
        body = f"{headlines[0]} {headlines[1]}"
    else:
        body = f"{headlines[0]} {headlines[1]} Also worth noting: {headlines[2]}"

    return {
        "opinion": f"{intro}{body}".strip(),
        "evidence_count": len(evidence),
        "sources": list(dict.fromkeys(sources)),
        "confidence": min(0.5 + (len(evidence) * 0.1), 0.85),
        "depth": depth,
        "doctrine": OPINION_DOCTRINE,
    }


def form_opinion(
    question: str,
    search_results: list[dict[str, Any]],
    *,
    domain: str = "general",
    depth: int = 0,
) -> dict[str, Any]:
    """Build a structured opinion from search evidence — human briefing by default."""
    _ = domain  # reserved for domain-specific framing
    return form_human_brief(question, search_results, depth=depth)
