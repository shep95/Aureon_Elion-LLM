"""Opinion brain — SOLIA forms a grounded perspective from search results."""

from __future__ import annotations

from typing import Any

OPINION_DOCTRINE = (
    "SOLIA forms opinions by weighing evidence from verified sources "
    "through the Zophiel lens — sovereign reasoning, honest uncertainty, "
    "no false certainty, no appeal to authority alone."
)


def form_opinion(
    question: str,
    search_results: list[dict[str, Any]],
    *,
    domain: str = "general",
) -> dict[str, Any]:
    """Build a structured opinion from search evidence."""
    _ = domain  # reserved for domain-specific framing

    if not search_results or all(r.get("error") for r in search_results):
        return {
            "opinion": None,
            "confidence": 0.0,
            "reason": "no search results available",
        }

    evidence: list[str] = []
    sources: list[str] = []
    for item in search_results:
        text = str(item.get("text", "")).strip()
        source = str(item.get("source", "unknown"))
        if text and not item.get("error"):
            evidence.append(text)
            sources.append(source)

    if not evidence:
        return {
            "opinion": None,
            "confidence": 0.0,
            "reason": "search returned no usable text",
        }

    opinion_parts = [
        f"Based on {len(evidence)} sources including {', '.join(set(sources[:3]))}: ",
        evidence[0][:200],
    ]
    if len(evidence) > 1:
        opinion_parts.append(
            f" Additional context from search suggests: {evidence[1][:150]}"
        )
    opinion_parts.append(
        " From the Zophiel lens — I treat this evidence as a starting point "
        "for deeper inquiry, not a final conclusion."
    )

    return {
        "opinion": "".join(opinion_parts),
        "evidence_count": len(evidence),
        "sources": list(set(sources)),
        "confidence": min(0.5 + (len(evidence) * 0.1), 0.85),
        "doctrine": OPINION_DOCTRINE,
    }
