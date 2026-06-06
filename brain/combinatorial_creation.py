"""Combinatorial creation doctrine — novel things must fuse two or more existing precursors.

SOLIA rule: to create something genuinely new (medicine, software, algorithm, compound,
process), combine at least two established precursors so their interaction produces an
emergent effect neither precursor has alone — like two medicines forming a new compound
with a different bodily reaction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

COMBINATORIAL_CREATION_DOCTRINE = (
    "To create something new — medicine, software, algorithm, or any invention — "
    "you must combine two or more existing precursors into a new compound or system. "
    "The novelty lives in the interaction: the fused result must behave differently "
    "than either parent alone, the way two medicines form a new compound with a "
    "different reaction in the body. Invention from nothing is not permitted; "
    "combinatorial synthesis from verified precursors is required."
)

CREATE_TRIGGERS = (
    "create a new",
    "create new",
    "invent a",
    "invent new",
    "brand new",
    "never been created",
    "never been done",
    "nobody has created",
    "novel compound",
    "new compound",
    "new medicine",
    "new drug",
    "new cure",
    "cure for",
    "combine two",
    "combine different",
    "fuse ",
    "synthesize a new",
    "design a new",
    "build something new",
    "new algorithm",
    "new software",
    "new treatment",
)

DOMAIN_HINTS: dict[str, tuple[str, ...]] = {
    "medicine": ("medicine", "drug", "cure", "cancer", "compound", "treatment", "therapy", "pharma"),
    "software": ("software", "algorithm", "app", "code", "system", "api", "module", "program"),
    "general": ("create", "invent", "novel", "new"),
}


@dataclass
class Precursor:
    name: str
    role: str
    source: str
    citation: dict[str, Any] | None = None


@dataclass
class CombinatorialPlan:
    request: str
    domain: str
    precursors: list[Precursor] = field(default_factory=list)
    novel_artifact: str = ""
    emergent_effect: str = ""
    verification_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doctrine": COMBINATORIAL_CREATION_DOCTRINE,
            "request": self.request,
            "domain": self.domain,
            "precursors": [
                {
                    "name": p.name,
                    "role": p.role,
                    "source": p.source,
                    "citation": p.citation,
                }
                for p in self.precursors
            ],
            "novel_artifact": self.novel_artifact,
            "emergent_effect": self.emergent_effect,
            "verification_steps": self.verification_steps,
            "precursor_count": len(self.precursors),
            "valid": len(self.precursors) >= 2,
        }


def is_creation_request(text: str) -> bool:
    q = text.strip().lower()
    if not q or q.startswith("/"):
        return False
    return any(t in q for t in CREATE_TRIGGERS)


def _infer_domain(text: str) -> str:
    q = text.lower()
    for domain, hints in DOMAIN_HINTS.items():
        if domain == "general":
            continue
        if any(h in q for h in hints):
            return domain
    return "general"


def _default_precursors(domain: str, request: str) -> list[Precursor]:
    """Fallback precursors when corpus is thin — still enforces two-parent rule."""
    if domain == "medicine":
        return [
            Precursor(
                name="Targeted pathway inhibitor (existing class)",
                role="Parent A — blocks a specific disease pathway",
                source="combinatorial_doctrine_fallback",
            ),
            Precursor(
                name="Immune-modulating agent (existing class)",
                role="Parent B — shifts host response",
                source="combinatorial_doctrine_fallback",
            ),
        ]
    if domain == "software":
        return [
            Precursor(
                name="Retrieval-verified corpus pipeline",
                role="Parent A — grounds outputs in verified sources",
                source="solia_architecture",
            ),
            Precursor(
                name="Neural synthesis with unit-test gates",
                role="Parent B — generates novel structure under verification",
                source="solia_architecture",
            ),
        ]
    return [
        Precursor(
            name="Established method A",
            role="Parent A — first verified precursor",
            source="combinatorial_doctrine_fallback",
        ),
        Precursor(
            name="Established method B",
            role="Parent B — second verified precursor",
            source="combinatorial_doctrine_fallback",
        ),
    ]


def _hits_to_precursors(hits: list[Any], *, limit: int = 2) -> list[Precursor]:
    precursors: list[Precursor] = []
    seen_titles: set[str] = set()
    for hit in hits:
        title = str(getattr(hit, "title", "") or "corpus fragment").strip()
        key = title.lower()[:80]
        if key in seen_titles:
            continue
        seen_titles.add(key)
        snippet = hit.snippet(220) if hasattr(hit, "snippet") else str(hit)[:220]
        precursors.append(
            Precursor(
                name=title,
                role=f"Precursor {len(precursors) + 1} — {snippet[:180]}…",
                source=str(getattr(hit, "source", "corpus")),
                citation=hit.citation() if hasattr(hit, "citation") else None,
            )
        )
        if len(precursors) >= limit:
            break
    return precursors


def _novel_name(domain: str, request: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", request.lower())[:40].strip("-") or "synthesis"
    prefix = {"medicine": "Compound", "software": "System", "general": "Fusion"}.get(domain, "Fusion")
    return f"{prefix}-SOLIA-{slug}"


def _emergent_effect(domain: str, a: Precursor, b: Precursor) -> str:
    if domain == "medicine":
        return (
            f"Combining **{a.name}** with **{b.name}** targets two mechanisms simultaneously. "
            f"The emergent effect is not additive — the fused compound should produce a "
            f"pharmacodynamic reaction neither parent produces alone (e.g. synergy, "
            f"re-purposed binding, or immune reactivation). This must be validated in "
            f"vitro, in vivo, and clinical safety tiers before any human use."
        )
    if domain == "software":
        return (
            f"Merging **{a.name}** with **{b.name}** yields a system whose behavior "
            f"differs from either stack alone: retrieval alone cannot invent structure; "
            f"synthesis alone hallucinates. Together they produce verified novelty — "
            f"new outputs that pass tests neither parent could pass in isolation."
        )
    return (
        f"Fusing **{a.name}** and **{b.name}** produces emergent capability: "
        f"a third behavior that requires both precursors interacting, not either alone."
    )


def _verification_steps(domain: str) -> list[str]:
    if domain == "medicine":
        return [
            "Document both precursor mechanisms and interaction hypothesis",
            "Model binding / pathway interference computationally",
            "Synthesize compound; confirm structure (NMR / mass spec)",
            "In vitro efficacy + toxicity vs each parent alone",
            "Preclinical safety; only then consider trials",
        ]
    if domain == "software":
        return [
            "Name both parent modules and their contracts",
            "Implement fusion layer with explicit interface tests",
            "Benchmark emergent behavior vs each parent in isolation",
            "Security review + regression suite",
            "Graduate through grade gates before production",
        ]
    return [
        "Identify two verified precursors with citations",
        "Define fusion interface and emergent success criteria",
        "Test fused artifact vs each parent separately",
        "Peer review before claiming novelty",
    ]


def plan_combinatorial_creation(text: str) -> CombinatorialPlan:
    """Build a two-or-more precursor fusion plan for a creation request."""
    domain = _infer_domain(text)
    precursors: list[Precursor] = []

    try:
        from brain.vector_rag import retrieve_with_citations

        _, hits, _ = retrieve_with_citations(text, top_k=8)
        precursors = _hits_to_precursors(hits, limit=3)
    except Exception:
        precursors = []

    if len(precursors) < 2:
        defaults = _default_precursors(domain, text)
        existing_names = {p.name.lower() for p in precursors}
        for d in defaults:
            if d.name.lower() not in existing_names:
                precursors.append(d)
            if len(precursors) >= 2:
                break

    a, b = precursors[0], precursors[1]
    novel = _novel_name(domain, text)
    plan = CombinatorialPlan(
        request=text.strip(),
        domain=domain,
        precursors=precursors[:3],
        novel_artifact=novel,
        emergent_effect=_emergent_effect(domain, a, b),
        verification_steps=_verification_steps(domain),
    )
    return plan


def format_creation_reply(plan: CombinatorialPlan) -> str:
    lines = [
        "**SOLIA combinatorial creation** — novelty requires fusing two or more precursors.",
        "",
    ]
    for i, p in enumerate(plan.precursors, start=1):
        lines.append(f"**Parent {i} ({p.source}):** {p.name}")
        lines.append(f"  _{p.role}_")
        lines.append("")
    lines.extend(
        [
            f"**New compound / system:** `{plan.novel_artifact}`",
            "",
            f"**Emergent effect:** {plan.emergent_effect}",
            "",
            "**Verification path:**",
        ]
    )
    lines.extend(f"• {step}" for step in plan.verification_steps)
    if plan.domain == "medicine":
        lines.append("")
        lines.append(
            "_Research framework only — not medical advice. "
            "Human trials require regulatory approval._"
        )
    return "\n".join(lines)


def handle_creation_request(text: str, *, session_id: str | None = None) -> dict[str, Any]:
    plan = plan_combinatorial_creation(text)
    citations = [p.citation for p in plan.precursors if p.citation]
    payload: dict[str, Any] = {
        "reply": format_creation_reply(plan),
        "kind": "combinatorial_creation",
        "simple_qa": False,
        "session_id": session_id,
        "combinatorial": plan.to_dict(),
    }
    if citations:
        payload["citations"] = citations
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
