"""Analytical brain - route deep questions without hardcoded answers.

This module extracts the user's intended subject, domain, and taxonomy paths for
multi-clause questions. It deliberately does not store final answer prose; chat
must still answer through corpus retrieval, predict, or search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnalyticalRoute:
    domain: str
    subject: str
    normalized_query: str
    confidence: float
    taxonomy_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "subject": self.subject,
            "normalized_query": self.normalized_query,
            "confidence": self.confidence,
            "taxonomy_paths": list(self.taxonomy_paths),
            "method": "analytical_route",
        }


@dataclass(frozen=True)
class _RouteRule:
    domain: str
    subject: str
    normalized_query: str
    required: tuple[str, ...]
    any_of: tuple[str, ...]
    taxonomy_paths: tuple[str, ...] = ()
    confidence: float = 0.72


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


_ROUTE_RULES: tuple[_RouteRule, ...] = (
    _RouteRule(
        "history_and_civilization",
        "economic causes of Rome's fall",
        "economic causes behind the fall of the Roman Empire",
        ("roman", "empire"),
        ("economic", "fall"),
        ("humanities.history.ancient_history",),
    ),
    _RouteRule(
        "history_and_civilization",
        "ancient astronomy evidence",
        "ancient civilization astronomy evidence",
        ("ancient", "astronomy"),
        ("civilization", "evidence", "advanced"),
        ("science_and_natural_philosophy.astronomy_and_cosmology.planetary_science",),
    ),
    _RouteRule(
        "psychology_and_human_behavior",
        "self-sabotage near success",
        "psychological mechanism of self sabotage near success",
        ("self-sabotage", "success"),
        ("psychological", "mechanism", "closest"),
        ("social_sciences.psychology.clinical_psychology",),
    ),
    _RouteRule(
        "psychology_and_human_behavior",
        "narcissism vs dark triad",
        "narcissism dark triad personality conversation identification",
        ("narcissism", "dark triad"),
        ("conversation", "identify", "difference"),
        ("social_sciences.psychology.personality_psychology",),
    ),
    _RouteRule(
        "science_and_physics",
        "quantum entanglement",
        "quantum entanglement plain language physics",
        ("quantum", "entanglement"),
        ("plain", "physics", "explain"),
        ("science_and_natural_philosophy.physics.quantum_mechanics",),
    ),
    _RouteRule(
        "science_and_physics",
        "quantum artificial intelligence",
        "quantum artificial intelligence hybrid quantum computing AI",
        ("quantum", "artificial intelligence"),
        ("works", "explain", "intelligence"),
        (
            "science_and_natural_philosophy.physics.quantum_mechanics",
            "technology_and_engineering.computer_science.artificial_intelligence",
            "technology_and_engineering.computer_science.quantum_computing",
        ),
    ),
    _RouteRule(
        "science_and_physics",
        "quantum computer",
        "quantum computer qubits superposition entanglement gates measurement",
        ("quantum", "computer"),
        ("work", "works", "explain"),
        (
            "science_and_natural_philosophy.physics.quantum_mechanics",
            "technology_and_engineering.computer_science.quantum_computing",
        ),
    ),
    _RouteRule(
        "technology_and_engineering",
        "artificial intelligence algorithm",
        "artificial intelligence algorithm supervised learning machine learning",
        ("artificial intelligence", "algorithm"),
        ("what type", "type of", "what is", "what are"),
        (
            "technology_and_engineering.computer_science.artificial_intelligence",
            "technology_and_engineering.computer_science.machine_learning",
        ),
    ),
    _RouteRule(
        "science_and_physics",
        "entropy and the universe",
        "entropy universe thermodynamics heat death",
        ("entropy", "universe"),
        ("mean", "end", "context"),
        ("science_and_natural_philosophy.physics.thermodynamics",),
    ),
    _RouteRule(
        "economics_and_wealth",
        "legal tax minimization by the ultra-wealthy",
        "ultra wealthy legal tax minimization mechanisms",
        ("wealthy", "tax"),
        ("mechanisms", "zero", "legally"),
        ("social_sciences.economics.public_finance",),
    ),
    _RouteRule(
        "economics_and_wealth",
        "inflation beneficiaries",
        "inflation causes beneficiaries macroeconomics",
        ("inflation",),
        ("reason", "benefits", "exists"),
        ("social_sciences.economics.macroeconomics",),
    ),
    _RouteRule(
        "philosophy_and_consciousness",
        "free will",
        "free will determinism prior causes verdict",
        ("free will",),
        ("prior causes", "verdict", "both sides"),
        ("philosophy.metaphysics.philosophy_of_mind",),
    ),
    _RouteRule(
        "philosophy_and_consciousness",
        "consciousness as open problem",
        "consciousness science hard problem open problem",
        ("consciousness", "science"),
        ("answer", "open problem"),
        ("philosophy.metaphysics.philosophy_of_mind",),
    ),
    _RouteRule(
        "geopolitics_and_power",
        "power structure behind governments",
        "world governments power structure controllers",
        ("power", "governments"),
        ("controllers", "structure"),
        ("governance_and_political_systems.political_power.power_structures",),
    ),
    _RouteRule(
        "geopolitics_and_power",
        "revolutions reproducing systems",
        "why revolutions reproduce old systems",
        ("revolutions", "system"),
        ("same", "destroyed", "producing"),
        ("governance_and_political_systems.political_change.revolutions",),
    ),
    _RouteRule(
        "biology_and_human_body",
        "gut-brain connection",
        "gut brain connection microbiome mental health",
        ("gut-brain", "mental health"),
        ("modern science", "connection", "affect"),
        ("medicine_and_health_sciences.neuroscience.neurobiology",),
    ),
    _RouteRule(
        "biology_and_human_body",
        "aging mechanisms",
        "human aging biological mechanisms senescence epigenetics",
        ("humans", "age"),
        ("biological", "mechanism", "stop"),
        ("medicine_and_health_sciences.biomedical_sciences.gerontology",),
    ),
    _RouteRule(
        "spirituality_and_occult",
        "as above so below",
        "Hermetic principle as above so below physics support",
        ("as above so below",),
        ("hermetic", "physics", "support"),
        ("religion_and_spirituality.esoteric_and_occult_traditions.hermeticism",),
    ),
    _RouteRule(
        "spirituality_and_occult",
        "encoded core of religious texts",
        "major religious texts encoded core comparative religion",
        ("religious texts", "core"),
        ("encoded", "surface doctrine", "common"),
        ("religion_and_spirituality.comparative_religion.sacred_texts",),
    ),
    _RouteRule(
        "linguistics_and_pattern",
        "language revealing psychology",
        "language psychological state hidden intentions psycholinguistics",
        ("language", "psychological state"),
        ("hidden intentions", "person uses", "reveal"),
        ("humanities.linguistics.psycholinguistics",),
    ),
    _RouteRule(
        "linguistics_and_pattern",
        "recurring mythological archetypes",
        "recurring mythological archetypes ancient cultures",
        ("mythological", "archetypes"),
        ("ancient culture", "no contact", "independently"),
        ("humanities.comparative_mythology.archetypes",),
    ),
    _RouteRule(
        "future_and_prediction",
        "major global shift from historical cycles",
        "historical cycles likely global shift next decade",
        ("historical cycles", "global shift"),
        ("next 10 years", "likely"),
        ("social_sciences.sociology.social_change",),
        0.62,
    ),
    _RouteRule(
        "future_and_prediction",
        "dangerous idea spreading",
        "dangerous idea spreading through civilization",
        ("dangerous idea", "civilization"),
        ("spreading", "why"),
        ("philosophy.ethics.applied_ethics",),
        0.66,
    ),
)


def route_analytical_question(text: str) -> AnalyticalRoute | None:
    q = _norm(text)
    if not q or q.startswith("/"):
        return None

    for rule in _ROUTE_RULES:
        if all(term in q for term in rule.required) and any(term in q for term in rule.any_of):
            return AnalyticalRoute(
                domain=rule.domain,
                subject=rule.subject,
                normalized_query=rule.normalized_query,
                confidence=rule.confidence,
                taxonomy_paths=rule.taxonomy_paths,
            )
    return None

