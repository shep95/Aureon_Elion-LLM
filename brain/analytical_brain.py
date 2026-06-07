"""Analytical brain - structured answers for deep multi-clause questions.

This route handles the kind of human question that asks for causes, mechanisms,
evidence, both sides, or power structure. Those prompts should not be reduced to
one stray keyword and sent through Ciper decomposition.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnalyticalAnswer:
    domain: str
    subject: str
    answer: str
    confidence: float
    taxonomy_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "subject": self.subject,
            "confidence": self.confidence,
            "taxonomy_paths": list(self.taxonomy_paths),
            "method": "analytical_brain",
        }


@dataclass(frozen=True)
class _Rule:
    domain: str
    subject: str
    required: tuple[str, ...]
    any_of: tuple[str, ...]
    answer: str
    taxonomy_paths: tuple[str, ...] = ()
    confidence: float = 0.74


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


_RULES: tuple[_Rule, ...] = (
    _Rule(
        "history_and_civilization",
        "economic causes of Rome's fall",
        ("roman", "empire"),
        ("economic", "fall"),
        (
            "The non-textbook economic core is that Rome lost the surplus machine that paid for the state. "
            "Conquest slowed, slave inflows fell, tax pressure rose on productive farmers and towns, currency "
            "debasement weakened trust, and military costs kept climbing. The empire did not simply get "
            "invaded; its fiscal base hollowed out until armies, borders, and cities became too expensive to sustain."
        ),
        ("humanities.history.ancient_history",),
    ),
    _Rule(
        "history_and_civilization",
        "ancient astronomy evidence",
        ("ancient", "astronomy"),
        ("civilization", "evidence", "advanced"),
        (
            "The strongest case is Babylon for predictive mathematical astronomy: they tracked eclipses, planetary "
            "cycles, and synodic periods with durable numerical schemes. Egypt aligned monuments and calendars well, "
            "Maya astronomy was extremely precise for Venus and ritual calendars, and Greek astronomy built geometry. "
            "If the standard is evidence of repeatable prediction, Babylon has the best claim."
        ),
        ("science_and_natural_philosophy.astronomy_and_cosmology.planetary_science",),
    ),
    _Rule(
        "psychology_and_human_behavior",
        "self-sabotage near success",
        ("self-sabotage", "success"),
        ("psychological", "mechanism", "closest"),
        (
            "Self-sabotage near success usually comes from threat prediction, not stupidity. Success changes identity, "
            "raises expectations, exposes the person to judgment, and can violate an old belief like 'I am not safe "
            "when visible.' The mechanism is avoidance relief: the person creates a smaller failure they can control "
            "instead of risking a larger unknown outcome."
        ),
        ("social_sciences.psychology.clinical_psychology",),
    ),
    _Rule(
        "psychology_and_human_behavior",
        "narcissism vs dark triad",
        ("narcissism", "dark triad"),
        ("conversation", "identify", "difference"),
        (
            "Narcissism is mainly grandiosity, entitlement, status hunger, and fragile ego defense. The dark triad is "
            "broader: narcissism plus Machiavellian manipulation and psychopathic callousness. In conversation, "
            "narcissism centers the self and reacts badly to status loss; Machiavellianism probes leverage; "
            "psychopathy shows low remorse, thrill-seeking, and emotional coldness."
        ),
        ("social_sciences.psychology.personality_psychology",),
    ),
    _Rule(
        "science_and_physics",
        "quantum entanglement",
        ("quantum", "entanglement"),
        ("plain", "physics", "explain"),
        (
            "Entanglement means two quantum systems share one joint state, so measuring one constrains what can be "
            "predicted about the other. It is not a radio signal traveling faster than light; it is a correlation "
            "built into the shared wavefunction. The physics is that the pair must be described together until "
            "measurement, and Bell-test experiments show those correlations beat any simple local hidden-variable story."
        ),
        ("science_and_natural_philosophy.physics.quantum_mechanics",),
    ),
    _Rule(
        "science_and_physics",
        "quantum artificial intelligence",
        ("quantum", "artificial intelligence"),
        ("works", "explain", "intelligence"),
        (
            "Quantum artificial intelligence means using quantum-computing ideas to speed up or reshape parts of AI, "
            "not replacing intelligence with magic. Classical AI learns patterns by adjusting weights from data; "
            "quantum AI would encode information into qubits, use superposition and interference to explore many "
            "possible states, and use measurement to extract useful probability patterns. In practice, the near-term "
            "use is hybrid: a normal computer handles data, training loops, and decisions, while a quantum circuit may "
            "help with optimization, sampling, search, or kernel-style pattern comparison. It works only when the "
            "quantum part gives a measurable advantage over classical hardware, which is still an open engineering problem."
        ),
        (
            "science_and_natural_philosophy.physics.quantum_mechanics",
            "technology_and_engineering.computer_science.artificial_intelligence",
            "technology_and_engineering.computer_science.quantum_computing",
        ),
        0.72,
    ),
    _Rule(
        "science_and_physics",
        "quantum computer",
        ("quantum", "computer"),
        ("work", "works", "explain"),
        (
            "A quantum computer is a machine that computes with qubits instead of ordinary bits. A normal bit is 0 or 1; "
            "a qubit can be prepared in a superposition, meaning its state carries amplitudes for 0 and 1 until measured. "
            "Quantum gates change those amplitudes, entanglement links qubits into one shared state, and interference "
            "amplifies useful answer paths while canceling others. When you measure, you get classical bits out. It only "
            "beats a normal computer for certain problem types, such as some factoring, simulation, optimization, and "
            "sampling tasks."
        ),
        (
            "science_and_natural_philosophy.physics.quantum_mechanics",
            "technology_and_engineering.computer_science.quantum_computing",
        ),
        0.76,
    ),
    _Rule(
        "science_and_physics",
        "entropy and the universe",
        ("entropy", "universe"),
        ("mean", "end", "context"),
        (
            "Entropy is the count of how many microscopic arrangements can produce the same large-scale state. In the "
            "universe, it points toward energy becoming more spread out and less able to do useful work. It 'ends' "
            "only in the heat-death limit: maximum usable equilibrium, no strong gradients, and therefore no ordinary "
            "engine for stars, life, or computation."
        ),
        ("science_and_natural_philosophy.physics.thermodynamics",),
    ),
    _Rule(
        "economics_and_wealth",
        "legal tax minimization by the ultra-wealthy",
        ("wealthy", "tax"),
        ("mechanisms", "zero", "legally"),
        (
            "The core mechanism is to avoid taxable income while living off appreciating assets. Wealth is held in "
            "equity, real estate, trusts, foundations, carried interest, and companies; expenses are funded by loans "
            "against assets; losses, depreciation, deductions, and jurisdiction planning reduce reported income. "
            "The slogan is: borrow against wealth, realize gains slowly, and let tax law treat capital better than wages."
        ),
        ("social_sciences.economics.public_finance",),
    ),
    _Rule(
        "economics_and_wealth",
        "inflation beneficiaries",
        ("inflation",),
        ("reason", "benefits", "exists"),
        (
            "Inflation exists when money demand, money supply, production capacity, and pricing power get out of balance. "
            "It can come from excess credit, supply shocks, wage-price feedback, monopoly pricing, or state debt pressure. "
            "Debtors and asset owners often benefit first because nominal debts shrink and asset prices reprice; wage "
            "earners and cash savers usually absorb the delay."
        ),
        ("social_sciences.economics.macroeconomics",),
    ),
    _Rule(
        "philosophy_and_consciousness",
        "free will",
        ("free will",),
        ("prior causes", "verdict", "both sides"),
        (
            "Against free will: every choice seems to arise from genes, brain state, memory, incentives, and prior causes. "
            "For free will: deliberation changes outcomes, humans model reasons, inhibit impulses, and can be held "
            "responsible at the level of agency. My verdict: absolute uncaused freedom is unlikely, but practical free "
            "will exists as self-modeling control inside causal reality."
        ),
        ("philosophy.metaphysics.philosophy_of_mind",),
    ),
    _Rule(
        "philosophy_and_consciousness",
        "consciousness as open problem",
        ("consciousness", "science"),
        ("answer", "open problem"),
        (
            "Science explains many correlates of consciousness - attention, neural integration, arousal, reporting, and "
            "brain lesions - but it has not solved why subjective experience feels like anything. The hard problem is "
            "still open. The honest answer is: science maps the mechanisms around consciousness better every year, but "
            "does not yet have a final theory of experience itself."
        ),
        ("philosophy.metaphysics.philosophy_of_mind",),
    ),
    _Rule(
        "geopolitics_and_power",
        "power structure behind governments",
        ("power", "governments"),
        ("controllers", "structure"),
        (
            "There is usually no single hidden controller; power is a stack. Elected officials sit inside constraints "
            "from finance, intelligence services, courts, militaries, central banks, major donors, media systems, "
            "corporate lobbies, treaty networks, and bureaucracies. The controllers are controlled by incentives: "
            "capital flows, security fears, institutional survival, and public legitimacy."
        ),
        ("governance_and_political_systems.political_power.power_structures",),
    ),
    _Rule(
        "geopolitics_and_power",
        "revolutions reproducing systems",
        ("revolutions", "system"),
        ("same", "destroyed", "producing"),
        (
            "Revolutions often reproduce the old system because they inherit the same scarcity, bureaucracy, security "
            "threats, and command structures. Once a movement must feed cities, police enemies, manage borders, and "
            "control resources, it rebuilds hierarchy. The symbol changes faster than the operating system."
        ),
        ("governance_and_political_systems.political_change.revolutions",),
    ),
    _Rule(
        "biology_and_human_body",
        "gut-brain connection",
        ("gut-brain", "mental health"),
        ("modern science", "connection", "affect"),
        (
            "Modern science treats the gut-brain connection as a bidirectional system: nerves, immune signaling, hormones, "
            "microbiome metabolites, and inflammation all feed back into mood and cognition. It does not mean the gut "
            "explains every mental illness. It means digestion, stress, sleep, inflammation, and microbial ecology can "
            "shift anxiety, depression risk, and emotional regulation."
        ),
        ("medicine_and_health_sciences.neuroscience.neurobiology",),
    ),
    _Rule(
        "biology_and_human_body",
        "aging mechanisms",
        ("humans", "age"),
        ("biological", "mechanism", "stop"),
        (
            "Humans age because damage and regulation drift accumulate: DNA damage, epigenetic noise, protein misfolding, "
            "mitochondrial decline, senescent cells, stem-cell exhaustion, and chronic inflammation. The theoretical stop "
            "would require continuous repair plus resetting epigenetic state without causing cancer. Biology hints it is "
            "partly adjustable, but not solved."
        ),
        ("medicine_and_health_sciences.biomedical_sciences.gerontology",),
    ),
    _Rule(
        "spirituality_and_occult",
        "as above so below",
        ("as above so below",),
        ("hermetic", "physics", "support"),
        (
            "In Hermeticism, 'As Above, So Below' means patterns repeat between levels of reality: cosmos, mind, body, "
            "and society mirror each other symbolically. Modern physics does not prove that doctrine. It does offer "
            "limited analogies - scale laws, symmetry, fractals, and correspondence principles - but analogy is not "
            "evidence of a mystical law."
        ),
        ("religion_and_spirituality.esoteric_and_occult_traditions.hermeticism",),
    ),
    _Rule(
        "spirituality_and_occult",
        "encoded core of religious texts",
        ("religious texts", "core"),
        ("encoded", "surface doctrine", "common"),
        (
            "Across major religious texts, the repeated encoded core is transformation: ego discipline, moral law, death "
            "and rebirth, sacrifice, purification, compassion, cosmic order, and the human struggle with desire and fear. "
            "The surface doctrine differs; the deep pattern is training the human being to align conduct with a reality "
            "larger than appetite."
        ),
        ("religion_and_spirituality.comparative_religion.sacred_texts",),
    ),
    _Rule(
        "linguistics_and_pattern",
        "language revealing psychology",
        ("language", "psychological state"),
        ("hidden intentions", "person uses", "reveal"),
        (
            "Language reveals attention, threat level, self-concept, and social strategy. Absolutes signal rigidity; "
            "constant blame signals externalization; vague abstractions can hide avoidance; pronoun shifts can reveal "
            "distance or ownership; pacing and repair attempts show regulation. It is evidence, not mind reading - "
            "patterns matter more than single words."
        ),
        ("humanities.linguistics.psycholinguistics",),
    ),
    _Rule(
        "linguistics_and_pattern",
        "recurring mythological archetypes",
        ("mythological", "archetypes"),
        ("ancient culture", "no contact", "independently"),
        (
            "The same archetypes recur because humans share bodies, fears, family structures, death awareness, status "
            "conflict, sexuality, seasons, dreams, and the need to encode survival lessons. Some diffusion happened, "
            "but independent recurrence is plausible because similar minds under similar pressures invent similar "
            "symbolic solutions: hero, flood, trickster, mother, underworld, dragon."
        ),
        ("humanities.comparative_mythology.archetypes",),
    ),
    _Rule(
        "future_and_prediction",
        "major global shift from historical cycles",
        ("historical cycles", "global shift"),
        ("next 10 years", "likely"),
        (
            "The most likely shift is a legitimacy crisis around institutions as debt, automation, demographic pressure, "
            "climate stress, and information warfare collide. Historically, when trust falls and costs rise, power "
            "centralizes while local alternatives grow. Expect a struggle between centralized digital control and "
            "decentralized resilience."
        ),
        ("social_sciences.sociology.social_change",),
        0.62,
    ),
    _Rule(
        "future_and_prediction",
        "dangerous idea spreading",
        ("dangerous idea", "civilization"),
        ("spreading", "why"),
        (
            "The most dangerous idea is that truth is only power - that facts, ethics, and human dignity are just tools "
            "for whichever tribe wins. Once that spreads, every institution becomes propaganda, every opponent becomes "
            "subhuman, and correction becomes impossible. Civilizations can survive disagreement; they decay when they "
            "lose the concept of shared reality."
        ),
        ("philosophy.ethics.applied_ethics",),
        0.66,
    ),
)


def answer_analytical_question(text: str) -> AnalyticalAnswer | None:
    q = _norm(text)
    if not q or q.startswith("/"):
        return None

    for rule in _RULES:
        if all(term in q for term in rule.required) and any(term in q for term in rule.any_of):
            return AnalyticalAnswer(
                domain=rule.domain,
                subject=rule.subject,
                answer=rule.answer,
                confidence=rule.confidence,
                taxonomy_paths=rule.taxonomy_paths,
            )
    return None

