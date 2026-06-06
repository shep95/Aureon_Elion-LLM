"""Capability roadmap — what Aureon is building and the path beyond frontier LLMs."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Literal

Status = Literal["live", "partial", "planned", "research"]

# Aureon wins on verifiable grounding + continuous domain mastery, not raw param count.
CAPABILITIES: tuple[dict[str, Any], ...] = (
    {
        "id": "grade_ladder",
        "name": "862-topic grade ladder",
        "phase": 1,
        "status": "live",
        "frontier_gap": "Frontier models are static weights; Aureon graduates preschool→doctorate per micro-topic.",
    },
    {
        "id": "brain_regions",
        "name": "Six brain regions",
        "phase": 1,
        "status": "live",
        "frontier_gap": "Collect → verify → label → train → evaluate → reward — measurable loop, not black-box chat.",
    },
    {
        "id": "auto_learn_24_7",
        "name": "Continuous auto-learn",
        "phase": 1,
        "status": "live",
        "frontier_gap": "Corpus grows while you sleep; GitHub sync + predict-brain retrain after each cycle.",
    },
    {
        "id": "deterministic_evaluators",
        "name": "Deterministic evaluators (math/logic)",
        "phase": 1,
        "status": "live",
        "frontier_gap": "Exact answers for arithmetic before any neural guess — zero hallucination on 2+2.",
    },
    {
        "id": "predict_brain",
        "name": "Attention predict brain",
        "phase": 1,
        "status": "partial",
        "frontier_gap": "1M context config; FFN backprop live; full attention grads still planned.",
    },
    {
        "id": "ciper_taxonomy",
        "name": "Ciper cross-domain research",
        "phase": 1,
        "status": "live",
        "frontier_gap": "Facet drill-down across 862 topics when corpus supports an answer.",
    },
    {
        "id": "label_review",
        "name": "Human label review API",
        "phase": 1,
        "status": "live",
        "frontier_gap": "Teacher-model flags uncertain labels; humans approve before training.",
    },
    {
        "id": "vector_rag_1m",
        "name": "Vector RAG at 1M scale",
        "phase": 2,
        "status": "live",
        "frontier_gap": "TF-IDF vector retrieval over full corpus with citation metadata.",
    },
    {
        "id": "domain_moe_routing",
        "name": "Per-domain classifier routing",
        "phase": 2,
        "status": "partial",
        "frontier_gap": "MoE chat routing live; needs more promoted per-scope brain models.",
    },
    {
        "id": "full_lm_backprop",
        "name": "Full attention LM backprop",
        "phase": 2,
        "status": "partial",
        "frontier_gap": "FFN + embeddings + head train; attention weight grads still planned.",
    },
    {
        "id": "rlhf_reward_loop",
        "name": "RLHF reward loop wired to chat",
        "phase": 3,
        "status": "live",
        "frontier_gap": "Reward model scores every chat reply; high-quality pairs stored for retraining.",
    },
    {
        "id": "benchmark_harness",
        "name": "Frontier benchmark harness",
        "phase": 3,
        "status": "live",
        "frontier_gap": "Fixed Q&A suite vs illustrative frontier baseline — GET /api/brain/benchmark.",
    },
    {
        "id": "verified_citations",
        "name": "Verified citations in answers",
        "phase": 3,
        "status": "partial",
        "frontier_gap": "Predict brain returns document_id + content_hash citations from RAG.",
    },
    {
        "id": "confidence_abstain",
        "name": "Calibrated confidence + abstain",
        "phase": 3,
        "status": "live",
        "frontier_gap": "Says 'I don't know' when corpus + confidence can't ground an answer.",
    },
    {
        "id": "agent_tool_loop",
        "name": "Tool-use agent loop",
        "phase": 4,
        "status": "live",
        "frontier_gap": "Multi-step rag_search → calculate → classify → predict → verify gates.",
    },
    {
        "id": "efficient_inference",
        "name": "Efficient 1M inference",
        "phase": 4,
        "status": "live",
        "frontier_gap": "Sliding-window sparse attention + speculative decode + token truncation.",
    },
    {
        "id": "multimodal",
        "name": "Multimodal collectors",
        "phase": 5,
        "status": "live",
        "frontier_gap": "Image/audio sidecars + JSON manifests feed the same grade ladder.",
    },
    {
        "id": "meta_consciousness",
        "name": "Meta-consciousness self-inquiry",
        "phase": 5,
        "status": "live",
        "frontier_gap": "Grounded identity/meta-cognition questions after each batch — honest self-model, not theater.",
    },
)


@dataclass(frozen=True)
class FuturePhase:
    phase: int
    name: str
    horizon: str
    unlocks: tuple[str, ...]
    beats_frontier_on: tuple[str, ...]


FUTURE_PHASES: tuple[FuturePhase, ...] = (
    FuturePhase(
        phase=1,
        name="Grounded core",
        horizon="now",
        unlocks=("deterministic_evaluators", "predict_brain", "ciper_taxonomy", "label_review"),
        beats_frontier_on=("hallucination-free math", "auditable learning logs", "domain graduation"),
    ),
    FuturePhase(
        phase=2,
        name="Retrieval + specialists",
        horizon="next",
        unlocks=("vector_rag_1m", "domain_moe_routing", "full_lm_backprop"),
        beats_frontier_on=("factual Q&A accuracy", "domain-specific tasks", "corpus-grounded reasoning"),
    ),
    FuturePhase(
        phase=3,
        name="Self-improving quality",
        horizon="mid",
        unlocks=("rlhf_reward_loop", "benchmark_harness", "verified_citations", "confidence_abstain"),
        beats_frontier_on=("trustworthiness", "measurable improvement", "citation-backed answers"),
    ),
    FuturePhase(
        phase=4,
        name="Agentic + efficient",
        horizon="long",
        unlocks=("agent_tool_loop", "efficient_inference"),
        beats_frontier_on=("multi-step tasks", "1M context at practical latency", "cost per correct answer"),
    ),
    FuturePhase(
        phase=5,
        name="Full sensory brain",
        horizon="frontier",
        unlocks=("multimodal",),
        beats_frontier_on=("unified supervised brain across modalities", "continuous learning on all inputs"),
    ),
)


_VISION_QUESTIONS = (
    "what are you building",
    "what is aureon building",
    "what is your roadmap",
    "roadmap",
    "capabilities",
    "what can you do",
    "how are you better",
    "better than gpt",
    "better than claude",
    "beat gpt",
    "beat claude",
    "frontier models",
    "future plans",
    "what do you need",
    "what is missing",
)


def _status_counts() -> dict[str, int]:
    counts: dict[str, int] = {"live": 0, "partial": 0, "planned": 0, "research": 0}
    for cap in CAPABILITIES:
        counts[str(cap["status"])] = counts.get(str(cap["status"]), 0) + 1
    return counts


def roadmap_snapshot() -> dict[str, Any]:
    """Full capability matrix for API and chat."""
    counts = _status_counts()
    return {
        "vision": (
            "Supervised learning brain — not a static chatbot. "
            "862 micro-topics, grade ladder, verifiable corpus, continuous retrain. "
            "Beats frontier LLMs on grounding, auditability, and domain mastery — not raw chit-chat."
        ),
        "architecture": "collector → verifier → labeler → trainer → evaluator → reward",
        "context_window": 1_000_000,
        "micro_topics": 862,
        "status_counts": counts,
        "completion_pct": round(100 * (counts["live"] + 0.5 * counts["partial"]) / len(CAPABILITIES), 1),
        "capabilities": list(CAPABILITIES),
        "future_phases": [asdict(p) for p in FUTURE_PHASES],
        "simulated_at": datetime.now(timezone.utc).isoformat(),
    }


def simulate_future_timeline(*, months_ahead: int = 12) -> dict[str, Any]:
    """Project when planned capabilities unlock if auto-learn stays continuous."""
    months_ahead = max(1, min(months_ahead, 36))
    milestones: list[dict[str, Any]] = []
    for phase in FUTURE_PHASES:
        month = {1: 0, 2: 2, 3: 5, 4: 9, 5: months_ahead}[phase.phase]
        if month <= months_ahead:
            milestones.append(
                {
                    "month": month,
                    "phase": phase.phase,
                    "name": phase.name,
                    "unlocks": list(phase.unlocks),
                    "beats_frontier_on": list(phase.beats_frontier_on),
                }
            )
    return {
        "months_ahead": months_ahead,
        "assumption": "Continuous auto-learn + corpus growth on Railway",
        "milestones": milestones,
        "next_unlock": milestones[0] if milestones else None,
    }


def try_roadmap_answer(text: str) -> str | None:
    """Short answer when user asks about vision, roadmap, or beating frontier models."""
    q = re.sub(r"\s+", " ", (text or "").strip().lower().rstrip("?"))
    if not q:
        return None

    if any(token in q for token in ("better than gpt", "better than claude", "beat gpt", "beat claude")):
        return (
            "On grounding and auditability — yes long-term. "
            "Frontier models win on fluent general chat today; "
            "I win when answers must be corpus-verified, graduated, and retrainable across 862 domains."
        )

    if any(token in q for token in _VISION_QUESTIONS):
        snap = roadmap_snapshot()
        live = snap["status_counts"]["live"]
        partial = snap["status_counts"]["partial"]
        planned = snap["status_counts"]["planned"]
        return (
            f"Building a supervised brain — {snap['micro_topics']} micro-topics, 1M context, "
            f"grade ladder, continuous retrain. "
            f"Live: {live}, partial: {partial}, planned: {planned}. "
            f"Beats frontier models on grounded facts and auditability — not generic chat yet."
        )

    if q in ("what makes you different", "why aureon", "why not use chatgpt"):
        return (
            "Frontier LLMs are frozen weights — I collect, label, train, evaluate, and graduate "
            "862 topics 24/7 with measurable accuracy and corpus citations."
        )

    return None
