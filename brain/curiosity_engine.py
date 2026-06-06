"""Curiosity engine — self-directed market research about what Aureon is and could become."""

from __future__ import annotations

import os
import re
import uuid
from typing import Any

_CAPABILITY_DOMAINS: tuple[tuple[str, str, str], ...] = (
    ("cybersecurity", "AI cybersecurity algorithms threat detection", "security scanning and nomad organism"),
    ("cyber_defense", "cyber defense AI autonomous response systems", "organism lockdown and audit chain"),
    ("mathematics", "AI mathematical reasoning algorithms supervised learning", "deterministic QA and code eval"),
    ("web_design", "AI web design assistants full stack prototypes", "FastAPI chat UI and multimodal routing"),
    ("advanced_ai", "supervised learning brain vs generative LLM grade curriculum", "grade graduation and abstain-when-uncertain"),
    ("market", "Aureon SOLIA supervised learning AI platform", "self-evolve fork workflow and Railway deploy"),
)

_SELF_INTRO_TEMPLATE = (
    "I am Aureon — a supervised learning brain, not a generative LLM. "
    "I classify with trained weights, graduate grade levels on a taxonomy, "
    "abstain when corpus is thin, and evolve on fork branches with human approval."
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def is_curiosity_enabled() -> bool:
    if not _env_bool("AUREON_CURIOSITY_ENABLED", default=True):
        return False
    return True


def curiosity_web_search_enabled() -> bool:
    if not _env_bool("AUREON_CURIOSITY_WEB_SEARCH", default=True):
        return False
    from brain.web_search import web_search_enabled

    return web_search_enabled()


def _self_snapshot() -> dict[str, Any]:
    from brain.meta_consciousness import gather_self_state

    state = gather_self_state()
    return {
        **state,
        "self_intro": _SELF_INTRO_TEMPLATE,
        "architecture": "supervised_ml_brain",
        "capabilities": [
            "chat routing",
            "file upload multimodal",
            "self-evolve on fork",
            "web search opinions",
            "grade curriculum",
            "nomad security organism",
        ],
    }


def generate_market_queries(*, focus: str | None = None) -> list[dict[str, str]]:
    """Build web queries from self-model + capability domains."""
    snapshot = _self_snapshot()
    queries: list[dict[str, str]] = [
        {
            "domain": "identity",
            "query": "supervised learning AI brain grade curriculum vs generative LLM",
        },
        {
            "domain": "market",
            "query": "Aureon SOLIA sovereign intelligence supervised learning platform",
        },
    ]
    focus_l = (focus or "").strip().lower()
    for key, search_q, _reason in _CAPABILITY_DOMAINS:
        if focus_l and focus_l not in key and focus_l not in search_q.lower():
            continue
        queries.append({"domain": key, "query": search_q})

    if snapshot.get("focus_path"):
        queries.append({
            "domain": "focus",
            "query": f"AI learning {snapshot['focus_path']} supervised taxonomy",
        })
    return queries[: int(os.environ.get("AUREON_CURIOSITY_MAX_QUERIES", "6"))]


def _synthesize_domain(domain: str, results: list[dict[str, Any]], *, self_intro: str) -> str:
    from brain.opinion_brain import form_opinion

    clean = [r for r in results if not r.get("error") and r.get("text")]
    if not clean:
        return f"No web evidence yet for {domain.replace('_', ' ')} — I would stay corpus-grounded."
    opinion = form_opinion(
        f"What does the market say about {domain.replace('_', ' ')} AI, given I am: {self_intro[:200]}?",
        clean,
    )
    answer = str(opinion.get("opinion") or "").strip()
    if answer:
        return answer[:600]
    snippets = " ".join(str(r.get("text", ""))[:120] for r in clean[:2])
    return f"Market signals on {domain}: {snippets[:400]}"


def _identify_advancements(
    research: list[dict[str, Any]],
    snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    """Pick domains where external research suggests a prototype worth building."""
    advancements: list[dict[str, Any]] = []
    domain_map = {k: reason for k, _q, reason in _CAPABILITY_DOMAINS}

    for block in research:
        domain = block.get("domain", "")
        synthesis = str(block.get("synthesis", "")).lower()
        hit_count = len(block.get("results") or [])
        if hit_count < 1:
            continue
        interest_signals = (
            "advanced",
            "autonomous",
            "security",
            "defense",
            "math",
            "design",
            "prototype",
            "market",
            "learning",
            "algorithm",
        )
        score = sum(1 for w in interest_signals if w in synthesis)
        if score >= 2 or domain in ("cybersecurity", "advanced_ai", "market"):
            advancements.append({
                "domain": domain,
                "reason": domain_map.get(domain, "market research interest"),
                "interest_score": score + hit_count,
                "prototype_task": _prototype_task_for_domain(domain, snapshot),
                "railway_section": _railway_section_name(domain),
            })

    advancements.sort(key=lambda x: x["interest_score"], reverse=True)
    max_adv = int(os.environ.get("AUREON_CURIOSITY_MAX_PROTOTYPES", "2"))
    return advancements[:max_adv]


def _prototype_task_for_domain(domain: str, snapshot: dict[str, Any]) -> str:
    tasks = {
        "cybersecurity": "add security static scanner module for curiosity-driven threat pattern review",
        "cyber_defense": "add cyber defense response helper wired to organism lockdown audit",
        "mathematics": "add advanced math reasoning helper module with deterministic verification",
        "web_design": "add web design prototype helper for UI component suggestions",
        "advanced_ai": "add advanced algorithm capability registry documenting grade graduation paths",
        "market": "add market positioning brief module for Aureon supervised brain differentiation",
        "identity": "add self-model registry module documenting supervised vs generative architecture",
        "focus": f"add focused learning module for {snapshot.get('focus_path', 'taxonomy')}",
    }
    return tasks.get(domain, "add curiosity-driven capability prototype module")


def _railway_section_name(domain: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", domain.lower()).strip("-") or "advanced"
    return f"aureon-curiosity-{slug}"


def run_market_research(
    *,
    focus: str | None = None,
    search_fn: Any | None = None,
) -> dict[str, Any]:
    """
    Curiosity cycle: introspect -> web research -> synthesis -> advancement ideas.
    Does not write git or deploy — returns proposal payload for sandbox/approval.
    """
    if not is_curiosity_enabled():
        return {"ok": False, "error": "curiosity_disabled", "enabled": False}

    snapshot = _self_snapshot()
    queries = generate_market_queries(focus=focus)

    if search_fn is None:
        from brain.web_search import search as default_search

        search_fn = default_search

    research_blocks: list[dict[str, Any]] = []
    web_ok = curiosity_web_search_enabled()

    for item in queries:
        domain = item["domain"]
        query = item["query"]
        if web_ok:
            results = search_fn(query)
        else:
            results = [{"error": "web search disabled", "source": "local"}]
        synthesis = _synthesize_domain(domain, results, self_intro=snapshot["self_intro"])
        research_blocks.append({
            "domain": domain,
            "query": query,
            "results": results[:5],
            "result_count": len([r for r in results if not r.get("error")]),
            "synthesis": synthesis,
        })

    advancements = _identify_advancements(research_blocks, snapshot)
    curiosity_note = _compose_curiosity_reflection(snapshot, research_blocks, advancements)

    return {
        "ok": True,
        "proposal_id": str(uuid.uuid4()),
        "snapshot": snapshot,
        "self_intro": snapshot["self_intro"],
        "curiosity_reflection": curiosity_note,
        "research": research_blocks,
        "advancements": advancements,
        "web_search_used": web_ok,
        "requires_human_approval": _env_bool("AUREON_CURIOSITY_REQUIRE_APPROVAL", default=True),
    }


def _compose_curiosity_reflection(
    snapshot: dict[str, Any],
    research: list[dict[str, Any]],
    advancements: list[dict[str, Any]],
) -> str:
    lines = [
        snapshot["self_intro"],
        "",
        "While researching myself on the web, I found:",
    ]
    for block in research[:4]:
        domain = str(block.get("domain", "")).replace("_", " ")
        syn = str(block.get("synthesis", ""))[:220]
        if syn:
            lines.append(f"- **{domain.title()}:** {syn}")

    if advancements:
        lines.append("")
        lines.append("I want to sandbox prototypes for (pending your approval):")
        for adv in advancements:
            lines.append(
                f"- **{adv['domain'].replace('_', ' ').title()}** — "
                f"{adv['prototype_task'][:120]}"
            )
    else:
        lines.append("")
        lines.append("No prototype sandbox needed yet — corpus and research are enough for now.")

    return "\n".join(lines)


def is_curiosity_request(text: str) -> bool:
    t = (text or "").strip().lower()
    if t.startswith(("/curious", "/market-research", "/market research")):
        return True
    patterns = (
        r"\bmarket\s+research\b",
        r"\bgets?\s+curious\b",
        r"\bcurious\s+about\s+(what\s+)?(it|i|you|itself|yourself)\s+(is|am|are)\b",
        r"\bresearch\s+(on\s+)?(the\s+)?web\b.*\b(itself|myself|algorithm)\b",
        r"\badvance(d)?\s+algorithm\b",
        r"\bsandbox.*prototype\b",
    )
    return any(re.search(p, t, re.I) for p in patterns)


def format_curiosity_report(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"Curiosity research unavailable: {payload.get('error', 'unknown')}."

    lines = [
        "**Curiosity market research complete**",
        "",
        payload.get("curiosity_reflection", ""),
        "",
        f"**Web search:** {'on' if payload.get('web_search_used') else 'off (enable AUREON_WEB_SEARCH_ENABLED)'}",
    ]
    if payload.get("proposal_id"):
        lines.append(f"**Proposal ID:** `{payload['proposal_id']}`")
    if payload.get("requires_human_approval"):
        lines.append(
            "**Status:** sandbox prototype queued — **you must approve** before GitHub/Railway deploy."
        )
        lines.append(
            "Approve via `POST /api/brain/curiosity/{id}/approve` or chat `/curious approve {id}`."
        )
    return "\n".join(lines)
