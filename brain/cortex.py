"""Cortex — coordinates all brain regions across every knowledge domain."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

from sqlalchemy import func, select

from brain.base import AgentContext
from brain.domains.taxonomy import (
    KNOWLEDGE_TAXONOMY,
    micro_subdomains_for,
    subdomains_for,
    total_micro_subdomains,
    total_subdomains,
)
from brain.grades import get_grade, grade_slugs
from brain.graduation import (
    current_grade,
    mark_grade_in_progress,
    process_graduation,
    progress_report,
    require_grade_unlocked,
)
from brain.regions.collector import CollectorAgent
from brain.regions.evaluator import EvaluatorAgent
from brain.regions.labeler import LabelerAgent
from brain.regions.reward import RewardAgent
from brain.regions.trainer import TrainerAgent
from brain.regions.verifier import VerifierAgent
from db.models import Document, GradeProgress, KnowledgeDomain, KnowledgeMicroSubdomain, KnowledgeSubdomain, MicroAgent
from db.seed import seed_knowledge_taxonomy
from db.session import get_session, init_db

REGION_ORDER = (
    ("collector", CollectorAgent),
    ("verifier", VerifierAgent),
    ("labeler", LabelerAgent),
    ("trainer", TrainerAgent),
    ("evaluator", EvaluatorAgent),
    ("reward", RewardAgent),
)


def bootstrap_brain() -> dict[str, int]:
    """Create tables and seed domains, subdomains, micro-subdomains, and micro-agents."""
    from app.activity_log import log_ai_activity

    log_ai_activity("bootstrap_start")
    init_db()
    with get_session() as session:
        stats = seed_knowledge_taxonomy(session)
    log_ai_activity("bootstrap_complete", stats=stats)
    return stats


def _run_region_cycle(
    session,
    domain,
    subdomain,
    micro,
    ctx: AgentContext,
) -> list[dict]:
    results: list[dict] = []
    for region_name, agent_cls in REGION_ORDER:
        agent_row = session.scalar(
            select(MicroAgent).where(
                MicroAgent.region == region_name,
                MicroAgent.domain_id == domain.id,
                MicroAgent.subdomain_id == (subdomain.id if subdomain else None),
                MicroAgent.micro_subdomain_id == (micro.id if micro else None),
            )
        )
        if not agent_row:
            continue
        agent_impl = agent_cls()
        result = agent_impl.execute(session, agent_row, ctx)
        results.append(
            {
                "region": region_name,
                "status": result.status,
                "metrics": result.metrics,
                "error": result.error,
            }
        )
    return results


def run_grade_cycle(
    domain_slug: str,
    subdomain_slug: str,
    micro_subdomain_slug: str,
    grade_slug: str | None = None,
    epochs: int = 200,
    *,
    source: str = "internal",
) -> dict[str, Any]:
    """Run the 6 brain regions at a specific academic grade level."""
    from app.activity_log import log_ai_activity

    bootstrap_brain()

    with get_session() as session:
        domain = session.scalar(
            select(KnowledgeDomain).where(KnowledgeDomain.slug == domain_slug)
        )
        if not domain:
            return {"error": f"unknown domain: {domain_slug}"}

        subdomain = session.scalar(
            select(KnowledgeSubdomain).where(
                KnowledgeSubdomain.domain_id == domain.id,
                KnowledgeSubdomain.slug == subdomain_slug,
            )
        )
        if not subdomain:
            return {"error": f"unknown subdomain: {subdomain_slug}"}

        micro = session.scalar(
            select(KnowledgeMicroSubdomain).where(
                KnowledgeMicroSubdomain.subdomain_id == subdomain.id,
                KnowledgeMicroSubdomain.slug == micro_subdomain_slug,
            )
        )
        if not micro:
            return {"error": f"unknown micro_subdomain: {micro_subdomain_slug}"}

        if grade_slug is None:
            row = current_grade(session, micro.id)
            if not row:
                return {
                    "domain": domain_slug,
                    "subdomain": subdomain_slug,
                    "micro_subdomain": micro_subdomain_slug,
                    "fully_graduated": True,
                    "progress": progress_report(session, micro.id),
                }
            grade_slug = row.grade_slug

        try:
            grade = require_grade_unlocked(session, micro.id, grade_slug)
        except ValueError as exc:
            return {"error": str(exc)}

        mark_grade_in_progress(session, micro.id, grade_slug)
        session.commit()

        log_ai_activity(
            "grade_cycle_start",
            source=source,
            epochs=epochs,
            domain=domain_slug,
            subdomain=subdomain_slug,
            micro_subdomain=micro_subdomain_slug,
            grade=grade_slug,
            path=f"{domain_slug}.{subdomain_slug}.{micro_subdomain_slug}",
        )

        ctx = AgentContext(
            domain_slug=domain_slug,
            subdomain_slug=subdomain_slug,
            micro_subdomain_slug=micro_subdomain_slug,
            domain_id=domain.id,
            subdomain_id=subdomain.id,
            micro_subdomain_id=micro.id,
            epochs=epochs,
            grade_slug=grade.slug,
            grade=grade,
            extra={"source": source},
        )
        results = _run_region_cycle(session, domain, subdomain, micro, ctx)

        trainer_row = next((r for r in results if r["region"] == "trainer"), {})
        evaluator_row = next((r for r in results if r["region"] == "evaluator"), {})
        trainer_payload = {**trainer_row.get("metrics", {}), "status": trainer_row.get("status")}
        evaluator_payload = {**evaluator_row.get("metrics", {}), "status": evaluator_row.get("status")}
        graduation = process_graduation(
            session,
            micro.id,
            grade_slug,
            trainer_metrics=trainer_payload,
            evaluator_metrics=evaluator_payload,
        )
        report = progress_report(session, micro.id)
        session.commit()

    outcome = {
        "domain": domain_slug,
        "subdomain": subdomain_slug,
        "micro_subdomain": micro_subdomain_slug,
        "grade": grade_slug,
        "grade_name": grade.name,
        "regions": results,
        "graduation": graduation,
        "progress": report,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    log_ai_activity(
        "grade_cycle_complete",
        source=source,
        status="graduated" if graduation.get("passed") else "attempted",
        domain=domain_slug,
        subdomain=subdomain_slug,
        micro_subdomain=micro_subdomain_slug,
        grade=grade_slug,
        path=f"{domain_slug}.{subdomain_slug}.{micro_subdomain_slug}",
        graduation=graduation,
        regions_summary=[{"region": r["region"], "status": r["status"]} for r in results],
    )
    from brain.self_inquiry import run_self_inquiry_for_cycle

    inquiry = run_self_inquiry_for_cycle(outcome, source=source)
    if inquiry:
        outcome["self_inquiry"] = inquiry
    return outcome


def run_graduation_ladder(
    domain_slug: str,
    subdomain_slug: str,
    micro_subdomain_slug: str,
    epochs: int = 200,
    max_grades: int | None = None,
    *,
    source: str = "internal",
) -> dict[str, Any]:
    """Advance through grade levels until failure or doctorate graduation."""
    from app.activity_log import clear_cycle_id, log_ai_activity, new_cycle_id

    bootstrap_brain()
    cycle_id = new_cycle_id("ladder")
    path = f"{domain_slug}.{subdomain_slug}.{micro_subdomain_slug}"
    log_ai_activity(
        "graduation_ladder_start",
        cycle_id=cycle_id,
        source=source,
        path=path,
        max_grades=max_grades,
        epochs=epochs,
    )
    ladder: list[dict] = []
    steps = 0

    try:
        while True:
            if max_grades is not None and steps >= max_grades:
                break
            step = run_grade_cycle(
                domain_slug,
                subdomain_slug,
                micro_subdomain_slug,
                grade_slug=None,
                epochs=epochs,
                source=source,
            )
            ladder.append(step)
            steps += 1

            if step.get("error") or step.get("fully_graduated"):
                break
            graduation = step.get("graduation", {})
            if not graduation.get("passed"):
                break
            if graduation.get("fully_graduated"):
                break
    finally:
        log_ai_activity(
            "graduation_ladder_complete",
            cycle_id=cycle_id,
            source=source,
            path=path,
            steps_completed=steps,
            last_graduation=ladder[-1].get("graduation") if ladder else {},
        )
        clear_cycle_id()

    return {
        "domain": domain_slug,
        "subdomain": subdomain_slug,
        "micro_subdomain": micro_subdomain_slug,
        "steps_completed": steps,
        "ladder": ladder,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def iter_training_targets(
    *,
    domain_limit: int | None = None,
    subdomain_limit: int | None = None,
    micro_subdomain_limit: int | None = None,
    domain_slugs: list[str] | None = None,
) -> list[tuple[str, str, str]]:
    """Expand domain/subdomain/micro limits into (domain, subdomain, micro) triples."""
    from app.security import (
        clamp_domain_limit,
        clamp_micro_subdomain_limit,
        clamp_subdomain_limit,
    )

    domains = domain_slugs if domain_slugs is not None else list(KNOWLEDGE_TAXONOMY.keys())
    if domain_limit is not None:
        domains = domains[: clamp_domain_limit(domain_limit) or len(domains)]

    targets: list[tuple[str, str, str]] = []
    for domain_slug in domains:
        if domain_slug not in KNOWLEDGE_TAXONOMY:
            continue
        subs = subdomains_for(domain_slug)
        if subdomain_limit is not None:
            subs = subs[: clamp_subdomain_limit(subdomain_limit) or len(subs)]
        for sub in subs:
            micros = micro_subdomains_for(domain_slug, sub)
            if micro_subdomain_limit is not None:
                micros = micros[: clamp_micro_subdomain_limit(micro_subdomain_limit) or len(micros)]
            for micro in micros:
                targets.append((domain_slug, sub, micro))
    return targets


def run_batch_graduation_ladder(
    *,
    epochs: int = 150,
    max_grades: int | None = 1,
    domain_limit: int | None = None,
    subdomain_limit: int | None = None,
    micro_subdomain_limit: int | None = None,
    domain_slugs: list[str] | None = None,
    targets: list[tuple[str, str, str]] | None = None,
    source: str = "auto_learn",
) -> dict[str, Any]:
    """Run graduation ladders across many domain/subdomain/micro targets in one batch."""
    from app.activity_log import clear_cycle_id, log_ai_activity, new_cycle_id
    from brain.meta_consciousness import reset_batch_meta_budget
    from brain.self_inquiry import reset_batch_inquiry_budget

    bootstrap_brain()
    reset_batch_inquiry_budget()
    reset_batch_meta_budget()
    if targets is None:
        targets = iter_training_targets(
            domain_limit=domain_limit,
            subdomain_limit=subdomain_limit,
            micro_subdomain_limit=micro_subdomain_limit,
            domain_slugs=domain_slugs,
        )
    cycle_id = new_cycle_id("batch")
    log_ai_activity(
        "batch_graduation_start",
        cycle_id=cycle_id,
        source=source,
        targets=len(targets),
        domain_limit=domain_limit,
        subdomain_limit=subdomain_limit,
        micro_subdomain_limit=micro_subdomain_limit,
        max_grades=max_grades,
        epochs=epochs,
    )

    results: list[dict[str, Any]] = []
    try:
        for domain_slug, subdomain_slug, micro_slug in targets:
            path = f"{domain_slug}.{subdomain_slug}.{micro_slug}"
            log_ai_activity(
                "batch_graduation_target_start",
                cycle_id=cycle_id,
                source=source,
                path=path,
            )
            ladder_result = run_graduation_ladder(
                domain_slug,
                subdomain_slug,
                micro_slug,
                epochs=epochs,
                max_grades=max_grades,
                source=source,
            )
            graduation = (
                ladder_result.get("ladder", [{}])[-1].get("graduation")
                if ladder_result.get("ladder")
                else {}
            )
            results.append(
                {
                    "target": {
                        "domain": domain_slug,
                        "subdomain": subdomain_slug,
                        "micro_subdomain": micro_slug,
                    },
                    "path": path,
                    "steps": ladder_result.get("steps_completed", 0),
                    "graduation": graduation,
                }
            )
    finally:
        log_ai_activity(
            "batch_graduation_complete",
            cycle_id=cycle_id,
            source=source,
            targets_processed=len(results),
            targets_total=len(targets),
        )
        clear_cycle_id()

    return {
        "targets_total": len(targets),
        "targets_processed": len(results),
        "results": results,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def run_micro_subdomain_cycle(
    domain_slug: str,
    subdomain_slug: str,
    micro_subdomain_slug: str,
    epochs: int = 200,
    grade_slug: str | None = None,
) -> dict[str, Any]:
    """Run brain circuit at current (or specified) grade level."""
    return run_grade_cycle(
        domain_slug,
        subdomain_slug,
        micro_subdomain_slug,
        grade_slug=grade_slug,
        epochs=epochs,
    )


def run_subdomain_cycle(
    domain_slug: str,
    subdomain_slug: str,
    epochs: int = 200,
    micro_subdomain_limit: int | None = None,
) -> dict[str, Any]:
    """Run all micro-subdomain cycles under one subdomain."""
    from app.security import clamp_micro_subdomain_limit

    if domain_slug not in KNOWLEDGE_TAXONOMY:
        return {"error": f"unknown domain: {domain_slug}"}
    if subdomain_slug not in KNOWLEDGE_TAXONOMY[domain_slug]:
        return {"error": f"unknown subdomain: {subdomain_slug}"}

    micros = micro_subdomains_for(domain_slug, subdomain_slug)
    if micro_subdomain_limit:
        micros = micros[: clamp_micro_subdomain_limit(micro_subdomain_limit)]

    cycles = [
        run_micro_subdomain_cycle(domain_slug, subdomain_slug, micro, epochs=epochs)
        for micro in micros
    ]
    return {
        "domain": domain_slug,
        "subdomain": subdomain_slug,
        "micro_subdomains_processed": len(cycles),
        "cycles": cycles,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def run_domain_cycle(
    domain_slug: str,
    epochs: int = 200,
    subdomain_limit: int = 5,
    micro_subdomain_limit: int | None = 1,
) -> dict[str, Any]:
    """Run subdomains (and their micro-subdomains) within one knowledge domain."""
    from app.security import clamp_subdomain_limit

    if domain_slug not in KNOWLEDGE_TAXONOMY:
        return {"error": f"unknown domain: {domain_slug}"}
    subs = subdomains_for(domain_slug)[: clamp_subdomain_limit(subdomain_limit) or 5]
    cycles = [
        run_subdomain_cycle(
            domain_slug,
            sub,
            epochs=epochs,
            micro_subdomain_limit=micro_subdomain_limit,
        )
        for sub in subs
    ]
    return {
        "domain": domain_slug,
        "subdomains_processed": len(cycles),
        "cycles": cycles,
    }


def run_full_brain(
    epochs: int = 200,
    domain_limit: int | None = None,
    subdomain_limit: int | None = 1,
    micro_subdomain_limit: int | None = 1,
    *,
    source: str = "internal",
) -> dict[str, Any]:
    """
    Train across human knowledge domains → subdomains → micro-subdomains.

    domain_limit: max top-level domains (None = all 29)
    subdomain_limit: max subdomains per domain (None = all)
    micro_subdomain_limit: max micro-subdomains per subdomain (None = all)
    """
    bootstrap_brain()
    started = datetime.now(timezone.utc).isoformat()
    from app.activity_log import clear_cycle_id, log_ai_activity, new_cycle_id

    cycle_id = new_cycle_id("full")
    log_ai_activity(
        "full_brain_start",
        cycle_id=cycle_id,
        source=source,
        domain_limit=domain_limit,
        subdomain_limit=subdomain_limit,
        micro_subdomain_limit=micro_subdomain_limit,
        epochs=epochs,
    )
    domains = list(KNOWLEDGE_TAXONOMY.keys())
    if domain_limit:
        domains = domains[:domain_limit]

    domain_results: list[dict] = []
    total_cycles = 0

    for domain_slug in domains:
        subs = subdomains_for(domain_slug)
        if subdomain_limit:
            subs = subs[:subdomain_limit]
        subdomain_cycles = []
        for sub in subs:
            cycle = run_subdomain_cycle(
                domain_slug,
                sub,
                epochs=epochs,
                micro_subdomain_limit=micro_subdomain_limit,
            )
            subdomain_cycles.append(cycle)
            total_cycles += cycle.get("micro_subdomains_processed", 0)
        domain_results.append({"domain": domain_slug, "subdomain_cycles": subdomain_cycles})

    with get_session() as session:
        doc_count = session.scalar(select(func.count()).select_from(Document)) or 0
        agent_count = session.scalar(select(func.count()).select_from(MicroAgent)) or 0

    result = {
        "architecture": "domain_subdomain_micro_subdomain_micro_agents",
        "regions": [r[0] for r in REGION_ORDER],
        "total_domains": len(KNOWLEDGE_TAXONOMY),
        "total_subdomains": total_subdomains(),
        "total_micro_subdomains": total_micro_subdomains(),
        "domains_processed": len(domains),
        "micro_subdomain_cycles_executed": total_cycles,
        "documents_in_db": doc_count,
        "micro_agents_registered": agent_count,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "domain_results": domain_results,
    }
    log_ai_activity(
        "full_brain_complete",
        cycle_id=cycle_id,
        source=source,
        domains_processed=len(domains),
        micro_subdomain_cycles_executed=total_cycles,
        documents_in_db=doc_count,
    )
    clear_cycle_id()
    return result


def brain_status() -> dict[str, Any]:
    """Snapshot of brain state from PostgreSQL."""
    init_db()
    with get_session() as session:
        domain_count = session.scalar(select(func.count()).select_from(KnowledgeDomain)) or 0
        subdomain_count = session.scalar(select(func.count()).select_from(KnowledgeSubdomain)) or 0
        micro_count = session.scalar(select(func.count()).select_from(KnowledgeMicroSubdomain)) or 0
        agent_count = session.scalar(select(func.count()).select_from(MicroAgent)) or 0
        doc_count = session.scalar(select(func.count()).select_from(Document)) or 0
        verified = (
            session.scalar(
                select(func.count()).select_from(Document).where(Document.verified.is_(True))
            )
            or 0
        )

        if domain_count == 0:
            seed_stats = seed_knowledge_taxonomy(session)
            domain_count = seed_stats["domains"]
            subdomain_count = seed_stats["subdomains"]
            micro_count = seed_stats["micro_subdomains"]
            agent_count = seed_stats["agents"]

        grade_graduated = (
            session.scalar(
                select(func.count())
                .select_from(GradeProgress)
                .where(GradeProgress.status == "graduated")
            )
            or 0
        )
        grade_total = session.scalar(select(func.count()).select_from(GradeProgress)) or 0

    return {
        "domains": domain_count,
        "subdomains": subdomain_count,
        "micro_subdomains": micro_count,
        "micro_agents": agent_count,
        "documents": doc_count,
        "verified_documents": verified,
        "grade_progress_rows": grade_total,
        "grade_levels_graduated": grade_graduated,
        "grade_curriculum": grade_slugs(),
        "regions": [r[0] for r in REGION_ORDER],
        "taxonomy_domains": list(KNOWLEDGE_TAXONOMY.keys()),
        "hierarchy": "domain → subdomain → micro_subdomain → grade → micro_agent (6 regions)",
    }
