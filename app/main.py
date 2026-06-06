"""FastAPI application for Railway deployment."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated, Any

from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from app.auto_learn import get_auto_learn_scheduler, start_auto_learn, stop_auto_learn
from app.chat_service import chat, estimate_learning_timeline, learning_snapshot
from app.brain_routes import get_grade_progress, get_taxonomy
from brain.grades import curriculum_public, get_grade
from app.middleware import SecurityGatewayMiddleware, SecurityHeadersMiddleware
from app.organism import get_organism
from app.pipeline_routes import run_pipeline_all, run_pipeline_step
from app.security import (
    api_key_required,
    clamp_domain_limit,
    clamp_epochs,
    clamp_micro_subdomain_limit,
    clamp_subdomain_limit,
    exclusive_training_lock,
    require_mutating_access,
    safe_error_message,
    validate_slug,
)
from app.security_routes import router as security_router
from app.service import concepts, run_identify_demo, run_match_demo, run_synthetic_demo
from brain.cortex import (
    bootstrap_brain,
    brain_status,
    run_domain_cycle,
    run_full_brain,
    run_grade_cycle,
    run_graduation_ladder,
    run_micro_subdomain_cycle,
    run_subdomain_cycle,
)
from brain.domains.taxonomy import KNOWLEDGE_TAXONOMY

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    from app.activity_log import configure_logging
    from app.railway_env import bootstrap_railway_environment
    from db.session import init_db
    from app.startup import start_deferred_startup

    configure_logging()
    logging.basicConfig(level=logging.INFO, force=True)
    bootstrap_railway_environment()
    from app.nomad.supply_spleen import verify_supply_chain

    verify_supply_chain()
    init_db()
    start_deferred_startup()
    yield
    try:
        from app.nomad.organism_pulse import stop_organism_pulse

        stop_organism_pulse()
        stop_auto_learn()
    except Exception:
        logger.exception("Auto-learn shutdown failed")


app = FastAPI(
    title="SOLIA",
    description="Sovereign Organism with Living Intelligence Architecture — supervised learning brain (Aureon)",
    version="1.1.0",
    lifespan=lifespan,
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(SecurityGatewayMiddleware)
app.include_router(security_router)

Mutating = Annotated[None, Depends(require_mutating_access)]


@app.get("/health")
def health() -> dict[str, str | bool]:
    from app.startup import get_startup_state

    state = get_startup_state()
    build = os.environ.get("RAILWAY_GIT_COMMIT_SHA", "local")[:12]
    return {
        "status": "ok",
        "build": build,
        "ready": state.ready,
        "bootstrap_done": state.bootstrap_done,
        "auto_learn": state.auto_learn_started,
    }


@app.get("/health/ready")
def health_ready() -> dict:
    from app.startup import get_startup_state

    state = get_startup_state()
    if not get_startup_state().ready:
        raise HTTPException(status_code=503, detail="Startup in progress")
    from app.railway_env import get_railway_bootstrap_report

    return {
        "status": "ready",
        "details": state.details,
        "railway_bootstrap": get_railway_bootstrap_report(),
    }


@app.get("/organism/vitals")
def organism_vitals() -> dict:
    """Security organism health — nomad_cyber_algorithm pattern."""
    organism = get_organism()
    organism.pulse()
    return organism.get_vitals_report()


@app.get("/chat")
def chat_ui() -> FileResponse:
    """Modern dark chat UI — connect to Railway or run same-origin locally."""
    page = Path(__file__).resolve().parent.parent / "static" / "chat" / "index.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="Chat UI not found")
    return FileResponse(page)


@app.get("/api/chat/access")
def api_chat_access() -> dict[str, Any]:
    """How to get your free Aureon API key — metadata only (never exposes the secret)."""
    from app.railway_env import get_railway_bootstrap_report
    from app.security import api_key_required

    report = get_railway_bootstrap_report()
    configured = api_key_required()
    source = report.get("api_key", "none") if report.get("railway") else ("env" if configured else "none")

    return {
        "free_custom_api_key": True,
        "included_with": "SOLIA on Railway — no extra charge for your personal key",
        "configured": configured,
        "source": source,
        "secrets_file": report.get("secrets_file"),
        "how_to_get": [
            "Deploy SOLIA on Railway — a unique AUREON_API_KEY is auto-generated for you at no cost.",
            "Open Railway → your web service → Variables and copy AUREON_API_KEY "
            "(or read data/railway-secrets.json on the server after first boot).",
            "Paste the key in this UI or send header X-API-Key from your own app for training endpoints.",
        ],
        "chat_is_free": True,
        "training_requires_key": True,
        "learning_logs": {
            "railway_logs": "Filter deploy logs for aureon.ai — every grade cycle and brain region is logged as JSON.",
            "audit_chain": "Tamper-evident audit log at AUREON_AUDIT_LOG_DIR (nomad audit_immune pattern).",
            "endpoints": ["/api/brain/auto-learn", "/api/chat/learning", "/security/audit"],
        },
    }


@app.get("/api/chat/learning")
def api_chat_learning() -> dict:
    """Public learning snapshot for chat sidebar and mobile apps."""
    return learning_snapshot()


@app.get("/api/chat/timeline")
def api_chat_timeline() -> dict:
    """Grade mastery time estimates."""
    scheduler = get_auto_learn_scheduler().status()
    cfg = scheduler.get("config", {})
    return estimate_learning_timeline(
        interval_sec=cfg.get("interval_sec", 3600),
        max_grades_per_cycle=cfg.get("max_grades_per_cycle", 1),
    )


@app.get("/api/chat/self-inquiry")
def api_chat_self_inquiry(limit: int = Query(default=20, ge=1, le=100)) -> dict:
    """Recent inner monologue — learning reflections + meta-consciousness."""
    from brain.meta_consciousness import combined_recent_inquiries, is_meta_consciousness_enabled
    from brain.self_inquiry import is_self_inquiry_enabled, recent_inquiries

    return {
        "enabled": is_self_inquiry_enabled() or is_meta_consciousness_enabled(),
        "self_inquiry_enabled": is_self_inquiry_enabled(),
        "meta_consciousness_enabled": is_meta_consciousness_enabled(),
        "inquiries": combined_recent_inquiries(limit),
        "learning_inquiries": recent_inquiries(limit),
    }


@app.post("/api/brain/think")
def api_brain_think(_: Mutating, count: int = Query(default=3, ge=1, le=5)) -> dict:
    """Trigger meta-cognitive self-inquiry on demand."""
    from brain.meta_consciousness import is_meta_consciousness_enabled, run_meta_inquiry

    if not is_meta_consciousness_enabled():
        return {"enabled": False, "exchanges": [], "error": "meta_consciousness disabled"}
    exchanges = run_meta_inquiry(count=count, source="api")
    return {"enabled": True, "exchanges": exchanges}


@app.get("/api/learning/github-sync")
def api_github_sync_status() -> dict:
    """GitHub learning corpus sync status (export → learning-data branch)."""
    from app.learning_github_sync import get_github_sync_state

    return get_github_sync_state().to_dict()


@app.post("/api/learning/github-sync")
def api_github_sync_run(_: Mutating) -> dict:
    """Export learning data and push to configured GitHub repos."""
    from app.learning_github_sync import run_github_sync

    return run_github_sync(reason="api")


@app.get("/api/labels/review")
def api_labels_review_pending(
    _: Mutating,
    limit: int = Query(default=50, ge=1, le=200),
    domain_slug: str | None = Query(default=None),
) -> dict:
    """Labels flagged by the teacher model for human review."""
    from app.label_review import list_pending_review

    if domain_slug:
        validate_slug(domain_slug)
    return list_pending_review(limit=limit, domain_slug=domain_slug)


@app.post("/api/labels/review/{label_id}")
def api_labels_review_resolve(label_id: int, body: dict[str, Any], _: Mutating) -> dict:
    """Approve (optionally relabel) or reject a pending label."""
    from app.label_review import resolve_label

    return resolve_label(
        label_id,
        label=str(body.get("label", "")).strip() or None,
        approve=bool(body.get("approve", True)),
    )


@app.post("/api/chat")
def api_chat(request: Request, body: dict[str, Any]) -> dict[str, Any]:
    """Public chat — supervised classification + live learning context (no training)."""
    from app.chat_rate_limit import get_chat_rate_limiter
    from app.middleware import _client_ip

    if not get_chat_rate_limiter().try_acquire(_client_ip(request)):
        raise HTTPException(status_code=429, detail="Chat rate limit exceeded — try again shortly")
    try:
        return chat(
            str(body.get("message", "")),
            session_id=body.get("session_id"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_error_message(exc)) from exc


@app.post("/api/chat/file")
async def api_chat_file(
    request: Request,
    file: UploadFile = File(...),
    message: str = Form(default=""),
    session_id: str | None = Form(default=None),
    persist: bool = Form(default=True),
) -> dict[str, Any]:
    """Upload PDF/image/audio/text — route through Tier 3–4 processors, then chat on extracted context."""
    from app.chat_rate_limit import get_chat_rate_limiter
    from app.middleware import _client_ip
    from brain.file_router import ingest_upload

    if not get_chat_rate_limiter().try_acquire(_client_ip(request)):
        raise HTTPException(status_code=429, detail="Chat rate limit exceeded — try again shortly")

    data = await file.read()
    filename = (file.filename or "upload.bin").replace("\\", "/").split("/")[-1]
    try:
        ingested = ingest_upload(filename, data, message=message, persist=persist)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        chat_result = chat(ingested.text[:8000], session_id=session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_error_message(exc)) from exc

    return {
        **chat_result,
        "file": ingested.to_dict(),
    }


@app.get("/api/concepts")
def get_concepts() -> dict[str, Any]:
    return concepts()


@app.post("/api/demo/synthetic")
def demo_synthetic(
    _auth: Mutating,
    epochs: int = Query(default=200, ge=1, le=500),
    seed: int = Query(default=42, ge=0, le=2_147_483_647),
) -> dict[str, Any]:
    try:
        with exclusive_training_lock():
            return run_synthetic_demo(epochs=epochs, seed=seed)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_error_message(exc)) from exc


@app.post("/api/demo/match")
def demo_match(
    _auth: Mutating,
    epochs: int = Query(default=200, ge=1, le=500),
    people: int = Query(default=40, ge=2, le=40),
    seed: int = Query(default=42, ge=0, le=2_147_483_647),
) -> dict[str, Any]:
    try:
        with exclusive_training_lock():
            return run_match_demo(epochs=epochs, people=people, seed=seed)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_error_message(exc)) from exc


@app.post("/api/demo/identify")
def demo_identify(
    _auth: Mutating,
    epochs: int = Query(default=200, ge=1, le=500),
    people: int = Query(default=10, ge=2, le=10),
    seed: int = Query(default=42, ge=0, le=2_147_483_647),
) -> dict[str, Any]:
    try:
        with exclusive_training_lock():
            return run_identify_demo(epochs=epochs, people=people, seed=seed)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_error_message(exc)) from exc


@app.post("/api/brain/bootstrap")
def brain_bootstrap(_auth: Mutating) -> dict:
    return bootstrap_brain()


@app.get("/api/brain/status")
def get_brain_status() -> dict:
    return brain_status()


@app.get("/api/brain/taxonomy")
def get_brain_taxonomy() -> dict:
    return get_taxonomy()


@app.get("/api/brain/roadmap")
def get_brain_roadmap(
    months: int = Query(default=12, ge=1, le=36),
) -> dict:
    """Capability matrix + simulated timeline toward surpassing frontier LLMs."""
    from brain.capability_roadmap import roadmap_snapshot, simulate_future_timeline

    return {
        "roadmap": roadmap_snapshot(),
        "simulation": simulate_future_timeline(months_ahead=months),
    }


@app.get("/api/brain/benchmark")
def get_brain_benchmark() -> dict:
    """Run fixed Q&A benchmark vs illustrative frontier baseline."""
    from brain.frontier_benchmark import run_frontier_benchmark

    return run_frontier_benchmark(use_chat=True)


@app.get("/api/brain/benchmark/humaneval")
def get_humaneval_benchmark(_: Mutating, limit: int = Query(default=20, ge=1, le=164)) -> dict:
    """HumanEval pass@1 — retrieval + verification pipeline."""
    from brain.code_master import benchmark_humaneval

    return benchmark_humaneval(limit=limit, use_retrieval=True)


@app.post("/api/brain/rag/rebuild")
def rebuild_rag_index(_: Mutating) -> dict:
    """Force-rebuild the vector RAG index from PostgreSQL corpus."""
    from brain.vector_rag import get_rag_index

    count = get_rag_index(force_rebuild=True).document_count
    return {"ok": True, "documents_indexed": count}


@app.get("/api/brain/rag/status")
def rag_index_status() -> dict:
    from brain.vector_rag import get_rag_index

    index = get_rag_index()
    return {"documents_indexed": index.document_count, "ready": index.document_count > 0}


@app.post("/api/brain/agent")
def api_brain_agent(body: dict[str, Any], _: Mutating) -> dict:
    """Multi-step agent tool loop — search, calculate, classify, verify."""
    from brain.agent_loop import run_agent_loop

    question = str(body.get("message", "")).strip()
    if not question:
        raise HTTPException(status_code=400, detail="message required")
    max_steps = int(body.get("max_steps", 5))
    return run_agent_loop(question, max_steps=max(1, min(max_steps, 10)))


@app.post("/api/brain/code/generate")
def api_brain_code_generate(body: dict[str, Any], _: Mutating) -> dict:
    """Doctorate-level code generation — retrieval + verification pipeline."""
    from brain.code_master import generate_master_code

    question = str(body.get("message", body.get("question", ""))).strip()
    if not question:
        raise HTTPException(status_code=400, detail="message or question required")
    result = generate_master_code(question)
    pred = result.get("prediction")
    if isinstance(pred, dict):
        result = {**result, "prediction": {k: v for k, v in pred.items() if k != "error"}}
    return result


@app.post("/api/brain/code/ingest")
def api_brain_code_ingest(_: Mutating, limit: int = Query(default=2000, ge=1, le=5000)) -> dict:
    """Ingest HumanEval + MBPP into document corpus."""
    from brain.code_corpus_ingest import ingest_code_corpus

    added = ingest_code_corpus(limit=limit)
    return {"ok": True, "documents_added": added}


@app.post("/api/brain/self/plan")
def api_self_evolve_plan(body: dict[str, Any], _: Mutating) -> dict:
    from app.self_evolve import plan_evolution, repo_status

    task = str(body.get("task", "")).strip()
    if not task:
        raise HTTPException(status_code=400, detail="task required")
    return {"plan": plan_evolution(task), "repo": repo_status()}


@app.post("/api/brain/self/analyze")
def api_self_evolve_analyze(body: dict[str, Any], _: Mutating) -> dict:
    """AST-based code understanding for one file before any edit."""
    from app.self_evolve import analyze_file_for_task

    path = str(body.get("path", "")).strip()
    task = str(body.get("task", body.get("description", "review"))).strip()
    if not path:
        raise HTTPException(status_code=400, detail="path required")
    try:
        return {"analysis": analyze_file_for_task(path, task)}
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/brain/self/status")
def api_self_evolve_status(_: Mutating) -> dict:
    from app.self_evolve import list_source_files, repo_status

    return {"repo": repo_status(), "files": list_source_files(limit=100)}


@app.post("/api/brain/self/read")
def api_self_evolve_read(body: dict[str, Any], _: Mutating) -> dict:
    from app.self_evolve import read_source

    path = str(body.get("path", "")).strip()
    if not path:
        raise HTTPException(status_code=400, detail="path required")
    try:
        return read_source(path)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/brain/self/write")
def api_self_evolve_write(body: dict[str, Any], _: Mutating) -> dict:
    from app.self_evolve import write_source

    path = str(body.get("path", "")).strip()
    content = body.get("content")
    if not path or content is None:
        raise HTTPException(status_code=400, detail="path and content required")
    try:
        return write_source(path, str(content))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/brain/self/branch")
def api_self_evolve_branch(body: dict[str, Any], _: Mutating) -> dict:
    from app.self_evolve import create_evolution_branch

    desc = str(body.get("description", body.get("task", "upgrade"))).strip()
    try:
        return create_evolution_branch(desc)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/brain/self/commit")
def api_self_evolve_commit(body: dict[str, Any], _: Mutating) -> dict:
    from app.self_evolve import commit_evolution

    message = str(body.get("message", "self-evolve commit")).strip()
    paths = body.get("paths")
    if paths is not None and not isinstance(paths, list):
        raise HTTPException(status_code=400, detail="paths must be a list")
    try:
        return commit_evolution(message, paths=paths)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/brain/self/push")
def api_self_evolve_push(body: dict[str, Any], _: Mutating) -> dict:
    from app.self_evolve import push_fork

    approved = bool(body.get("approve_push", False))
    branch = body.get("branch")
    try:
        return push_fork(branch=str(branch).strip() if branch else None, approved=approved)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/brain/self/propose")
def api_self_evolve_propose(body: dict[str, Any], _: Mutating) -> dict:
    """Algorithmic patch proposals — AST + predict + code_master, no git writes."""
    from app.self_evolve import plan_evolution, repo_status
    from brain.evolve_engine import propose_evolution_writes

    task = str(body.get("task", "")).strip()
    if not task:
        raise HTTPException(status_code=400, detail="task required")
    plan = plan_evolution(task)
    max_files = int(body.get("max_files", 3))
    evolution = propose_evolution_writes(task, plan, max_files=max_files)
    safe_proposals = [
        {k: v for k, v in p.items() if k != "content"}
        for p in evolution.get("proposals", [])
    ]
    return {
        "task": task,
        "strategy": evolution.get("strategy"),
        "brain": evolution.get("brain"),
        "proposals": safe_proposals,
        "write_count": len(evolution.get("writes", [])),
        "plan": plan,
        "repo": repo_status(),
    }


@app.post("/api/brain/self/evolve")
def api_self_evolve_run(body: dict[str, Any], _: Mutating) -> dict:
    """Full cycle: branch → optional writes → commit → optional fork push (never main)."""
    from app.self_evolve import run_evolution_cycle

    task = str(body.get("task", "")).strip()
    if not task:
        raise HTTPException(status_code=400, detail="task required")
    writes = body.get("writes")
    if writes is not None and not isinstance(writes, list):
        raise HTTPException(status_code=400, detail="writes must be a list of {path, content}")
    try:
        return run_evolution_cycle(
            task,
            writes=writes,
            approve_push=bool(body.get("approve_push", False)),
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/brain/self/auto")
def api_self_evolve_auto(body: dict[str, Any], _: Mutating) -> dict:
    """Autonomous fork cycle — branch, patch, commit, push fork without human PR approval (main blocked)."""
    from app.self_evolve_agent import run_autonomous_evolution

    task = str(body.get("task", "")).strip()
    if not task:
        raise HTTPException(status_code=400, detail="task required")
    try:
        return run_autonomous_evolution(
            task,
            auto_push_fork=bool(body.get("auto_push_fork", True)),
            max_files=int(body.get("max_files", 3)),
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/brain/self/history")
def api_self_evolve_history(_: Mutating, limit: int = Query(default=50, ge=1, le=200)) -> dict:
    from app.self_evolve_agent import get_history

    return {"history": get_history(limit=limit)}


@app.get("/api/brain/inference/profile")
def api_inference_profile(seq_len: int = Query(default=1024, ge=1, le=1_000_000)) -> dict:
    from src.efficient_inference import attention_window, inference_profile

    return {
        "profile": inference_profile(seq_len),
        "attention_window": attention_window(),
        "max_seq_config": int(__import__("os").environ.get("AUREON_PREDICT_MAX_SEQ", "1000000")),
    }


@app.get("/api/brain/multimodal/status")
def api_multimodal_status() -> dict:
    from brain.multimodal_collector import multimodal_status
    from brain.multimodal_processors import tier_status
    from brain.pgvector_store import status as pgvector_status

    return {
        **multimodal_status(),
        "tiers": tier_status(),
        "pgvector": pgvector_status(),
    }


@app.post("/api/brain/multimodal/ingest")
def api_multimodal_ingest(_: Mutating, limit: int = Query(default=10, ge=1, le=50)) -> dict:
    """Collect multimodal sidecars from data/raw/multimodal and persist to documents + RAG."""
    from brain.multimodal_collector import MultimodalCollector
    from brain.multimodal_persist import persist_multimodal_docs

    docs = MultimodalCollector().collect(limit=limit)
    stats = persist_multimodal_docs(docs)
    return {
        "collected": len(docs),
        "persisted": stats,
        "sources": [d.source for d in docs],
    }


@app.post("/api/chat/feedback")
def api_chat_feedback(body: dict[str, Any], _: Mutating) -> dict:
    """Submit preferred/rejected pair for RLHF retraining."""
    from brain.chat_reward import record_preference

    context = str(body.get("context", "")).strip()
    preferred = str(body.get("preferred", "")).strip()
    rejected = str(body.get("rejected", "")).strip()
    if not context or not preferred or not rejected:
        raise HTTPException(status_code=400, detail="context, preferred, rejected required")
    return record_preference(context=context, preferred=preferred, rejected=rejected)


@app.post("/api/brain/run")
def brain_run(
    _auth: Mutating,
    epochs: int = Query(default=150, ge=50, le=500),
    domain_limit: int | None = Query(default=3, ge=1, le=30),
    subdomain_limit: int | None = Query(default=1, ge=1, le=20),
    micro_subdomain_limit: int | None = Query(default=1, ge=1, le=10),
) -> dict:
    try:
        with exclusive_training_lock():
            return run_full_brain(
                epochs=clamp_epochs(epochs),
                domain_limit=clamp_domain_limit(domain_limit),
                subdomain_limit=clamp_subdomain_limit(subdomain_limit),
                micro_subdomain_limit=clamp_micro_subdomain_limit(micro_subdomain_limit),
                source="api",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_error_message(exc)) from exc


@app.post("/api/brain/domain/{domain_slug}")
def brain_run_domain(
    domain_slug: str,
    _auth: Mutating,
    epochs: int = Query(default=150, ge=50, le=500),
    subdomain_limit: int | None = Query(default=5, ge=1, le=20),
    micro_subdomain_limit: int | None = Query(default=1, ge=1, le=10),
) -> dict:
    domain_slug = validate_slug(domain_slug, label="domain")
    if domain_slug not in KNOWLEDGE_TAXONOMY:
        raise HTTPException(status_code=404, detail="Unknown domain")
    try:
        with exclusive_training_lock():
            return run_domain_cycle(
                domain_slug,
                epochs=epochs,
                subdomain_limit=clamp_subdomain_limit(subdomain_limit) or 5,
                micro_subdomain_limit=clamp_micro_subdomain_limit(micro_subdomain_limit),
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_error_message(exc)) from exc


@app.post("/api/brain/domain/{domain_slug}/{subdomain_slug}")
def brain_run_subdomain(
    domain_slug: str,
    subdomain_slug: str,
    _auth: Mutating,
    epochs: int = Query(default=150, ge=50, le=500),
    micro_subdomain_limit: int | None = Query(default=3, ge=1, le=10),
) -> dict:
    domain_slug = validate_slug(domain_slug, label="domain")
    subdomain_slug = validate_slug(subdomain_slug, label="subdomain")
    if domain_slug not in KNOWLEDGE_TAXONOMY:
        raise HTTPException(status_code=404, detail="Unknown domain")
    if subdomain_slug not in KNOWLEDGE_TAXONOMY[domain_slug]:
        raise HTTPException(status_code=404, detail="Unknown subdomain")
    try:
        with exclusive_training_lock():
            return run_subdomain_cycle(
                domain_slug,
                subdomain_slug,
                epochs=epochs,
                micro_subdomain_limit=clamp_micro_subdomain_limit(micro_subdomain_limit),
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_error_message(exc)) from exc


@app.get("/api/brain/auto-learn")
def get_auto_learn_status() -> dict:
    """Automated learning scheduler status (Railway background cycles)."""
    return get_auto_learn_scheduler().status()


@app.get("/api/brain/grades")
def get_grade_curriculum() -> dict:
    return {"curriculum": curriculum_public(), "total_levels": len(curriculum_public())}


@app.get("/api/brain/grades/{domain_slug}/{subdomain_slug}/{micro_subdomain_slug}")
def get_micro_grade_progress(
    domain_slug: str,
    subdomain_slug: str,
    micro_subdomain_slug: str,
) -> dict:
    domain_slug = validate_slug(domain_slug, label="domain")
    subdomain_slug = validate_slug(subdomain_slug, label="subdomain")
    micro_subdomain_slug = validate_slug(micro_subdomain_slug, label="micro_subdomain")
    return get_grade_progress(domain_slug, subdomain_slug, micro_subdomain_slug)


@app.post("/api/brain/domain/{domain_slug}/{subdomain_slug}/{micro_subdomain_slug}/grade/{grade_slug}")
def brain_run_grade(
    domain_slug: str,
    subdomain_slug: str,
    micro_subdomain_slug: str,
    grade_slug: str,
    _auth: Mutating,
    epochs: int = Query(default=150, ge=50, le=500),
) -> dict:
    domain_slug = validate_slug(domain_slug, label="domain")
    subdomain_slug = validate_slug(subdomain_slug, label="subdomain")
    micro_subdomain_slug = validate_slug(micro_subdomain_slug, label="micro_subdomain")
    grade_slug = validate_slug(grade_slug, label="grade")
    if not get_grade(grade_slug):
        raise HTTPException(status_code=404, detail="Unknown grade level")
    if domain_slug not in KNOWLEDGE_TAXONOMY:
        raise HTTPException(status_code=404, detail="Unknown domain")
    if subdomain_slug not in KNOWLEDGE_TAXONOMY[domain_slug]:
        raise HTTPException(status_code=404, detail="Unknown subdomain")
    if micro_subdomain_slug not in KNOWLEDGE_TAXONOMY[domain_slug][subdomain_slug]:
        raise HTTPException(status_code=404, detail="Unknown micro_subdomain")
    try:
        with exclusive_training_lock():
            return run_grade_cycle(
                domain_slug,
                subdomain_slug,
                micro_subdomain_slug,
                grade_slug=grade_slug,
                epochs=epochs,
                source="api",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_error_message(exc)) from exc


@app.post("/api/brain/domain/{domain_slug}/{subdomain_slug}/{micro_subdomain_slug}/graduate")
def brain_run_graduation_ladder(
    domain_slug: str,
    subdomain_slug: str,
    micro_subdomain_slug: str,
    _auth: Mutating,
    epochs: int = Query(default=150, ge=50, le=500),
    max_grades: int | None = Query(default=3, ge=1, le=7),
) -> dict:
    domain_slug = validate_slug(domain_slug, label="domain")
    subdomain_slug = validate_slug(subdomain_slug, label="subdomain")
    micro_subdomain_slug = validate_slug(micro_subdomain_slug, label="micro_subdomain")
    if domain_slug not in KNOWLEDGE_TAXONOMY:
        raise HTTPException(status_code=404, detail="Unknown domain")
    if subdomain_slug not in KNOWLEDGE_TAXONOMY[domain_slug]:
        raise HTTPException(status_code=404, detail="Unknown subdomain")
    if micro_subdomain_slug not in KNOWLEDGE_TAXONOMY[domain_slug][subdomain_slug]:
        raise HTTPException(status_code=404, detail="Unknown micro_subdomain")
    try:
        with exclusive_training_lock():
            return run_graduation_ladder(
                domain_slug,
                subdomain_slug,
                micro_subdomain_slug,
                epochs=epochs,
                max_grades=max_grades,
                source="api",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_error_message(exc)) from exc


@app.post("/api/brain/domain/{domain_slug}/{subdomain_slug}/{micro_subdomain_slug}")
def brain_run_micro_subdomain(
    domain_slug: str,
    subdomain_slug: str,
    micro_subdomain_slug: str,
    _auth: Mutating,
    epochs: int = Query(default=150, ge=50, le=500),
) -> dict:
    domain_slug = validate_slug(domain_slug, label="domain")
    subdomain_slug = validate_slug(subdomain_slug, label="subdomain")
    micro_subdomain_slug = validate_slug(micro_subdomain_slug, label="micro_subdomain")
    if domain_slug not in KNOWLEDGE_TAXONOMY:
        raise HTTPException(status_code=404, detail="Unknown domain")
    if subdomain_slug not in KNOWLEDGE_TAXONOMY[domain_slug]:
        raise HTTPException(status_code=404, detail="Unknown subdomain")
    if micro_subdomain_slug not in KNOWLEDGE_TAXONOMY[domain_slug][subdomain_slug]:
        raise HTTPException(status_code=404, detail="Unknown micro_subdomain")
    try:
        with exclusive_training_lock():
            return run_micro_subdomain_cycle(
                domain_slug,
                subdomain_slug,
                micro_subdomain_slug,
                epochs=epochs,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_error_message(exc)) from exc


@app.post("/api/pipeline/run")
def pipeline_run_all_endpoint(
    _auth: Mutating,
    epochs: int = Query(default=200, ge=50, le=500),
) -> dict:
    try:
        with exclusive_training_lock():
            return run_pipeline_all(epochs=epochs)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_error_message(exc)) from exc


@app.post("/api/pipeline/step/{step}")
def pipeline_run_step_endpoint(
    step: int,
    _auth: Mutating,
    epochs: int = Query(default=200, ge=50, le=500),
) -> dict:
    if step not in (1, 2, 3, 4, 5):
        raise HTTPException(status_code=400, detail="step must be 1-5")
    try:
        with exclusive_training_lock():
            return run_pipeline_step(step=step, epochs=epochs)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_error_message(exc)) from exc


@app.get("/api/pipeline/status")
def pipeline_status() -> dict:
    from pipeline.config import REGISTRY_DIR
    from pipeline.step3_training.registry import ModelRegistry

    registry = ModelRegistry()
    latest = REGISTRY_DIR / "latest_pipeline_run.json"
    payload: dict = {
        "production_model": registry.get_production(),
        "best_run": registry.get_best_run(),
    }
    if latest.exists():
        import json

        from app.security import load_json_file_bounded

        payload["latest_run"] = load_json_file_bounded(latest)
    return payload


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SOLIA — Sovereign Organism with Living Intelligence Architecture</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #0f1419;
      --card: #1a2332;
      --text: #e7ecf3;
      --muted: #9aa7b8;
      --accent: #5b9fd4;
      --accent-hover: #7ab3e0;
      --border: #2a3647;
      --ok: #6bcf8e;
    }
    @media (prefers-color-scheme: light) {
      :root {
        --bg: #f4f7fb;
        --card: #ffffff;
        --text: #1a2332;
        --muted: #5a6778;
        --accent: #2563eb;
        --accent-hover: #1d4ed8;
        --border: #d8e0ea;
        --ok: #15803d;
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    main { max-width: 920px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }
    h1 { font-size: 1.75rem; margin: 0 0 0.5rem; }
    .subtitle { color: var(--muted); margin-bottom: 2rem; }
    .grid { display: grid; gap: 1rem; margin-bottom: 2rem; }
    @media (min-width: 700px) { .grid { grid-template-columns: repeat(3, 1fr); } }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem;
    }
    .card h2 { font-size: 1rem; margin: 0 0 0.5rem; }
    .card p { font-size: 0.9rem; color: var(--muted); margin: 0 0 1rem; min-height: 3.5rem; }
    button {
      width: 100%;
      border: none;
      border-radius: 8px;
      padding: 0.75rem 1rem;
      background: var(--accent);
      color: white;
      font-weight: 600;
      cursor: pointer;
    }
    button:hover { background: var(--accent-hover); }
    button:disabled { opacity: 0.6; cursor: wait; }
    #status { color: var(--muted); min-height: 1.25rem; margin-bottom: 0.75rem; }
    #status.running { color: var(--accent); }
    #status.done { color: var(--ok); }
    pre {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1rem;
      overflow: auto;
      font-size: 0.82rem;
      min-height: 200px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    ul.constraints { padding-left: 1.2rem; color: var(--muted); }
    code { background: var(--border); padding: 0.1rem 0.35rem; border-radius: 4px; }
  </style>
</head>
<body>
  <main>
    <h1>SOLIA</h1>
    <p class="subtitle">
      Supervised machine learning with backpropagation — what &ldquo;AI&rdquo; actually is.
    </p>

    <div class="card" style="margin-bottom: 2rem;">
      <h2>How it works</h2>
      <p style="min-height: auto; color: var(--text);">
        Traditional code: you write the algorithm (<code>output = input + input</code>).
        Supervised ML: you provide labeled inputs and measurable outputs; the network
        learns weights via <strong>backpropagation</strong>.
      </p>
      <ul class="constraints">
        <li><strong>Clean data</strong> — labeled images/features, not opinions</li>
        <li><strong>Measurable goal</strong> — yes/no or person ID</li>
        <li><strong>Defined parameters</strong> — a bounded face database</li>
      </ul>
    </div>

    <div class="card" style="margin-bottom: 2rem;">
      <h2>Brain — micro-algorithms per domain</h2>
      <p style="min-height: auto; color: var(--text);">
        30 Zophiel knowledge domains, 135 subdomains, 862 micro-subdomains, 6 brain regions each.
      </p>
      <button id="runBrainBtn" style="max-width: 320px;">Run brain cycle</button>
    </div>

    <div class="card" style="margin-bottom: 2rem;">
      <h2>Automated training pipeline</h2>
      <p style="min-height: auto; color: var(--text);">
        Run all 5 steps in order: collect &rarr; label &rarr; train &rarr; evaluate &rarr; RLHF.
      </p>
      <button id="runPipelineBtn" style="max-width: 320px;">Run full pipeline</button>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Synthetic features</h2>
        <p>Eye, nose, chin weights — the lecture&rsquo;s core example.</p>
        <button data-demo="synthetic">Run demo</button>
      </div>
      <div class="card">
        <h2>Face matching</h2>
        <p>Binary yes/no: do two faces belong to the same person?</p>
        <button data-demo="match">Run demo</button>
      </div>
      <div class="card">
        <h2>Person ID</h2>
        <p>Multi-class identification + edge-case fragility demo.</p>
        <button data-demo="identify">Run demo</button>
      </div>
    </div>

    <div id="status"></div>
    <pre id="output">Click a demo to train a neural network and see results.</pre>
  </main>
  <script>
    const statusEl = document.getElementById('status');
    const outputEl = document.getElementById('output');
    const buttons = document.querySelectorAll('button');

    function apiHeaders() {
      const headers = { 'Content-Type': 'application/json' };
      const key = window.AUREON_API_KEY;
      if (key) {
        headers['X-API-Key'] = key;
        headers['X-Timestamp'] = String(Date.now());
        headers['X-Nonce'] = crypto.randomUUID();
        headers['X-Correlation-ID'] = crypto.randomUUID();
      }
      return headers;
    }

    async function postJson(url) {
      const res = await fetch(url, { method: 'POST', headers: apiHeaders() });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || res.statusText);
      return data;
    }

    async function runBrain() {
      buttons.forEach(b => b.disabled = true);
      statusEl.textContent = 'Running brain micro-algorithms across knowledge domains…';
      statusEl.className = 'running';
      outputEl.textContent = '';
      try {
        outputEl.textContent = JSON.stringify(
          await postJson('/api/brain/run?epochs=150&domain_limit=2&subdomain_limit=1'),
          null, 2
        );
        statusEl.textContent = 'Brain cycle complete.';
        statusEl.className = 'done';
      } catch (err) {
        outputEl.textContent = String(err);
        statusEl.textContent = 'Error.';
        statusEl.className = '';
      } finally {
        buttons.forEach(b => b.disabled = false);
      }
    }

    async function runPipeline() {
      buttons.forEach(b => b.disabled = true);
      statusEl.textContent = 'Running 5-step pipeline…';
      statusEl.className = 'running';
      outputEl.textContent = '';
      try {
        const data = await postJson('/api/pipeline/run?epochs=200');
        outputEl.textContent = JSON.stringify(data, null, 2);
        statusEl.textContent = 'Pipeline ' + (data.status || 'done') + '.';
        statusEl.className = 'done';
      } catch (err) {
        outputEl.textContent = String(err);
        statusEl.textContent = 'Error.';
        statusEl.className = '';
      } finally {
        buttons.forEach(b => b.disabled = false);
      }
    }

    async function runDemo(name) {
      buttons.forEach(b => b.disabled = true);
      statusEl.textContent = 'Training neural network via backpropagation…';
      statusEl.className = 'running';
      outputEl.textContent = '';
      try {
        outputEl.textContent = JSON.stringify(await postJson('/api/demo/' + name + '?epochs=200'), null, 2);
        statusEl.textContent = 'Done.';
        statusEl.className = 'done';
      } catch (err) {
        outputEl.textContent = String(err);
        statusEl.textContent = 'Error.';
        statusEl.className = '';
      } finally {
        buttons.forEach(b => b.disabled = false);
      }
    }

    document.getElementById('runBrainBtn').addEventListener('click', runBrain);
    document.getElementById('runPipelineBtn').addEventListener('click', runPipeline);
    document.querySelectorAll('[data-demo]').forEach(btn => {
      btn.addEventListener('click', () => runDemo(btn.dataset.demo));
    });
  </script>
</body>
</html>"""


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
