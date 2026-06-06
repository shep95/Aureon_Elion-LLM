"""Chat with Aureon — supervised inference + live learning context."""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import numpy as np

from app.auto_learn import get_auto_learn_scheduler
from app.predict_rate_limit import get_predict_rate_limiter
from app.session_memory import append_turn, history_as_context
from brain.cortex import brain_status
from brain.domains.taxonomy import total_micro_subdomains
from brain.grades import GRADE_CURRICULUM, curriculum_public, epochs_for_grade, get_grade
from brain.graduation import current_grade, progress_report
from brain.cipher_logic import ciper_research
from brain.agent_loop import is_agent_task, run_agent_loop
from brain.brain_classifiers import classify_moe
from brain.capability_roadmap import roadmap_snapshot, simulate_future_timeline, try_roadmap_answer
from brain.chat_reward import apply_chat_reward
from brain.deterministic_qa import try_arithmetic_answer
from brain.predict_engine import is_prediction_question, predict_with_steps
from brain.psychology_brain import finalize_chat_payload
from brain.meta_consciousness import (
    combined_recent_inquiries,
    is_meta_consciousness_enabled,
    run_meta_inquiry,
    try_meta_answer,
)
from brain.self_inquiry import is_self_inquiry_enabled, recent_inquiries
from brain.simple_qa import is_simple_question, to_simple_answer
from db.models import KnowledgeDomain, KnowledgeMicroSubdomain, KnowledgeSubdomain
from db.session import get_session
from pipeline.step4_evaluation.benchmarks import _load_production_model, _predict_label
from sqlalchemy import func, select
from sqlalchemy.orm import Session

_snapshot_lock = threading.Lock()
_snapshot_cache: dict[str, Any] | None = None
_snapshot_cached_at: float = 0.0


def _snapshot_ttl_sec() -> float:
    raw = os.environ.get("AUREON_LEARNING_SNAPSHOT_TTL_SEC", "30").strip()
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 30.0


def learning_snapshot(*, force: bool = False) -> dict[str, Any]:
    global _snapshot_cache, _snapshot_cached_at
    now = time.monotonic()
    if not force:
        with _snapshot_lock:
            if _snapshot_cache is not None and (now - _snapshot_cached_at) < _snapshot_ttl_sec():
                return _snapshot_cache

    scheduler = get_auto_learn_scheduler()
    status = scheduler.status()
    brain = brain_status()
    timeline = estimate_learning_timeline(
        interval_sec=status.get("config", {}).get("interval_sec", 3600),
        max_grades_per_cycle=status.get("config", {}).get("max_grades_per_cycle", 1),
        micro_subdomain_count=brain.get("micro_subdomains", total_micro_subdomains()),
    )
    payload = {
        "auto_learn": status,
        "brain": {
            "domains": brain.get("domains"),
            "micro_subdomains": brain.get("micro_subdomains"),
            "micro_agents": brain.get("micro_agents"),
            "documents": brain.get("documents"),
            "grade_levels_graduated": brain.get("grade_levels_graduated"),
            "grade_progress_rows": brain.get("grade_progress_rows"),
        },
        "timeline": timeline,
        "self_inquiry": {
            "enabled": is_self_inquiry_enabled(),
            "recent": recent_inquiries(8),
        },
        "meta_consciousness": {
            "enabled": is_meta_consciousness_enabled(),
            "recent": combined_recent_inquiries(8),
        },
    }
    with _snapshot_lock:
        _snapshot_cache = payload
        _snapshot_cached_at = now
    return payload


def _learning_snapshot() -> dict[str, Any]:
    """Cached snapshot for chat payloads."""
    return learning_snapshot()


def _finalize(payload: dict[str, Any], user_message: str) -> dict[str, Any]:
    return finalize_chat_payload(apply_chat_reward(payload, user_message), user_message)


def _agent_payload(text: str, *, session_id: str | None) -> dict[str, Any]:
    result = run_agent_loop(text, session_id=session_id)
    return {
        "reply": result["answer"],
        "kind": "agent",
        "session_id": session_id,
        "learning": learning_snapshot(),
        "agent": {
            "plan": result["plan"],
            "steps": result["steps"],
            "confidence": result.get("confidence"),
        },
        "citations": result.get("citations", []),
    }


def estimate_learning_timeline(
    *,
    interval_sec: int = 3600,
    max_grades_per_cycle: int = 1,
    micro_subdomain_count: int | None = None,
) -> dict[str, Any]:
    """Wall-clock estimates for grade mastery (auto-learn defaults)."""
    if micro_subdomain_count is None:
        micro_subdomain_count = total_micro_subdomains()
    grades = []
    for grade in GRADE_CURRICULUM:
        grades.append(
            {
                "slug": grade.slug,
                "name": grade.name,
                "epochs_per_cycle": epochs_for_grade(150, grade),
                "min_train_accuracy": grade.min_train_accuracy,
                "cycles_to_clear_one_micro": 1,
                "wall_clock_if_one_grade_per_hour": f"~{interval_sec // 60} min per grade step",
            }
        )

    steps_one_micro = len(GRADE_CURRICULUM)
    sec_one_micro = steps_one_micro * interval_sec
    sec_full_corpus = micro_subdomain_count * steps_one_micro * interval_sec

    def fmt(seconds: int) -> str:
        if seconds < 3600:
            return f"{seconds // 60} min"
        if seconds < 86400:
            return f"{seconds // 3600} hours"
        return f"{seconds / 86400:.1f} days"

    return {
        "grades": grades,
        "grade_count": len(GRADE_CURRICULUM),
        "auto_learn_defaults": {
            "interval_sec": interval_sec,
            "max_grades_per_cycle": max_grades_per_cycle,
            "epochs_base": 150,
        },
        "estimates": {
            "one_grade_step": fmt(interval_sec),
            "one_micro_subdomain_all_grades": fmt(sec_one_micro),
            "full_corpus_sequential": fmt(sec_full_corpus),
            "note": (
                "Assumes one successful grade graduation per auto-learn cycle. "
                "Failed grades retry; trainer needs ≥2 label classes for full accuracy gates."
            ),
        },
    }


def _active_micro_progress(session: Session) -> dict[str, Any] | None:
    target = get_auto_learn_scheduler().state.current_target
    if not target:
        return None
    domain = session.scalar(
        select(KnowledgeDomain).where(KnowledgeDomain.slug == target["domain"])
    )
    if not domain:
        return None
    subdomain = session.scalar(
        select(KnowledgeSubdomain).where(
            KnowledgeSubdomain.domain_id == domain.id,
            KnowledgeSubdomain.slug == target["subdomain"],
        )
    )
    if not subdomain:
        return None
    micro = session.scalar(
        select(KnowledgeMicroSubdomain).where(
            KnowledgeMicroSubdomain.subdomain_id == subdomain.id,
            KnowledgeMicroSubdomain.slug == target["micro_subdomain"],
        )
    )
    if not micro:
        return None
    row = current_grade(session, micro.id)
    return {
        "path": f"{target['domain']}.{target['subdomain']}.{target['micro_subdomain']}",
        "current_grade": row.grade_slug if row else "graduated",
        "progress": progress_report(session, micro.id),
    }


def _classify_message(text: str) -> dict[str, Any] | None:
    moe = classify_moe(text)
    if moe:
        return moe

    from pipeline.step4_evaluation.benchmarks import _load_production_model

    loaded = _load_production_model()
    if not loaded:
        return None
    network, labels, extractor = loaded
    prediction = _predict_label(network, labels, extractor, text)
    x = extractor.transform([text])
    proba = network.predict_proba(x)[0]
    confidence = float(np.max(proba))
    return {
        "label": prediction,
        "confidence": round(confidence, 4),
        "labels_available": labels,
        "model": "production_classifier",
        "routing": "pipeline_fallback",
    }


def _deterministic_payload(text: str, *, session_id: str | None) -> dict[str, Any] | None:
    """Exact evaluators (arithmetic) — no neural guess."""
    result = try_arithmetic_answer(text)
    if not result:
        return None
    return {
        "reply": result["answer"],
        "kind": "chat",
        "session_id": session_id,
        "learning": learning_snapshot(),
        "simple_qa": True,
        "deterministic": {
            "evaluator": result["evaluator"],
            "expression": result["expression"],
        },
    }


def _resolve_followup(text: str, session_id: str | None) -> str:
    """Map 'tell me more about that' to the prior user question."""
    if not session_id:
        return text
    q = text.lower()
    triggers = ("tell me more", "more about that", "about that", "explain that", "what about that")
    if any(t in q for t in triggers):
        from app.session_memory import get_history

        hist = get_history(session_id, limit=1)
        if hist:
            return f"{hist[-1]['user']} — follow up: {text}"
    return text


def _predict_with_timeout(
    question: str,
    *,
    session_id: str | None = None,
    seconds: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run predict brain with rate limit + timeout — always returns a dict."""
    if not get_predict_rate_limiter().try_acquire(session_id):
        return {
            "answer": "Too many predict requests — wait a minute and try again.",
            "abstained": False,
            "rate_limited": True,
            "confidence": 0.0,
            "model": "stacked_attention_lm",
            "citations": [],
        }

    limit = seconds if seconds is not None else float(os.environ.get("AUREON_PREDICT_TIMEOUT_SEC", "8"))
    conv = history_as_context(session_id)
    result: list[dict[str, Any] | None] = [None]

    def _run() -> None:
        try:
            result[0] = predict_with_steps(question, conversation_context=conv, force=force)
        except Exception:
            result[0] = None

    worker = threading.Thread(target=_run, name="aureon-predict", daemon=True)
    worker.start()
    worker.join(timeout=limit)

    if worker.is_alive():
        return {
            "answer": (
                "This question needs deeper corpus grounding than I can compute in time. "
                "Try again after auto-learn finishes, or ask `/mind` for what I know now."
            ),
            "abstained": False,
            "timed_out": True,
            "confidence": 0.1,
            "model": "stacked_attention_lm",
            "citations": [],
        }

    if result[0] is None:
        from brain.predict_engine import _TRAINING_NEED_MSG

        return {
            "answer": _TRAINING_NEED_MSG,
            "abstained": True,
            "confidence": 0.0,
            "model": "stacked_attention_lm",
            "citations": [],
        }
    return result[0]


def _brain_predict_payload(text: str, *, session_id: str | None) -> dict[str, Any]:
    """Attention LM — embed, attend, predict next tokens autoregressively."""
    result = _predict_with_timeout(text, session_id=session_id)
    payload: dict[str, Any] = {
        "reply": result["answer"],
        "kind": "chat",
        "session_id": session_id,
        "learning": learning_snapshot(),
        "prediction": {
            "model": result["model"],
            "model_version": result.get("model_version"),
            "context_window": result.get("context_window"),
            "vocab_size": result.get("vocab_size"),
            "confidence": result.get("confidence"),
            "citations": result.get("citations", []),
            "pipeline": result.get("pipeline", []),
            "prompt": result.get("prompt"),
        },
        "brain_predict": True,
        "abstained": result.get("abstained", False),
        "timed_out": result.get("timed_out", False),
    }
    if result.get("citations"):
        cites = result["citations"][:3]
        payload["citations"] = cites
    return payload


def is_code_question(text: str) -> bool:
    q = text.strip().lower()
    triggers = (
        "write a function",
        "write a python",
        "write code",
        "create a function",
        "implement a function",
        "implement ",
        "def ",
        "how do i code",
        "write a script",
        "debug this",
        "fix this code",
        "what does this code do",
        "write a program",
        "code that",
        "generate python",
        "python function",
    )
    return any(t in q for t in triggers)


def _code_payload(text: str, *, session_id: str | None) -> dict[str, Any] | None:
    from brain.code_evaluator import extract_python_code
    from brain.code_master import generate_master_code

    def _predict(q: str) -> dict[str, Any] | None:
        return _predict_with_timeout(q, session_id=session_id, force=True)

    master = generate_master_code(text, predict_fn=_predict)
    if not master.get("answer"):
        return None

    code = extract_python_code(master["answer"])
    evaluation = master.get("code_eval") or {}

    answer = code
    if not evaluation.get("syntax_valid"):
        answer = (
            f"{code}\n\n"
            f"# Note: syntax check flagged an issue — {evaluation.get('error', 'invalid syntax')}"
        )
    elif evaluation.get("passed_tests") is False:
        answer = f"{code}\n\n# Note: unit tests did not pass."

    payload: dict[str, Any] = {
        "reply": answer,
        "kind": "code",
        "session_id": session_id,
        "learning": learning_snapshot(),
        "code_eval": evaluation,
        "brain_predict": master.get("method") == "neural_synthesis",
        "code_master": {
            "method": master.get("method"),
            "match_score": master.get("match_score"),
            "problem_id": master.get("problem_id"),
            "confidence": master.get("confidence"),
        },
        "citations": master.get("citations", []),
    }
    pred = master.get("prediction")
    if isinstance(pred, dict):
        payload["prediction"] = {k: v for k, v in pred.items() if k != "error"}
    return payload


def _ciper_chat_payload(text: str, *, session_id: str | None) -> dict[str, Any] | None:
    """Marie/Ciper decomposition + cross-domain research when applicable."""
    result = ciper_research(text)
    if not result:
        return None
    payload: dict[str, Any] = {
        "reply": result.reply,
        "kind": "chat",
        "session_id": session_id,
        "learning": learning_snapshot(),
        "ciper": result.to_dict(),
        "simple_qa": result.mode in ("answer", "decompose", "cross_domain"),
    }
    if result.mode == "answer" and result.grounded:
        payload["grounded"] = True
    return payload


def _simple_nl_response(text: str) -> str | None:
    """Short answers for common natural-language questions."""
    q = text.strip().lower().rstrip("?").strip()
    if not q:
        return None

    if "what are you learning" in q or "what you learning" in q:
        scheduler = get_auto_learn_scheduler()
        status = scheduler.status()
        brain = brain_status()
        cycles = status.get("cycles_completed", 0)
        docs = brain.get("documents", 0)
        lr = status.get("last_result") or {}
        if lr.get("batch"):
            cursor = lr.get("batch_cursor", lr.get("targets_processed", 0))
            total = lr.get("targets_in_corpus", 862)
            return f"Batch topic {cursor}/{total}, preschool grade, {docs} docs, {cycles} cycles done."
        return f"{docs} docs ingested, {cycles} auto-learn cycles done."

    if q in ("what is aureon",):
        from brain.identity_handler import _build_live_identity

        return _build_live_identity()

    if any(t in q for t in ("who are you", "what are you", "tell me about yourself", "introduce yourself")):
        from brain.identity_handler import get_identity_response

        return get_identity_response(text)

    roadmap = try_roadmap_answer(text)
    if roadmap:
        return roadmap

    if "how do you work" in q or "how does aureon work" in q:
        return "Inputs and labels in, backpropagation finds weights, measurable accuracy out."

    if q in ("what is ai", "what is artificial intelligence"):
        return "Supervised machine learning — labels plus weights, not magic."

    if q in ("what is math", "what is mathematics"):
        return (
            "Mathematics — study of numbers, patterns, and logical structure "
            "underlying all science and reasoning."
        )

    if "consciousness" in q and q.startswith("what is"):
        return "Consciousness — lived experience of awareness and self-knowledge."

    if "meaning of life" in q:
        return "Meaning — purpose, connection, and understanding; traditions answer differently."

    if "god" in q and (
        "who is" in q
        or "what is" in q
        or "to you" in q
        or "your thoughts" in q
        or "do you believe" in q
        or "believe in" in q
    ):
        return (
            "God is one of the deepest questions I engage with — ultimate origin, "
            "consciousness, and meaning. Traditions answer differently; "
            "I hold the question with respect and no false certainty."
        )

    if "what are you asking" in q or "questions do you ask" in q:
        items = combined_recent_inquiries(1)
        if not items:
            return "No reflections yet — wait for the next auto-learn cycle."
        item = items[0]
        return f"{item.get('question')} → {item.get('answer')}"

    meta = try_meta_answer(text)
    if meta:
        return meta

    return None


def _simple_chat_reply(text: str, *, session_id: str | None = None) -> dict[str, Any]:
    """Simple Question, Simple Answer path — classification feeds predict, never replaces reply."""
    nl = _simple_nl_response(text)
    if nl:
        return {"reply": to_simple_answer(nl), "kind": "chat", "simple_qa": True}

    classification = _classify_message(text)
    if classification:
        enriched = f"domain context {classification['label']} question {text.strip().lower()}"
        result = _predict_with_timeout(enriched, session_id=session_id, force=True)
        if result and result.get("answer") and not result.get("abstained"):
            reply = str(result["answer"]).strip()
            if len(reply) > 20 and "philosophy." not in reply.lower():
                payload: dict[str, Any] = {
                    "reply": to_simple_answer(reply),
                    "kind": "predict",
                    "simple_qa": True,
                    "brain_predict": True,
                    "classification": classification,
                }
                if result.get("citations"):
                    payload["citations"] = result["citations"][:3]
                return payload

        from brain.philosophy_handler import philosophy_fallback_if_needed

        fallback = philosophy_fallback_if_needed(text, result)
        if fallback:
            fallback["simple_qa"] = True
            fallback["classification"] = classification
            return fallback

    with get_session() as session:
        from db.models import Document

        doc_count = session.scalar(select(func.count()).select_from(Document)) or 0
        active = _active_micro_progress(session)

    if active:
        reply = f"Still training — {doc_count} docs, focus {active['path']} @ {active['current_grade']}."
    else:
        reply = f"Still training — {doc_count} docs, no promoted classifier yet."

    return {
        "reply": to_simple_answer(reply),
        "kind": "chat",
        "simple_qa": True,
        "classification": None,
        "active_micro": active,
    }


def _command_response(message: str) -> dict[str, Any] | None:
    cmd = message.strip().lower()
    if cmd in ("/help", "help"):
        return {
            "reply": (
                "Aureon is a **supervised learning brain** (not a generative LLM). "
                "I classify text with trained weights, run grade cycles on Railway, "
                "and report live learning status.\n\n"
                "Commands:\n"
                "• `/status` — brain + auto-learn snapshot\n"
                "• `/grades` — curriculum and time estimates\n"
                "• `/mind` — recent learning reflections (collected docs + cycle metrics)\n"
                "• `/think` — ask myself meta-cognitive questions (identity, gaps, consciousness)\n"
                "• `/roadmap` — capability matrix + path beyond frontier LLMs\n"
                "• `/agent <task>` — multi-step tool loop (search → calculate → verify)\n"
                "• `/evolve <task>` — self-upgrade on a fork branch (never pushes main without approval)\n"
                "• `/research <topic>` — cross-domain taxonomy + Ciper drill-down\n"
                "• `/vitals` — security organism (nomad stack)\n\n"
                "**Prediction brain:** factual questions run through token embeddings → "
                "self-attention → stacked layers → next-token probabilities → autoregressive answer.\n\n"
                "Logic: **Marie/Ciper** — broad claims get facet drill-down; "
                "specific questions get cross-domain answers when corpus supports it.\n"
                "**Two brains:** psychology layer (how I act human) + algorithm layer "
                "(six regions, supervised learning).\n"
                "Rule: **Simple Question, Simple Answer** — ask short, get short.\n\n"
                "Ask about a topic — I'll classify it when a production model exists."
            ),
            "kind": "help",
        }
    if cmd == "/status":
        snap = learning_snapshot()
        al = snap["auto_learn"]
        brain = snap["brain"]
        reply = (
            f"**Brain:** {brain['domains']} domains · {brain['micro_subdomains']} micro-topics · "
            f"{brain['documents']} documents · {brain['grade_levels_graduated']} grade graduations logged.\n"
            f"**Auto-learn:** {'ON' if al.get('enabled') else 'OFF'} · "
            f"cycles={al.get('cycles_completed', 0)} · "
            f"last={al.get('last_run_at') or 'never'} · "
            f"next={al.get('next_run_at') or 'pending'}.\n"
        )
        if al.get("last_result"):
            lr = al["last_result"]
            if lr.get("batch"):
                reply += (
                    f"**Last batch:** {lr.get('targets_processed', 0)}/{lr.get('targets_total', 0)} "
                    f"micro-topics · {lr.get('graduations_passed', 0)} graduations passed.\n"
                )
                if lr.get("last_target"):
                    lt = lr["last_target"]
                    reply += (
                        f"**Last in batch:** {lt.get('domain')}.{lt.get('subdomain')}."
                        f"{lt.get('micro_subdomain')} · graduation={lr.get('last_graduation')}.\n"
                    )
            else:
                reply += f"**Last cycle:** {lr.get('target')} · steps={lr.get('steps')} · graduation={lr.get('graduation')}.\n"
        if al.get("last_error"):
            reply += f"**Last error:** {al['last_error']}\n"
        return {"reply": reply.strip(), "kind": "status", "snapshot": snap}
    if cmd == "/grades":
        timeline = estimate_learning_timeline()
        lines = ["**Grade curriculum** (must graduate each before the next unlocks):\n"]
        for g in timeline["grades"]:
            lines.append(f"• **{g['name']}** — min accuracy {g['min_train_accuracy']:.0%}")
        est = timeline["estimates"]
        lines.append(
            f"\n**Timing** (default 1 grade / hour on Railway):\n"
            f"• One grade step: {est['one_grade_step']}\n"
            f"• One micro-topic preschool→PhD: {est['one_micro_subdomain_all_grades']}\n"
            f"• Full corpus sequential: {est['full_corpus_sequential']}\n"
            f"_{est['note']}_"
        )
        return {"reply": "\n".join(lines), "kind": "grades", "timeline": timeline}
    if cmd in ("/mind", "/reflect", "/questions"):
        items = combined_recent_inquiries(8)
        if not items:
            return {
                "reply": (
                    "No reflections yet. When auto-learn runs, I ask myself learning questions "
                    "after each grade cycle and meta-cognitive questions after each batch — "
                    "check back after the first cycle or filter logs for `self_inquiry` / "
                    "`meta_consciousness`."
                ),
                "kind": "mind",
            }
        lines = ["**Inner monologue** (learning + meta-cognition):\n"]
        for item in items:
            tag = "meta" if item.get("kind") == "meta" else "learn"
            lines.append(f"**[{tag}] Q:** {item.get('question')}")
            answer = str(item.get("answer", ""))
            cycle = item.get("cycle")
            if cycle:
                answer = f"{answer} ({cycle})"
            if len(answer) > 200:
                answer = answer[:197] + "..."
            lines.append(f"**A:** {answer}\n")
        return {"reply": "\n".join(lines).strip(), "kind": "mind", "inquiries": items}
    if cmd in ("/think", "/conscious", "/metacog"):
        exchanges = run_meta_inquiry(count=3, source="chat_command")
        if not exchanges:
            return {
                "reply": (
                    "Meta-consciousness is off. Set `AUREON_META_CONSCIOUSNESS=1` "
                    "(or enable self-inquiry)."
                ),
                "kind": "think",
            }
        lines = ["**Meta-cognitive self-inquiry** (grounded in live state):\n"]
        for ex in exchanges:
            lines.append(f"**Q:** {ex['question']}")
            lines.append(f"**A:** {ex['answer']}\n")
        return {"reply": "\n".join(lines).strip(), "kind": "think", "meta": exchanges}
    if cmd in ("/roadmap", "/capabilities", "/future"):
        snap = roadmap_snapshot()
        sim = simulate_future_timeline(months_ahead=12)
        counts = snap["status_counts"]
        lines = [
            "**Aureon capability roadmap** — supervised brain vs static frontier LLMs\n",
            f"_{snap['vision']}_\n",
            f"**Status:** {counts['live']} live · {counts['partial']} partial · "
            f"{counts['planned']} planned · {counts['research']} research · "
            f"~{snap['completion_pct']}% core complete.\n",
            "**Live now:**",
        ]
        for cap in snap["capabilities"]:
            if cap["status"] == "live":
                lines.append(f"• {cap['name']}")
        lines.append("\n**Next unlock (simulated):**")
        nxt = sim["milestones"][1] if len(sim["milestones"]) > 1 else sim["milestones"][0]
        if nxt:
            lines.append(f"• Month {nxt['month']}: {nxt['name']} — {', '.join(nxt['unlocks'][:3])}")
        lines.append(
            "\n**Why this beats GPT/Claude long-term:** grounded corpus, grade graduation, "
            "citations, abstain-when-uncertain — not hallucinated fluency."
        )
        return {"reply": "\n".join(lines), "kind": "roadmap", "roadmap": snap, "simulation": sim}
    if cmd.startswith("/evolve"):
        task = message.strip()[7:].strip(" :.")
        if not task:
            return {
                "reply": (
                    "Usage: `/evolve improve philosophy routing` — "
                    "I create a fork branch, can read/write my own source, commit locally, "
                    "and only push to your fork remote when you approve via API."
                ),
                "kind": "self_evolve",
            }
        from app.self_evolve import plan_evolution, repo_status

        plan = plan_evolution(task)
        status = repo_status()
        return {
            "reply": (
                f"**Self-evolve plan** for: {task}\n\n"
                f"Suggested files: {', '.join(plan['suggested_files'])}\n"
                f"Current branch: `{status['current_branch']}` · fork remote: `{status['fork_remote']}`\n\n"
                "Use authenticated API:\n"
                "• `POST /api/brain/self/plan` — file suggestions\n"
                "• `POST /api/brain/self/branch` — create fork branch\n"
                "• `POST /api/brain/self/write` — edit source (app/, brain/, src/)\n"
                "• `POST /api/brain/self/commit` — commit locally\n"
                "• `POST /api/brain/self/push` with `approve_push: true` — push fork only\n\n"
                "Main is never pushed without your explicit approval."
            ),
            "kind": "self_evolve",
            "plan": plan,
            "repo": status,
        }
    if cmd.startswith("/agent"):
        topic = message.strip()[6:].strip(" :.")
        if not topic:
            return {
                "reply": "Usage: `/agent What is DNA and how does it relate to genetics?`",
                "kind": "agent",
            }
        agent = run_agent_loop(topic)
        return {
            "reply": agent["answer"],
            "kind": "agent",
            "agent": agent,
            "citations": agent.get("citations", []),
        }
    if cmd.startswith("/research"):
        topic = message.strip()[9:].strip(" ?.")
        if not topic:
            return {
                "reply": "Usage: `/research blood` — cross-domain taxonomy search + Ciper facets.",
                "kind": "research",
            }
        result = ciper_research(topic)
        if not result:
            return {"reply": f"No taxonomy match for «{topic}» yet.", "kind": "research"}
        lines = [
            f"**Subject:** {result.subject}",
            f"**Mode:** {result.mode}",
            f"**Reply:** {result.reply}",
        ]
        if result.domains_spanned:
            lines.append(f"**Domains:** {', '.join(result.domains_spanned)}")
        if result.cross_domain_paths:
            lines.append("**Paths:** " + ", ".join(result.cross_domain_paths[:5]))
        if result.facets:
            lines.append("**Ciper facets:** " + ", ".join(result.facets))
        if result.agi_traits:
            lines.append("**AGI traits:** " + ", ".join(result.agi_traits))
        return {"reply": "\n".join(lines), "kind": "research", "ciper": result.to_dict()}
    if cmd == "/vitals":
        return {"reply": "Open `/security/status` or `/organism/vitals` for live organ vitals.", "kind": "vitals"}
    return None


def chat(message: str, *, session_id: str | None = None) -> dict[str, Any]:
    """Process a chat message — psychology brain wraps algorithm brain output."""
    text = _resolve_followup((message or "").strip(), session_id)

    def done(payload: dict[str, Any]) -> dict[str, Any]:
        out = _finalize(payload, text or (message or ""))
        reply = str(out.get("reply", "")).strip()
        if session_id and reply:
            append_turn(session_id, user=text or (message or ""), assistant=reply)
        return out

    if not text:
        return done({"error": "empty message", "reply": "Send a message or try `/help`."})

    if len(text) > 8000:
        return done(
            {"error": "message too long", "reply": "Please keep messages under 8000 characters."}
        )

    cmd = _command_response(text)
    if cmd:
        payload = {
            "reply": cmd["reply"],
            "kind": cmd["kind"],
            "session_id": session_id,
            "learning": learning_snapshot(),
        }
        if "snapshot" in cmd:
            payload["snapshot"] = cmd["snapshot"]
        if "timeline" in cmd:
            payload["timeline"] = cmd["timeline"]
        if "ciper" in cmd:
            payload["ciper"] = cmd["ciper"]
        if "roadmap" in cmd:
            payload["roadmap"] = cmd["roadmap"]
        if "simulation" in cmd:
            payload["simulation"] = cmd["simulation"]
        if "agent" in cmd:
            payload["agent"] = cmd["agent"]
        if "plan" in cmd:
            payload["plan"] = cmd["plan"]
        if "repo" in cmd:
            payload["repo"] = cmd["repo"]
        if "citations" in cmd:
            payload["citations"] = cmd["citations"]
        return done(payload)

    if is_agent_task(text):
        return done(_agent_payload(text, session_id=session_id))

    from brain.identity_handler import handle_identity, is_identity_question
    from brain.philosophy_handler import handle_philosophy_question, is_philosophy_question

    if is_identity_question(text):
        return done(handle_identity(text, session_id=session_id))

    if is_philosophy_question(text):
        phil = handle_philosophy_question(
            text,
            session_id=session_id,
            predict_fn=_predict_with_timeout,
            classify_fn=_classify_message,
            learning_snapshot_fn=learning_snapshot,
        )
        if phil:
            return done(phil)

    from brain.combinatorial_creation import handle_creation_request, is_creation_request

    if is_creation_request(text):
        return done(handle_creation_request(text, session_id=session_id))

    nl = _simple_nl_response(text)
    if nl:
        return done(
            {
                "reply": to_simple_answer(nl),
                "kind": "chat",
                "session_id": session_id,
                "learning": learning_snapshot(),
                "simple_qa": True,
            }
        )

    deterministic = _deterministic_payload(text, session_id=session_id)
    if deterministic:
        return done(deterministic)

    if is_code_question(text):
        code_payload = _code_payload(text, session_id=session_id)
        if code_payload:
            return done(code_payload)

    if is_prediction_question(text):
        return done(_brain_predict_payload(text, session_id=session_id))

    ciper_payload = _ciper_chat_payload(text, session_id=session_id)
    if ciper_payload:
        return done(ciper_payload)

    if is_simple_question(text):
        simple = _simple_chat_reply(text, session_id=session_id)
        simple["session_id"] = session_id
        simple["learning"] = learning_snapshot()
        return done(simple)

    classification = _classify_message(text)
    if ciper_payload:
        if classification:
            ciper_payload["classification"] = classification
        return done(ciper_payload)

    learning = learning_snapshot()

    with get_session() as session:
        from db.models import Document

        doc_count = session.scalar(select(func.count()).select_from(Document)) or 0
        active = _active_micro_progress(session)

    if classification:
        enriched = f"domain context {classification['label']} question {text.strip().lower()}"
        result = _predict_with_timeout(enriched, session_id=session_id, force=True)
        if result and result.get("answer") and not result.get("abstained"):
            reply = str(result["answer"]).strip()
            if len(reply) > 20 and "philosophy." not in reply.lower():
                return done(
                    {
                        "reply": reply,
                        "kind": "predict",
                        "session_id": session_id,
                        "learning": learning,
                        "brain_predict": True,
                        "classification": classification,
                    }
                )
        from brain.philosophy_handler import philosophy_fallback_if_needed

        fallback = philosophy_fallback_if_needed(text, result)
        if fallback:
            fallback["session_id"] = session_id
            fallback["learning"] = learning
            fallback["classification"] = classification
            return done(fallback)

        reply = (
            f"I mapped your question to **{classification['label']}** "
            f"({classification['confidence']:.0%} confidence) but need more corpus "
            f"in that domain for a full answer.\n\n"
            "Aureon learns by collecting domain text, labeling, training weights, "
            "and graduating grade levels — preschool through doctorate."
        )
    else:
        reply = (
            "No production classifier is promoted yet — I'm still in early grade cycles "
            "(trainer needs ≥2 label classes). Auto-learn is running on Railway; "
            "try `/status` to see the latest cycle or `/grades` for the curriculum timeline.\n\n"
            f"**Corpus:** {doc_count} documents ingested. "
        )
        if active:
            reply += (
                f"**Current focus:** `{active['path']}` @ grade **{active['current_grade']}**."
            )

    return done(
        {
            "reply": reply,
            "kind": "chat",
            "session_id": session_id,
            "classification": classification,
            "learning": learning,
            "active_micro": active,
        }
    )
