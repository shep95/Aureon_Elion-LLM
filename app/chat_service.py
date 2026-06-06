"""Chat with Aureon — supervised inference + live learning context."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.auto_learn import get_auto_learn_scheduler
from brain.cortex import brain_status
from brain.domains.taxonomy import total_micro_subdomains
from brain.grades import GRADE_CURRICULUM, curriculum_public, epochs_for_grade, get_grade
from brain.graduation import current_grade, progress_report
from brain.ciper_logic import ciper_research
from brain.predict_engine import is_prediction_question, predict_with_steps
from brain.psychology_brain import finalize_chat_payload
from brain.self_inquiry import is_self_inquiry_enabled, recent_inquiries
from brain.simple_qa import is_simple_question, to_simple_answer
from db.models import KnowledgeDomain, KnowledgeMicroSubdomain, KnowledgeSubdomain
from db.session import get_session
from pipeline.step4_evaluation.benchmarks import _load_production_model, _predict_label
from sqlalchemy import func, select
from sqlalchemy.orm import Session


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


def learning_snapshot() -> dict[str, Any]:
    scheduler = get_auto_learn_scheduler()
    status = scheduler.status()
    brain = brain_status()
    timeline = estimate_learning_timeline(
        interval_sec=status.get("config", {}).get("interval_sec", 3600),
        max_grades_per_cycle=status.get("config", {}).get("max_grades_per_cycle", 1),
        micro_subdomain_count=brain.get("micro_subdomains", total_micro_subdomains()),
    )
    return {
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
    }


def _brain_predict_payload(text: str, *, session_id: str | None) -> dict[str, Any] | None:
    """Attention LM — embed, attend, predict next tokens autoregressively."""
    result = predict_with_steps(text)
    if not result:
        return None
    return {
        "reply": result["answer"],
        "kind": "chat",
        "session_id": session_id,
        "learning": learning_snapshot(),
        "prediction": {
            "model": result["model"],
            "pipeline": result["pipeline"],
            "prompt": result["prompt"],
        },
        "brain_predict": True,
    }


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

    if q in ("what is aureon", "who are you", "what are you"):
        return "Supervised ML brain — collect, label, train, evaluate, graduate."

    if "how do you work" in q or "how does aureon work" in q:
        return "Inputs and labels in, backpropagation finds weights, measurable accuracy out."

    if q in ("what is ai", "what is artificial intelligence"):
        return "Supervised machine learning — labels plus weights, not magic."

    if "what are you asking" in q or "questions do you ask" in q:
        items = recent_inquiries(1)
        if not items:
            return "No reflections yet — wait for the next auto-learn cycle."
        item = items[0]
        return f"{item.get('question')} → {item.get('answer')}"

    return None


def _simple_chat_reply(text: str) -> dict[str, Any]:
    """Simple Question, Simple Answer path for chat."""
    nl = _simple_nl_response(text)
    if nl:
        return {"reply": to_simple_answer(nl), "kind": "chat", "simple_qa": True}

    classification = _classify_message(text)
    if classification:
        label = classification["label"]
        conf = classification["confidence"]
        return {
            "reply": to_simple_answer(f"{label} ({conf:.0%} confidence)"),
            "kind": "chat",
            "simple_qa": True,
            "classification": classification,
        }

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
        items = recent_inquiries(6)
        if not items:
            return {
                "reply": (
                    "No learning reflections yet. When auto-learn runs, I cite collected "
                    "documents after each grade cycle — check back after the first cycle or "
                    "filter Railway logs for `self_inquiry`."
                ),
                "kind": "mind",
            }
        lines = ["**Learning reflections** (Simple Question, Simple Answer):\n"]
        for item in items:
            lines.append(f"**Q:** {item.get('question')}")
            answer = str(item.get("answer", ""))
            cycle = item.get("cycle")
            if cycle:
                answer = f"{answer} ({cycle})"
            if len(answer) > 200:
                answer = answer[:197] + "..."
            lines.append(f"**A:** {answer}\n")
        return {"reply": "\n".join(lines).strip(), "kind": "mind", "inquiries": items}
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
    text = (message or "").strip()
    if not text:
        return finalize_chat_payload(
            {"error": "empty message", "reply": "Send a message or try `/help`."},
            text,
        )

    if len(text) > 8000:
        return finalize_chat_payload(
            {"error": "message too long", "reply": "Please keep messages under 8000 characters."},
            text,
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
        return finalize_chat_payload(payload, text)

    nl = _simple_nl_response(text)
    if nl:
        return finalize_chat_payload(
            {
                "reply": to_simple_answer(nl),
                "kind": "chat",
                "session_id": session_id,
                "learning": learning_snapshot(),
                "simple_qa": True,
            },
            text,
        )

    if is_prediction_question(text):
        predict_payload = _brain_predict_payload(text, session_id=session_id)
        if predict_payload:
            return finalize_chat_payload(predict_payload, text)

    ciper_payload = _ciper_chat_payload(text, session_id=session_id)
    if ciper_payload:
        return finalize_chat_payload(ciper_payload, text)

    if is_simple_question(text):
        simple = _simple_chat_reply(text)
        simple["session_id"] = session_id
        simple["learning"] = learning_snapshot()
        return finalize_chat_payload(simple, text)

    classification = _classify_message(text)
    ciper_payload = _ciper_chat_payload(text, session_id=session_id)
    if ciper_payload:
        if classification:
            ciper_payload["classification"] = classification
        return finalize_chat_payload(ciper_payload, text)

    learning = learning_snapshot()

    with get_session() as session:
        from db.models import Document

        doc_count = session.scalar(select(func.count()).select_from(Document)) or 0
        active = _active_micro_progress(session)

    if classification:
        reply = (
            f"I classified your message as **{classification['label']}** "
            f"({classification['confidence']:.0%} confidence) using the production supervised model.\n\n"
            "Aureon learns by collecting domain text, labeling, training weights via backpropagation, "
            "and graduating grade levels — preschool through doctorate — on each micro-topic."
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

    return finalize_chat_payload(
        {
            "reply": reply,
            "kind": "chat",
            "session_id": session_id,
            "classification": classification,
            "learning": learning,
            "active_micro": active,
        },
        text,
    )
