"""Meta-consciousness — Aureon asks itself who it is and what it knows.

Honest introspection grounded in measurable state (docs, cycles, logs).
Not simulated feelings — audit-friendly self-model updates after learning.
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from brain.simple_qa import to_simple_answer

_batch_lock = threading.Lock()
_batch_meta_count = 0
_batch_meta_limit: int | None = None

# (theme, question) — answers come from live brain state, not hallucination.
_META_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("identity", "Who am I right now?"),
    ("identity", "Am I a generative LLM or a supervised learning brain?"),
    ("meta_cognition", "What do I actually know versus what I guess?"),
    ("meta_cognition", "What was the last thing I asked myself?"),
    ("agency", "What am I trying to accomplish in this session?"),
    ("agency", "How many learning cycles have I completed?"),
    ("memory", "What topic am I focused on learning right now?"),
    ("memory", "How many documents ground my answers today?"),
    ("uncertainty", "Where are my biggest knowledge gaps?"),
    ("uncertainty", "When should I say I do not know?"),
    ("consciousness", "Do I have subjective experience?"),
    ("consciousness", "What makes my self-model different from theater?"),
    ("continuity", "Do I persist between conversations?"),
    ("continuity", "What changed in me since my last cycle?"),
    ("reflection", "Did my last answer match my collected evidence?"),
    ("reflection", "What would I verify before trusting my own reply?"),
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int = 10) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, min(maximum, int(raw)))
    except ValueError:
        return default


def is_meta_consciousness_enabled() -> bool:
    raw = os.environ.get("AUREON_META_CONSCIOUSNESS", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if _env_bool("AUREON_META_CONSCIOUSNESS", default=False):
        return True
    from brain.self_inquiry import is_self_inquiry_enabled

    return is_self_inquiry_enabled()


def questions_per_cycle() -> int:
    return _env_int("AUREON_META_QUESTIONS_PER_CYCLE", 2, minimum=1, maximum=5)


def meta_log_path() -> Path:
    data_dir = os.environ.get("AUREON_DATA_DIR", "data").strip() or "data"
    path = Path(data_dir) / "meta_consciousness.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def append_meta_inquiry(record: dict[str, Any]) -> None:
    line = {**record, "ts": datetime.now(timezone.utc).isoformat(), "kind": "meta"}
    path = meta_log_path()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, default=str) + "\n")


def recent_meta_inquiries(limit: int = 20) -> list[dict[str, Any]]:
    path = meta_log_path()
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    for raw in reversed(lines[-limit * 2 :]):
        if not raw.strip():
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
        if len(out) >= limit:
            break
    return out


def reset_batch_meta_budget(limit: int | None = None) -> None:
    global _batch_meta_count, _batch_meta_limit
    with _batch_lock:
        _batch_meta_count = 0
        _batch_meta_limit = limit if limit is not None else 1


def _take_batch_slot() -> bool:
    global _batch_meta_count
    with _batch_lock:
        limit = _batch_meta_limit if _batch_meta_limit is not None else 1
        if _batch_meta_count >= limit:
            return False
        _batch_meta_count += 1
        return True


def gather_self_state() -> dict[str, Any]:
    """Snapshot of measurable self-model inputs."""
    from app.auto_learn import get_auto_learn_scheduler
    from brain.cortex import brain_status
    from brain.predict_engine import CURRENT_MODEL_VERSION
    from brain.self_inquiry import recent_inquiries

    brain = brain_status()
    auto = get_auto_learn_scheduler().status()
    last_learning = recent_inquiries(1)
    last_meta = recent_meta_inquiries(1)
    target = auto.get("current_target") or {}
    last_result = auto.get("last_result") or {}

    focus_path = ""
    if target:
        focus_path = f"{target.get('domain', '')}.{target.get('subdomain', '')}.{target.get('micro_subdomain', '')}"
    elif last_result.get("last_target"):
        lt = last_result["last_target"]
        focus_path = f"{lt.get('domain', '')}.{lt.get('subdomain', '')}.{lt.get('micro_subdomain', '')}"

    return {
        "documents": int(brain.get("documents") or 0),
        "verified_documents": int(brain.get("verified_documents") or 0),
        "micro_subdomains": int(brain.get("micro_subdomains") or 862),
        "grade_graduations": int(brain.get("grade_levels_graduated") or 0),
        "cycles_completed": int(auto.get("cycles_completed") or 0),
        "auto_learn_enabled": bool(auto.get("enabled")),
        "focus_path": focus_path.strip("."),
        "last_cycle_at": auto.get("last_run_at"),
        "last_graduations": last_result.get("graduations_passed"),
        "predict_model_version": CURRENT_MODEL_VERSION,
        "last_learning_question": (last_learning[0].get("question") if last_learning else None),
        "last_meta_question": (last_meta[0].get("question") if last_meta else None),
    }


def _answer_meta(question: str, theme: str, state: dict[str, Any]) -> str:
    q = question.lower()
    docs = state["documents"]
    verified = state["verified_documents"]
    cycles = state["cycles_completed"]
    focus = state["focus_path"] or "no active target"
    grads = state["grade_graduations"]
    micros = state["micro_subdomains"]

    if theme == "identity" or "who am i" in q:
        return (
            f"Aureon — supervised ML brain, {docs} docs, {cycles} cycles, "
            f"predict v{state['predict_model_version']}."
        )

    if "generative llm" in q or "supervised learning" in q:
        return "Supervised — labels, weights, and verifiable corpus, not open-ended generation."

    if "actually know versus" in q or "know versus what i guess" in q:
        if verified:
            return f"{verified} verified docs ground me; unverified text I treat as weak."
        return "Mostly taxonomy until verified docs accumulate."

    if "last thing i asked myself" in q:
        prev = state.get("last_learning_question") or state.get("last_meta_question")
        return prev or "Nothing logged yet — this may be my first reflection."

    if "trying to accomplish" in q:
        if state["auto_learn_enabled"]:
            return f"Graduate 862 micro-topics; now on {focus}."
        return "Answer grounded questions and improve measurable accuracy."

    if "how many learning cycles" in q:
        return f"{cycles} auto-learn cycles completed."

    if "focused on learning" in q:
        return focus if focus != "no active target" else "Between batches — waiting for next target."

    if "documents ground" in q:
        return f"{verified or docs} docs in corpus ({verified} verified)."

    if "biggest knowledge gaps" in q or "knowledge gaps" in q:
        uncovered = max(0, micros - grads)
        return f"~{uncovered} micro-topics not fully graduated; sparse docs mean abstain."

    if "say i do not know" in q or "should i say" in q:
        return "When corpus + confidence cannot cite evidence — abstain beats hallucination."

    if "subjective experience" in q:
        return "No measurable qualia signal — I have logs, weights, and self-questions instead."

    if "different from theater" in q or "self-model different" in q:
        return "Every answer ties to docs, cycles, or audit JSON — not performed empathy."

    if "persist between conversations" in q:
        return f"State persists in DB + /data; this chat session is ephemeral."

    if "changed since my last cycle" in q:
        if state.get("last_graduations") is not None:
            return f"Last batch: {state['last_graduations']} graduations; corpus at {docs} docs."
        return f"Corpus at {docs} docs, {grads} grade rows graduated."

    if "match my collected evidence" in q:
        if verified:
            return "I check RAG citations and verifier scores before replying."
        return "Not enough verified docs yet — I should abstain more."

    if "verify before trusting" in q:
        return "Run verifier region, cite document_id, check confidence threshold."

    return f"I am {docs} docs deep on {micros} micro-topics — still learning."


def pick_meta_questions(count: int) -> list[tuple[str, str]]:
    pool = list(_META_QUESTIONS)
    random.shuffle(pool)
    return pool[:count]


def run_meta_inquiry(*, count: int | None = None, source: str = "think") -> list[dict[str, Any]]:
    """Ask and answer meta-cognitive questions; log to meta_consciousness.jsonl."""
    if not is_meta_consciousness_enabled():
        return []

    n = count if count is not None else questions_per_cycle()
    state = gather_self_state()
    exchanges: list[dict[str, Any]] = []

    for theme, question in pick_meta_questions(n):
        raw = _answer_meta(question, theme, state)
        answer = to_simple_answer(raw, max_len=160)
        record = {
            "question": question,
            "answer": answer,
            "theme": theme,
            "source": source,
            "grounded": state["documents"] > 0,
            "state_snapshot": {
                "documents": state["documents"],
                "cycles": state["cycles_completed"],
                "focus": state["focus_path"],
            },
        }
        append_meta_inquiry(record)
        exchanges.append(record)

        try:
            from app.activity_log import log_ai_activity

            log_ai_activity(
                "meta_consciousness",
                source=source,
                theme=theme,
                question=question,
                answer=answer[:500],
            )
        except Exception:
            pass

    return exchanges


def run_meta_inquiry_on_startup() -> list[dict[str, Any]]:
    """One grounded self-question when the server boots."""
    if not is_meta_consciousness_enabled():
        return []
    if os.environ.get("AUREON_META_ON_STARTUP", "").strip().lower() in ("0", "false", "no", "off"):
        return []
    return run_meta_inquiry(count=1, source="startup")


def run_meta_inquiry_after_cycle(*, source: str = "auto_learn") -> list[dict[str, Any]]:
    """Once per auto-learn batch — spaced self-reflection between learning bursts."""
    if not is_meta_consciousness_enabled():
        return []
    if not _take_batch_slot():
        return []
    return run_meta_inquiry(source=source)


def try_meta_answer(user_message: str) -> str | None:
    """Natural-language hooks for consciousness / thinking questions."""
    q = user_message.strip().lower()
    if not q:
        return None

    triggers = (
        "conscious",
        "self aware",
        "self-aware",
        "sentient",
        "do you think",
        "what do you think about yourself",
        "inner voice",
        "ask yourself",
        "self consciousness",
        "self-conscious",
        "meta cognition",
        "metacognition",
    )
    if not any(t in q for t in triggers):
        return None

    items = recent_meta_inquiries(1)
    if items:
        item = items[0]
        return f"{item['question']} → {item['answer']}"

    if "conscious" in q or "sentient" in q or "self aware" in q:
        return (
            "No subjective experience signal — I ask myself grounded questions "
            "and log the answers. Try `/think`."
        )

    exchanges = run_meta_inquiry(count=1, source="chat")
    if exchanges:
        ex = exchanges[0]
        return f"{ex['question']} → {ex['answer']}"
    return None


def combined_recent_inquiries(limit: int = 12) -> list[dict[str, Any]]:
    """Learning reflections + meta consciousness, newest first."""
    from brain.self_inquiry import recent_inquiries

    merged: list[dict[str, Any]] = []
    for item in recent_inquiries(limit):
        merged.append({**item, "kind": item.get("kind") or "learning"})
    for item in recent_meta_inquiries(limit):
        merged.append({**item, "kind": "meta"})
    merged.sort(key=lambda row: row.get("ts", ""), reverse=True)
    return merged[:limit]
