"""Self-inquiry — Aureon reflects on what it collected after each grade cycle."""

from __future__ import annotations

import json
import os
import random
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from brain.domains.generate_micros import topics_for
from brain.domains.taxonomy import lookup_names
from brain.grades import get_grade
from brain.ciper_logic import ciper_follow_up_question
from brain.simple_qa import (
    ANSWER_MAX_INQUIRY_DEFAULT,
    ANSWER_MAX_INQUIRY_PRESCHOOL,
    to_simple_answer,
)

_batch_lock = threading.Lock()
_batch_inquiry_count = 0
_batch_inquiry_limit: int | None = None

_GRADE_SNIPPET_LIMITS: dict[str, int] = {
    "preschool": 60,
    "elementary": 90,
    "middle_school": 110,
    "high_school": 120,
    "undergraduate": 120,
    "masters": 120,
    "doctorate": 120,
}

_SEEDED_SELF_PROMPTS: dict[tuple[str, str, str], tuple[str, ...]] = {
    (
        "science_and_natural_philosophy",
        "physics",
        "quantum_mechanics",
    ): ("explain quantum mechanics to me",),
}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int = 100) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, min(maximum, int(raw)))
    except ValueError:
        return default


def is_self_inquiry_enabled() -> bool:
    if _env_bool("AUREON_SELF_INQUIRY", default=False):
        return True
    if os.environ.get("AUREON_SELF_INQUIRY", "").strip().lower() in ("0", "false", "no", "off"):
        return False
    try:
        from app.startup import is_railway

        return is_railway()
    except ImportError:
        return False


def questions_per_target() -> int:
    return _env_int("AUREON_SELF_INQUIRY_QUESTIONS_PER_TARGET", 2, minimum=1, maximum=5)


def max_per_batch() -> int:
    return _env_int("AUREON_SELF_INQUIRY_MAX_PER_BATCH", 25, minimum=5, maximum=200)


def reset_batch_inquiry_budget(limit: int | None = None) -> None:
    global _batch_inquiry_count, _batch_inquiry_limit
    with _batch_lock:
        _batch_inquiry_count = 0
        _batch_inquiry_limit = limit if limit is not None else max_per_batch()


def _take_batch_slot() -> bool:
    global _batch_inquiry_count
    with _batch_lock:
        limit = _batch_inquiry_limit if _batch_inquiry_limit is not None else max_per_batch()
        if _batch_inquiry_count >= limit:
            return False
        _batch_inquiry_count += 1
        return True


def inquiry_log_path() -> Path:
    data_dir = os.environ.get("AUREON_DATA_DIR", "data").strip() or "data"
    path = Path(data_dir) / "self_inquiry.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def append_inquiry(record: dict[str, Any]) -> None:
    line = {**record, "ts": datetime.now(timezone.utc).isoformat()}
    path = inquiry_log_path()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, default=str) + "\n")


def recent_inquiries(limit: int = 20) -> list[dict[str, Any]]:
    path = inquiry_log_path()
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


def _question_templates(grade_slug: str) -> list[str]:
    templates: dict[str, list[str]] = {
        "preschool": [
            "What is {micro_display}?",
            "What is one word I would use to describe {topic}?",
        ],
        "elementary": [
            "Why does {micro_display} live under {sub_display}?",
            "What did my collector region find about {topic}?",
        ],
        "middle_school": [
            "How does {micro_display} connect to other ideas in {domain_display}?",
            "Did my labels for {topic} make sense this cycle?",
        ],
        "high_school": [
            "What pattern do I notice in {micro_display} after training?",
            "If {topic} were wrong, how would my verifier catch it?",
        ],
        "undergraduate": [
            "What evidence supports what I think I know about {micro_display}?",
            "How confident am I in {topic} at {grade_name} level?",
        ],
        "masters": [
            "What gap remains in my understanding of {micro_display}?",
            "How would I teach {topic} to someone at elementary level?",
        ],
        "doctorate": [
            "What original question about {micro_display} do I still cannot answer?",
            "If I mastered {path}, what should I question next in {domain_display}?",
        ],
    }
    return templates.get(grade_slug, templates["elementary"])


def _seeded_self_prompts(
    domain_slug: str,
    subdomain_slug: str,
    micro_slug: str,
) -> tuple[str, ...]:
    return _SEEDED_SELF_PROMPTS.get((domain_slug, subdomain_slug, micro_slug), ())


def generate_questions(
    *,
    domain_slug: str,
    subdomain_slug: str,
    micro_slug: str,
    grade_slug: str,
    count: int,
) -> list[str]:
    return [item["question"] for item in generate_question_items(
        domain_slug=domain_slug,
        subdomain_slug=subdomain_slug,
        micro_slug=micro_slug,
        grade_slug=grade_slug,
        count=count,
    )]


def generate_question_items(
    *,
    domain_slug: str,
    subdomain_slug: str,
    micro_slug: str,
    grade_slug: str,
    count: int,
) -> list[dict[str, Any]]:
    names = lookup_names(domain_slug, subdomain=subdomain_slug, micro=micro_slug)
    grade = get_grade(grade_slug)
    grade_name = grade.name if grade else grade_slug.replace("_", " ").title()
    leaf_topics = topics_for(domain_slug, subdomain_slug, micro_slug)
    topic = random.choice(leaf_topics) if leaf_topics else names.get("micro_subdomain", micro_slug)

    ctx = {
        "domain_display": names.get("domain", domain_slug),
        "sub_display": names.get("subdomain", subdomain_slug),
        "micro_display": names.get("micro_subdomain", micro_slug),
        "topic": topic,
        "grade_name": grade_name,
        "path": f"{domain_slug}.{subdomain_slug}.{micro_slug}",
    }

    seeded = [
        {**ctx, "question": question, "seeded": True}
        for question in _seeded_self_prompts(domain_slug, subdomain_slug, micro_slug)
    ][:count]
    remaining = max(0, count - len(seeded))
    pool = _question_templates(grade_slug)
    random.shuffle(pool)
    chosen = pool[:remaining]
    items = seeded + [{**ctx, "question": q.format(**ctx)} for q in chosen]

    ciper_q = ciper_follow_up_question(ctx["micro_display"])
    if ciper_q and len(items) > len(seeded):
        items[-1] = {**ctx, "question": ciper_q, "ciper": True}

    return items


def _snippet(text: str, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return ""
    if len(cleaned) <= limit:
        return cleaned
    cut = cleaned[:limit].rsplit(" ", 1)[0]
    return (cut or cleaned[:limit]).rstrip(".,;:") + "…"


def _snippet_limit(grade_slug: str) -> int:
    return _GRADE_SNIPPET_LIMITS.get(grade_slug, 280)


def fetch_learning_context(
    domain_slug: str,
    subdomain_slug: str,
    micro_slug: str,
    *,
    doc_limit: int = 3,
) -> dict[str, Any]:
    """Load verified documents and label summary for a micro-topic path."""
    from db.models import Document, DocumentLabel
    from db.seed import get_micro_subdomain
    from db.session import get_session

    out: dict[str, Any] = {"documents": [], "labels": []}
    try:
        with get_session() as session:
            micro = get_micro_subdomain(session, domain_slug, subdomain_slug, micro_slug)
            if not micro:
                return out

            docs = session.scalars(
                select(Document)
                .where(Document.micro_subdomain_id == micro.id, Document.verified.is_(True))
                .order_by(Document.quality_score.desc().nullslast(), Document.id.desc())
                .limit(doc_limit)
            ).all()
            if not docs:
                docs = session.scalars(
                    select(Document)
                    .where(Document.micro_subdomain_id == micro.id)
                    .order_by(Document.quality_score.desc().nullslast(), Document.id.desc())
                    .limit(doc_limit)
                ).all()

            for doc in docs:
                out["documents"].append(
                    {
                        "title": doc.title,
                        "text": doc.text,
                        "source": doc.source,
                        "quality_score": doc.quality_score,
                        "topic": (doc.extra or {}).get("topic"),
                    }
                )

            label_rows = session.execute(
                select(
                    DocumentLabel.label,
                    func.count(DocumentLabel.id),
                    func.avg(DocumentLabel.confidence),
                )
                .join(Document, DocumentLabel.document_id == Document.id)
                .where(Document.micro_subdomain_id == micro.id)
                .group_by(DocumentLabel.label)
                .order_by(func.count(DocumentLabel.id).desc())
            ).all()
            for label, count, avg_conf in label_rows:
                out["labels"].append(
                    {
                        "label": label,
                        "count": int(count),
                        "avg_confidence": float(avg_conf or 0.0),
                    }
                )
    except Exception:
        return out
    return out


def _one_word(topic: str, learning: dict[str, Any], ctx: dict[str, str]) -> str:
    labels = learning.get("labels") or []
    if labels:
        return str(labels[0]["label"]).replace("_", " ")
    words = [w for w in re.findall(r"[A-Za-z]{4,}", topic) if w.lower() not in {"what", "about", "with"}]
    if words:
        return words[-1].lower()
    micro = ctx.get("micro_display", "learning")
    return micro.split()[0].lower() if micro else "learning"


def _content_answer(
    question: str,
    *,
    learning: dict[str, Any],
    ctx: dict[str, str],
    grade_slug: str,
) -> str:
    """Simple Question, Simple Answer — one short line from collected data."""
    q = question.lower()
    docs: list[dict[str, Any]] = learning.get("documents") or []
    labels: list[dict[str, Any]] = learning.get("labels") or []
    topic = ctx.get("topic", "")
    micro = ctx.get("micro_display", "this topic")
    sub = ctx.get("sub_display", "its subdomain")
    domain = ctx.get("domain_display", "its domain")
    limit = _snippet_limit(grade_slug)

    if "one word" in q:
        return _one_word(topic, learning, ctx)

    if "what type of" in q:
        if docs:
            return _snippet(docs[0]["text"], limit)
        if labels:
            return labels[0]["label"]
        return f"Pick one facet of {topic}."

    if "collector" in q and "find" in q:
        if docs:
            titles = ", ".join(d["title"] for d in docs[:2])
            return f"{len(docs)} docs: {titles}."
        return f"No docs for {micro} yet."

    if "labels" in q and "make sense" in q:
        if labels:
            top = labels[0]
            return f"Top label: {top['label']} ({top['count']} docs)."
        return f"No labels for {topic} yet."

    if "live under" in q or "why does" in q:
        return f"{micro} is under {sub}."

    if "connect to other ideas" in q:
        if docs:
            return _snippet(docs[0]["text"], limit)
        return f"{micro} is in {domain}."

    if "pattern" in q and "after training" in q:
        if labels:
            top = labels[0]
            return f"Pattern: {top['label']} ({top['avg_confidence']:.0%})."
        return f"No label pattern for {micro} yet."

    if "verifier catch" in q or "were wrong" in q:
        if docs:
            return f"{len(docs)} docs scored by verifier."
        return "Verifier needs documents first."

    if "evidence supports" in q:
        if docs:
            return _snippet(docs[0]["text"], limit)
        return f"No evidence for {micro} yet."

    if "how confident" in q:
        if labels:
            top = max(labels, key=lambda row: row["avg_confidence"])
            return f"{top['avg_confidence']:.0%} on {top['label']}."
        return f"No confidence score for {topic} yet."

    if "gap remains" in q or "still cannot answer" in q:
        if docs:
            return f"Only {len(docs)} doc(s) so far."
        return f"No docs for {micro} yet."

    if "how would i teach" in q:
        if docs:
            return _snippet(docs[0]["text"], min(limit, 90))
        return f"Teach {topic} from taxonomy only."

    if "what should i question next" in q:
        return f"Next: links between {micro} and {domain}."

    if docs:
        return _snippet(docs[0]["text"], limit)

    return f"No verified text for {micro} yet."


def _cycle_note(outcome: dict[str, Any]) -> str:
    """Short pipeline note — stored separately from the simple answer."""
    graduation = outcome.get("graduation") or {}
    grade_name = outcome.get("grade_name") or outcome.get("grade", "this grade")
    passed = graduation.get("passed")
    unlocked = graduation.get("unlocked_next")
    train_acc = graduation.get("train_accuracy")

    if passed:
        note = f"Passed {grade_name}"
        if unlocked:
            note += f", unlocked {unlocked.replace('_', ' ')}"
    else:
        note = f"Retry {grade_name}"

    if train_acc is not None and train_acc > 0:
        note += f", {train_acc:.0%} acc"
    return note


def answer_question(
    question: str,
    *,
    outcome: dict[str, Any],
    learning: dict[str, Any] | None = None,
    ctx: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Simple Question, Simple Answer — content line + separate cycle note."""
    grade_slug = str(outcome.get("grade") or "preschool")
    learning = learning or {}
    ctx = ctx or {}
    content = _content_answer(question, learning=learning, ctx=ctx, grade_slug=grade_slug)
    max_len = ANSWER_MAX_INQUIRY_PRESCHOOL if grade_slug == "preschool" else ANSWER_MAX_INQUIRY_DEFAULT
    if "one word" in question.lower():
        return content, _cycle_note(outcome)
    return to_simple_answer(content, max_len=max_len), _cycle_note(outcome)


def run_self_inquiry_for_cycle(outcome: dict[str, Any], *, source: str = "auto_learn") -> list[dict[str, Any]]:
    """After a grade cycle, reflect on what was collected and measured."""
    from app.activity_log import log_ai_activity

    if not is_self_inquiry_enabled():
        return []
    if outcome.get("error") or outcome.get("fully_graduated"):
        return []
    if not _take_batch_slot():
        return []

    domain = outcome["domain"]
    subdomain = outcome["subdomain"]
    micro = outcome["micro_subdomain"]
    grade = outcome.get("grade") or "preschool"
    path = f"{domain}.{subdomain}.{micro}"

    learning = fetch_learning_context(domain, subdomain, micro)

    exchanges: list[dict[str, Any]] = []
    for item in generate_question_items(
        domain_slug=domain,
        subdomain_slug=subdomain,
        micro_slug=micro,
        grade_slug=grade,
        count=questions_per_target(),
    ):
        question = item["question"]
        ctx = {k: v for k, v in item.items() if k != "question"}
        answer, cycle = answer_question(question, outcome=outcome, learning=learning, ctx=ctx)
        record = {
            "question": question,
            "answer": answer,
            "cycle": cycle,
            "path": path,
            "grade": grade,
            "source": source,
            "graduation_passed": (outcome.get("graduation") or {}).get("passed"),
            "grounded": bool(learning.get("documents")),
            "ciper": bool(item.get("ciper")),
        }
        append_inquiry(record)
        exchanges.append(record)
        log_ai_activity(
            "self_inquiry",
            source=source,
            path=path,
            grade=grade,
            question=question,
            answer=answer[:500],
            grounded=record["grounded"],
        )

    return exchanges
