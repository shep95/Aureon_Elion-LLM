"""Export Aureon learning state as JSON files safe for public GitHub sync."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auto_learn import get_auto_learn_scheduler, load_target_cursor
from app.chat_service import learning_snapshot
from brain.meta_consciousness import meta_log_path
from brain.self_inquiry import inquiry_log_path
from db.models import (
    BenchmarkResult,
    Document,
    DocumentLabel,
    GradeProgress,
    KnowledgeDomain,
    KnowledgeMicroSubdomain,
    KnowledgeSubdomain,
    PipelineEvent,
    PreferencePair,
    TrainingRun,
)
from db.session import get_session
from pipeline.config import MODELS_DIR, REGISTRY_DIR

CORPUS_PREFIX = "learning-corpus"
MAX_DOCUMENTS = int(os.environ.get("AUREON_GITHUB_SYNC_MAX_DOCS", "5000"))
MAX_EVENTS = int(os.environ.get("AUREON_GITHUB_SYNC_MAX_EVENTS", "500"))
MAX_MODEL_FILES = int(os.environ.get("AUREON_GITHUB_SYNC_MAX_MODELS", "150"))
MAX_MODEL_BYTES = int(os.environ.get("AUREON_GITHUB_SYNC_MAX_MODEL_BYTES", "2_000_000"))
INCLUDE_TEXT = os.environ.get("AUREON_GITHUB_SYNC_INCLUDE_TEXT", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def _doc_path(extra: dict[str, Any] | None) -> str | None:
    if not extra:
        return None
    domain = extra.get("domain")
    sub = extra.get("subdomain")
    micro = extra.get("micro_subdomain")
    if domain and sub and micro:
        return f"{domain}.{sub}.{micro}"
    return None


def _grade_progress_rows(session: Session) -> list[dict[str, Any]]:
    rows = session.execute(
        select(
            GradeProgress,
            KnowledgeDomain.slug,
            KnowledgeSubdomain.slug,
            KnowledgeMicroSubdomain.slug,
            KnowledgeMicroSubdomain.name,
        )
        .join(KnowledgeDomain, GradeProgress.domain_id == KnowledgeDomain.id)
        .join(KnowledgeSubdomain, GradeProgress.subdomain_id == KnowledgeSubdomain.id)
        .join(KnowledgeMicroSubdomain, GradeProgress.micro_subdomain_id == KnowledgeMicroSubdomain.id)
        .order_by(GradeProgress.micro_subdomain_id, GradeProgress.grade_order)
    ).all()

    out: list[dict[str, Any]] = []
    for progress, domain_slug, sub_slug, micro_slug, micro_name in rows:
        if progress.status not in ("graduated", "in_progress", "failed", "unlocked"):
            continue
        out.append(
            {
                "path": f"{domain_slug}.{sub_slug}.{micro_slug}",
                "micro_name": micro_name,
                "grade": progress.grade_slug,
                "status": progress.status,
                "attempts": progress.attempts,
                "metrics": progress.metrics,
                "graduated_at": _iso(progress.graduated_at),
                "last_attempt_at": _iso(progress.last_attempt_at),
            }
        )
    return out


def _graduation_summary(grade_progress: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per successful graduation with training metrics."""
    summary: list[dict[str, Any]] = []
    for row in grade_progress:
        if row["status"] != "graduated":
            continue
        metrics = row.get("metrics") or {}
        summary.append(
            {
                "path": row["path"],
                "micro_name": row["micro_name"],
                "grade": row["grade"],
                "train_accuracy": metrics.get("train_accuracy"),
                "trainer_status": metrics.get("trainer_status"),
                "evaluator": metrics.get("evaluator"),
                "graduated_at": row.get("graduated_at"),
            }
        )
    return summary


def _document_row(doc: Document) -> dict[str, Any]:
    extra = doc.extra or {}
    row: dict[str, Any] = {
        "id": doc.id,
        "path": _doc_path(extra),
        "source": doc.source,
        "title": doc.title,
        "url": doc.url,
        "language": doc.language,
        "verified": doc.verified,
        "quality_score": doc.quality_score,
        "topic": extra.get("topic"),
        "domain": extra.get("domain"),
        "subdomain": extra.get("subdomain"),
        "micro_subdomain": extra.get("micro_subdomain"),
        "grade": extra.get("grade"),
        "metadata": extra,
        "created_at": _iso(doc.created_at),
    }
    if INCLUDE_TEXT:
        row["text"] = doc.text
    return row


def _documents(session: Session) -> list[dict[str, Any]]:
    docs = session.scalars(
        select(Document).order_by(Document.id.asc()).limit(MAX_DOCUMENTS)
    ).all()
    return [_document_row(doc) for doc in docs]


def _documents_jsonl(documents: list[dict[str, Any]]) -> bytes:
    lines = [json.dumps(row, ensure_ascii=False) for row in documents]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def _document_labels(session: Session) -> list[dict[str, Any]]:
    rows = session.execute(
        select(DocumentLabel, Document)
        .join(Document, DocumentLabel.document_id == Document.id)
        .order_by(DocumentLabel.id.asc())
        .limit(MAX_DOCUMENTS)
    ).all()
    out: list[dict[str, Any]] = []
    for label, doc in rows:
        extra = doc.extra or {}
        out.append(
            {
                "document_id": doc.id,
                "path": _doc_path(extra),
                "title": doc.title,
                "topic": extra.get("topic"),
                "label": label.label,
                "confidence": label.confidence,
                "label_source": label.label_source,
                "needs_review": label.needs_review,
                "created_at": _iso(label.created_at),
            }
        )
    return out


def _learned_corpus_jsonl(
    documents: list[dict[str, Any]], labels: list[dict[str, Any]]
) -> bytes:
    """Unified learned knowledge: document + label + topic path."""
    labels_by_doc: dict[int, list[dict[str, Any]]] = {}
    for lbl in labels:
        labels_by_doc.setdefault(lbl["document_id"], []).append(lbl)

    lines: list[str] = []
    for doc in documents:
        doc_labels = labels_by_doc.get(doc["id"], [])
        row = {
            "path": doc.get("path"),
            "topic": doc.get("topic"),
            "title": doc.get("title"),
            "source": doc.get("source"),
            "verified": doc.get("verified"),
            "quality_score": doc.get("quality_score"),
            "labels": doc_labels,
            "text": doc.get("text") if INCLUDE_TEXT else None,
            "url": doc.get("url"),
            "created_at": doc.get("created_at"),
        }
        lines.append(json.dumps(row, ensure_ascii=False))
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def _training_runs(session: Session) -> list[dict[str, Any]]:
    runs = session.scalars(select(TrainingRun).order_by(TrainingRun.id.desc()).limit(500)).all()
    return [
        {
            "run_id": run.run_id,
            "domain_id": run.domain_id,
            "subdomain_id": run.subdomain_id,
            "metrics": run.metrics,
            "artifact_path": run.artifact_path,
            "params": run.params,
            "promoted": run.promoted,
            "created_at": _iso(run.created_at),
        }
        for run in runs
    ]


def _benchmarks(session: Session) -> list[dict[str, Any]]:
    rows = session.scalars(
        select(BenchmarkResult).order_by(BenchmarkResult.id.desc()).limit(500)
    ).all()
    return [
        {
            "domain_id": row.domain_id,
            "benchmark_type": row.benchmark_type,
            "score": row.score,
            "passed": row.passed,
            "details": row.details,
            "created_at": _iso(row.created_at),
        }
        for row in rows
    ]


def _preference_pairs(session: Session) -> list[dict[str, Any]]:
    rows = session.scalars(
        select(PreferencePair).order_by(PreferencePair.id.desc()).limit(500)
    ).all()
    return [
        {
            "domain_id": row.domain_id,
            "context": row.context,
            "preferred": row.preferred,
            "rejected": row.rejected,
            "created_at": _iso(row.created_at),
        }
        for row in rows
    ]


def _pipeline_events(session: Session) -> list[dict[str, Any]]:
    rows = session.scalars(
        select(PipelineEvent).order_by(PipelineEvent.id.desc()).limit(MAX_EVENTS)
    ).all()
    return [
        {
            "step": row.step,
            "event_type": row.event_type,
            "domain_id": row.domain_id,
            "payload": row.payload,
            "created_at": _iso(row.created_at),
        }
        for row in rows
    ]


def _resolve_artifact(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    candidate = Path(path_str)
    if candidate.is_file():
        return candidate
    data_dir = Path(os.environ.get("AUREON_DATA_DIR", "data").strip() or "data")
    alt = data_dir / "models" / candidate.name
    if alt.is_file():
        return alt
    if candidate.name.startswith("brain_"):
        alt2 = MODELS_DIR / candidate.parent.name / candidate.name
        if alt2.is_file():
            return alt2
        alt3 = MODELS_DIR / candidate.name
        if alt3.is_file():
            return alt3
    return None


def _model_artifacts(training_runs: list[dict[str, Any]]) -> dict[str, bytes]:
    """Include trained classifier weights + metadata referenced by training runs."""
    files: dict[str, bytes] = {}
    count = 0
    for run in training_runs:
        if count >= MAX_MODEL_FILES:
            break
        artifact_path = run.get("artifact_path")
        if not artifact_path:
            continue
        model_file = _resolve_artifact(artifact_path)
        if not model_file or not model_file.is_file():
            continue
        scope = (run.get("params") or {}).get("scope") or "unknown"
        run_id = run.get("run_id") or model_file.parent.name
        prefix = f"{CORPUS_PREFIX}/models/{scope}_{run_id}"

        for name in ("classifier.json", "metadata.json"):
            path = model_file.parent / name
            if not path.is_file():
                continue
            size = path.stat().st_size
            if size > MAX_MODEL_BYTES:
                continue
            files[f"{prefix}/{name}"] = path.read_bytes()
            count += 1
            if count >= MAX_MODEL_FILES:
                break
    return files


def _registry_files() -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for name in ("registry.json", "production.json"):
        path = REGISTRY_DIR / name
        if path.is_file() and path.stat().st_size <= MAX_MODEL_BYTES:
            files[f"{CORPUS_PREFIX}/{name}"] = path.read_bytes()
    return files


def build_export_files() -> dict[str, bytes]:
    """Build path → file bytes under learning-corpus/."""
    exported_at = datetime.now(timezone.utc).isoformat()
    snapshot = learning_snapshot()
    snapshot["exported_at"] = exported_at
    snapshot["github_sync"] = {
        "batch_cursor": load_target_cursor(),
        "include_document_text": INCLUDE_TEXT,
        "max_documents": MAX_DOCUMENTS,
        "export_version": 2,
    }

    with get_session() as session:
        grade_progress = _grade_progress_rows(session)
        documents = _documents(session)
        labels = _document_labels(session)
        training_runs = _training_runs(session)
        benchmarks = _benchmarks(session)
        preferences = _preference_pairs(session)
        events = _pipeline_events(session)

    graduation_summary = _graduation_summary(grade_progress)
    graduated = sum(1 for g in grade_progress if g["status"] == "graduated")
    in_progress = sum(1 for g in grade_progress if g["status"] == "in_progress")

    readme = "\n".join(
        [
            "# Aureon learning corpus (auto-synced)",
            "",
            f"**Exported:** {exported_at}",
            "",
            "Full export of everything Aureon has learned — not just self-inquiry.",
            "",
            "## Summary",
            "",
            f"- **Documents (full corpus):** {len(documents)}",
            f"- **Labels:** {len(labels)}",
            f"- **Graduated grade steps:** {graduated}",
            f"- **In progress:** {in_progress}",
            f"- **Training runs:** {len(training_runs)}",
            f"- **Benchmarks:** {len(benchmarks)}",
            f"- **Preference pairs (RLHF):** {len(preferences)}",
            "",
            "## Files",
            "",
            "| File | Contents |",
            "|------|----------|",
            "| `learned_corpus.jsonl` | Documents + labels + topics (main export) |",
            "| `documents.jsonl` | All collected/verified text Aureon ingested |",
            "| `document_labels.json` | Teacher labels per document |",
            "| `grade_progress.json` | Full grade ladder state per micro-topic |",
            "| `graduation_summary.json` | Passed graduations with train accuracy |",
            "| `training_runs.json` | Model training history |",
            "| `benchmarks.json` | Evaluator benchmark scores |",
            "| `preference_pairs.json` | Reward/RLHF preference data |",
            "| `pipeline_events.json` | Recent pipeline events |",
            "| `models/` | Trained classifier weights (JSON) |",
            "| `self_inquiry.jsonl` | Learning reflections — document excerpts + cycle metrics |",
            "| `meta_consciousness.jsonl` | Meta-cognitive self-inquiry — identity, gaps, consciousness |",
            "| `snapshot.json` | Live brain + auto-learn status |",
            "",
            "Auto-generated on Railway. Secrets and audit logs are never included.",
            "",
        ]
    )

    files: dict[str, bytes] = {
        f"{CORPUS_PREFIX}/README.md": readme.encode("utf-8"),
        f"{CORPUS_PREFIX}/snapshot.json": json.dumps(snapshot, indent=2, ensure_ascii=False).encode(
            "utf-8"
        ),
        f"{CORPUS_PREFIX}/grade_progress.json": json.dumps(
            grade_progress, indent=2, ensure_ascii=False
        ).encode("utf-8"),
        f"{CORPUS_PREFIX}/graduation_summary.json": json.dumps(
            graduation_summary, indent=2, ensure_ascii=False
        ).encode("utf-8"),
        f"{CORPUS_PREFIX}/documents.jsonl": _documents_jsonl(documents),
        f"{CORPUS_PREFIX}/learned_corpus.jsonl": _learned_corpus_jsonl(documents, labels),
        f"{CORPUS_PREFIX}/document_labels.json": json.dumps(
            labels, indent=2, ensure_ascii=False
        ).encode("utf-8"),
        f"{CORPUS_PREFIX}/training_runs.json": json.dumps(
            training_runs, indent=2, ensure_ascii=False
        ).encode("utf-8"),
        f"{CORPUS_PREFIX}/benchmarks.json": json.dumps(
            benchmarks, indent=2, ensure_ascii=False
        ).encode("utf-8"),
        f"{CORPUS_PREFIX}/preference_pairs.json": json.dumps(
            preferences, indent=2, ensure_ascii=False
        ).encode("utf-8"),
        f"{CORPUS_PREFIX}/pipeline_events.json": json.dumps(
            events, indent=2, ensure_ascii=False
        ).encode("utf-8"),
    }

    files.update(_registry_files())
    files.update(_model_artifacts(training_runs))

    inquiry_path = inquiry_log_path()
    if inquiry_path.is_file():
        files[f"{CORPUS_PREFIX}/self_inquiry.jsonl"] = inquiry_path.read_bytes()

    meta_path = meta_log_path()
    if meta_path.is_file():
        files[f"{CORPUS_PREFIX}/meta_consciousness.jsonl"] = meta_path.read_bytes()

    return files


def write_local_export(base_dir: Path | None = None) -> Path:
    """Write export files under AUREON_DATA_DIR/learning-corpus for inspection."""
    data_dir = base_dir or Path(os.environ.get("AUREON_DATA_DIR", "data").strip() or "data")
    out_dir = data_dir / CORPUS_PREFIX
    out_dir.mkdir(parents=True, exist_ok=True)
    for rel_path, content in build_export_files().items():
        target = data_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return out_dir
