"""Unified file router — PDF, image, audio, text → corpus + chat context."""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from brain.multimodal_processors import (
    AUDIO_EXT,
    IMAGE_EXT,
    PDF_EXT,
    TEXT_EXT,
    extract_pdf,
    extract_text_file,
    process_audio,
    process_image,
    text_embedding,
    tier_status,
)
from pipeline.config import MULTIMODAL_DIR, ensure_dirs

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = int(__import__("os").environ.get("AUREON_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))


@dataclass
class FileIngestResult:
    filename: str
    modality: str
    text: str
    title: str
    metadata: dict[str, Any] = field(default_factory=dict)
    saved_path: str | None = None
    document_id: int | None = None
    content_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "modality": self.modality,
            "text_preview": self.text[:500],
            "title": self.title,
            "metadata": self.metadata,
            "saved_path": self.saved_path,
            "document_id": self.document_id,
            "content_hash": self.content_hash,
        }


def _ext(name: str) -> str:
    return Path(name).suffix.lower()


def route_bytes(filename: str, data: bytes, *, message: str = "") -> FileIngestResult:
    """Route upload by extension — Tier 3 PDF/image/audio/text processors."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File too large (max {MAX_UPLOAD_BYTES} bytes)")

    ext = _ext(filename)
    meta: dict[str, Any] = {"tier": tier_status(), "user_message": message[:500] if message else ""}

    if ext in PDF_EXT:
        text = extract_pdf(data)
        if not text:
            text = f"PDF {filename} uploaded — text extraction unavailable (install pypdf)."
        modality = "pdf"
        title = Path(filename).stem.replace("_", " ").title()
    elif ext in IMAGE_EXT:
        text, meta = process_image(data, filename)
        modality = "image"
        title = Path(filename).stem.replace("_", " ").title()
    elif ext in AUDIO_EXT:
        text, meta = process_audio(data, filename)
        modality = "audio"
        title = Path(filename).stem.replace("_", " ").title()
    elif ext in TEXT_EXT or ext == "":
        text = extract_text_file(data, filename)
        modality = "text"
        title = Path(filename).stem.replace("_", " ").title()
    else:
        text = extract_text_file(data, filename)
        modality = "binary"
        title = Path(filename).stem.replace("_", " ").title()
        if len(text) < 20:
            text = f"Uploaded file {filename} ({len(data)} bytes) — unsupported type for deep parsing."

    if message.strip():
        text = f"{text}\n\nUser question about this file: {message.strip()}"

    digest = hashlib.sha256(text.encode()).hexdigest()
    meta["content_hash"] = digest
    meta["embedding"] = text_embedding(text)

    saved = _save_to_inbox(filename, data, ext)
    return FileIngestResult(
        filename=filename,
        modality=modality,
        text=text,
        title=title or filename,
        metadata=meta,
        saved_path=str(saved) if saved else None,
        content_hash=digest,
    )


def _save_to_inbox(filename: str, data: bytes, ext: str) -> Path | None:
    """Persist raw bytes under multimodal inbox for collector replay."""
    ensure_dirs()
    inbox = MULTIMODAL_DIR
    inbox.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name.replace("..", "").replace("/", "_")[:120]
    if not safe_name:
        safe_name = f"upload_{uuid.uuid4().hex[:8]}{ext or '.bin'}"
    target = inbox / safe_name
    if target.exists():
        target = inbox / f"{target.stem}_{uuid.uuid4().hex[:6]}{target.suffix}"
    try:
        target.write_bytes(data)
        if ext in IMAGE_EXT | AUDIO_EXT:
            sidecar = target.with_suffix(target.suffix + ".txt")
            if not sidecar.exists():
                sidecar.write_text(
                    f"Auto sidecar for {safe_name} — re-ingest after Whisper/CLIP processing.",
                    encoding="utf-8",
                )
        return target
    except OSError as exc:
        logger.warning("Could not save upload to inbox: %s", exc)
        return None


def persist_to_corpus(result: FileIngestResult) -> int | None:
    """Write ingested text to documents table + invalidate RAG."""
    from sqlalchemy import select

    from db.models import Document, KnowledgeDomain, KnowledgeSubdomain
    from db.seed import get_micro_subdomain
    from db.session import get_session, init_db

    if len(result.text.strip()) < 20:
        return None

    init_db()
    domain_slug = "technology_and_engineering"
    subdomain_slug = "computer_science"
    micro_slug = "python_functions" if result.modality in ("text", "pdf") else "data_structures"

    with get_session() as session:
        domain = session.scalar(select(KnowledgeDomain).where(KnowledgeDomain.slug == domain_slug))
        if not domain:
            return None
        subdomain = session.scalar(
            select(KnowledgeSubdomain).where(
                KnowledgeSubdomain.domain_id == domain.id,
                KnowledgeSubdomain.slug == subdomain_slug,
            )
        )
        if not subdomain:
            return None
        micro = get_micro_subdomain(session, domain_slug, subdomain_slug, micro_slug)
        if not micro:
            return None

        digest = result.content_hash or hashlib.sha256(result.text.encode()).hexdigest()
        if session.scalar(select(Document).where(Document.content_hash == digest)):
            existing = session.scalar(select(Document).where(Document.content_hash == digest))
            return existing.id if existing else None

        extra = dict(result.metadata)
        extra["modality"] = result.modality
        extra["upload_filename"] = result.filename
        if result.saved_path:
            extra["media_path"] = result.saved_path

        doc = Document(
            domain_id=domain.id,
            subdomain_id=subdomain.id,
            micro_subdomain_id=micro.id,
            source="file_upload",
            title=result.title,
            text=result.text[:100_000],
            url=result.saved_path,
            language="en",
            quality_score=0.75,
            verified=False,
            content_hash=digest,
            extra=extra,
        )
        session.add(doc)
        session.commit()
        session.refresh(doc)
        result.document_id = doc.id

        try:
            from brain.pgvector_store import index_document_embedding

            index_document_embedding(doc.id, extra.get("embedding", []))
        except Exception:
            pass

        try:
            from brain.vector_rag import invalidate_rag_index

            invalidate_rag_index()
        except Exception:
            pass

        return doc.id


def ingest_upload(filename: str, data: bytes, *, message: str = "", persist: bool = True) -> FileIngestResult:
    result = route_bytes(filename, data, message=message)
    if persist:
        persist_to_corpus(result)
    return result
