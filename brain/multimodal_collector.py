"""Multimodal collectors — vision/audio sidecars into the supervised corpus."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from app.security import load_json_file_bounded, resolve_path_under
from pipeline.config import MULTIMODAL_DIR, ensure_dirs
from pipeline.step1_collection.collectors import RawDocument

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg"}
SIDEcar_EXT = {".txt", ".md", ".json"}


def _read_sidecar(path: Path) -> tuple[str, str]:
    """Return (title, text) from a sidecar file."""
    if path.suffix == ".json":
        payload = load_json_file_bounded(path)
        return (
            str(payload.get("title", path.stem)),
            str(payload.get("text") or payload.get("transcript") or payload.get("caption", "")),
        )
    text = path.read_text(encoding="utf-8", errors="ignore")[:50_000]
    return path.stem.replace("_", " ").title(), text


class MultimodalCollector:
    """Ingest multimodal drops: image/audio + caption/transcript sidecars."""

    name = "multimodal"

    def __init__(self, inbox: Path | None = None) -> None:
        ensure_dirs()
        self.inbox = inbox or MULTIMODAL_DIR
        self.inbox.mkdir(parents=True, exist_ok=True)

    def collect(self, limit: int = 20) -> list[RawDocument]:
        docs: list[RawDocument] = []
        media_files = sorted(
            p
            for p in self.inbox.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXT | AUDIO_EXT
        )

        for media in media_files[:limit]:
            modality = "image" if media.suffix.lower() in IMAGE_EXT else "audio"
            title = media.stem.replace("_", " ").title()
            text_parts: list[str] = []

            for ext in SIDEcar_EXT:
                sidecar = media.with_suffix(media.suffix + ext.lstrip("."))
                if not sidecar.is_file():
                    sidecar = media.with_suffix(ext)
                if sidecar.is_file():
                    try:
                        safe = resolve_path_under(self.inbox, sidecar.name)
                    except ValueError:
                        continue
                    t, body = _read_sidecar(safe)
                    title = t or title
                    if body.strip():
                        text_parts.append(body.strip())

            if not text_parts:
                text_parts.append(
                    f"{modality.title()} asset {media.name} pending caption or transcript sidecar."
                )

            text = " ".join(text_parts)
            digest = hashlib.sha256(text.encode()).hexdigest()
            docs.append(
                RawDocument(
                    doc_id=f"multimodal_{media.stem}_{uuid.uuid4().hex[:8]}",
                    source=self.name,
                    title=title,
                    text=text,
                    url=str(media),
                    metadata={
                        "modality": modality,
                        "filename": media.name,
                        "source_type": "multimodal_sidecar",
                        "media_path": str(media),
                        "content_hash_hint": digest,
                    },
                )
            )

        # Standalone JSON multimodal manifests
        for path in sorted(self.inbox.glob("*.json"))[: max(0, limit - len(docs))]:
            try:
                safe = resolve_path_under(self.inbox, path.name)
            except ValueError:
                continue
            payload = load_json_file_bounded(safe)
            if payload.get("modality") not in ("image", "audio", "video"):
                continue
            title = str(payload.get("title", path.stem))
            text = str(payload.get("text") or payload.get("transcript") or payload.get("caption", ""))
            if len(text) < 20:
                continue
            docs.append(
                RawDocument(
                    doc_id=f"multimodal_json_{path.stem}",
                    source=self.name,
                    title=title,
                    text=text,
                    url=str(path),
                    metadata={
                        "modality": payload.get("modality"),
                        "source_type": "multimodal_manifest",
                        **{k: v for k, v in payload.get("metadata", {}).items() if isinstance(v, (str, int, float, bool))},
                    },
                )
            )

        return docs


def multimodal_status() -> dict[str, Any]:
    ensure_dirs()
    inbox = MULTIMODAL_DIR
    counts = {"image": 0, "audio": 0, "manifest": 0}
    if inbox.is_dir():
        for p in inbox.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() in IMAGE_EXT:
                counts["image"] += 1
            elif p.suffix.lower() in AUDIO_EXT:
                counts["audio"] += 1
            elif p.suffix.lower() == ".json":
                counts["manifest"] += 1
    return {"inbox": str(inbox), "counts": counts}
