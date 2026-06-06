"""Tier 3–4 multimodal processors — PDF, CLIP-style vision, Whisper-style audio."""

from __future__ import annotations

import hashlib
import io
import logging
import os
import struct
from typing import Any

logger = logging.getLogger(__name__)

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
PDF_EXT = {".pdf"}
TEXT_EXT = {".txt", ".md", ".json", ".csv"}


def tier_status() -> dict[str, Any]:
    return {
        "pdf": _pdf_available(),
        "vision": _vision_tier(),
        "audio": _whisper_tier(),
        "pgvector": os.environ.get("AUREON_PGVECTOR", "1").strip().lower() not in ("0", "false", "no"),
    }


def _pdf_available() -> bool:
    try:
        import pypdf  # noqa: F401

        return True
    except ImportError:
        return False


def _vision_tier() -> str:
    if os.environ.get("AUREON_CLIP", "1").strip().lower() in ("0", "false", "no"):
        return "disabled"
    try:
        from PIL import Image  # noqa: F401

        return "pil_metadata"
    except ImportError:
        return "hash_fingerprint"


def _whisper_tier() -> str:
    if os.environ.get("AUREON_WHISPER", "1").strip().lower() in ("0", "false", "no"):
        return "disabled"
    try:
        import whisper  # noqa: F401

        return "openai_whisper"
    except ImportError:
        return "sidecar_required"


def extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except ImportError:
        return ""
    except Exception as exc:
        logger.warning("PDF extract failed: %s", exc)
        return ""


def process_image(data: bytes, filename: str) -> tuple[str, dict[str, Any]]:
    """Tier 3 vision — PIL metadata + hash fingerprint; Tier 4 CLIP when installed."""
    meta: dict[str, Any] = {"modality": "image", "filename": filename, "bytes": len(data)}
    caption_parts: list[str] = [f"Image upload: {filename}"]

    try:
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        meta.update({"width": img.width, "height": img.height, "format": img.format or "unknown"})
        caption_parts.append(f"{img.width}x{img.height} {img.format or 'image'}")
        if hasattr(img, "getcolors") and img.width * img.height < 250_000:
            colors = img.convert("RGB").getcolors(maxcolors=256)
            if colors:
                dominant = max(colors, key=lambda c: c[0])
                meta["dominant_rgb"] = dominant[1]
    except ImportError:
        meta["vision_tier"] = "hash_fingerprint"
    except Exception as exc:
        meta["vision_error"] = str(exc)[:120]

    digest = hashlib.sha256(data).hexdigest()
    meta["content_hash"] = digest
    caption_parts.append(f"sha256={digest[:16]}")

    clip_caption = _try_clip_caption(data)
    if clip_caption:
        meta["vision_tier"] = "clip"
        caption_parts.append(clip_caption)

    return " ".join(caption_parts), meta


def _try_clip_caption(data: bytes) -> str | None:
    """Optional CLIP — only when torch + transformers available."""
    if os.environ.get("AUREON_CLIP", "1").strip().lower() in ("0", "false", "no"):
        return None
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        labels = [
            "a diagram or chart",
            "a photograph of a person",
            "a screenshot of software code",
            "a medical scan",
            "a natural landscape",
            "a document or text page",
        ]
        model_name = os.environ.get("AUREON_CLIP_MODEL", "openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        inputs = processor(text=labels, images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
        best = int(probs.argmax())
        return f"CLIP classification: {labels[best]} ({float(probs[best]):.0%})"
    except Exception:
        return None


def process_audio(data: bytes, filename: str) -> tuple[str, dict[str, Any]]:
    """Tier 3 sidecar; Tier 4 Whisper when openai-whisper installed."""
    meta: dict[str, Any] = {"modality": "audio", "filename": filename, "bytes": len(data)}
    transcript = _try_whisper_transcribe(data, filename)
    if transcript:
        meta["audio_tier"] = "whisper"
        return transcript.strip(), meta
    meta["audio_tier"] = "pending_transcript"
    return (
        f"Audio upload {filename} ({len(data)} bytes) — transcript pending. "
        f"Install openai-whisper or add a .txt sidecar with transcript.",
        meta,
    )


def _try_whisper_transcribe(data: bytes, filename: str) -> str | None:
    if os.environ.get("AUREON_WHISPER", "1").strip().lower() in ("0", "false", "no"):
        return None
    try:
        import tempfile
        import whisper

        model_size = os.environ.get("AUREON_WHISPER_MODEL", "base")
        suffix = os.path.splitext(filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        model = whisper.load_model(model_size)
        result = model.transcribe(tmp_path)
        os.unlink(tmp_path)
        return str(result.get("text", "")).strip() or None
    except Exception as exc:
        logger.debug("Whisper unavailable: %s", exc)
        return None


def extract_text_file(data: bytes, filename: str) -> str:
    try:
        return data.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def text_embedding(text: str, *, dims: int = 128) -> list[float]:
    """Lightweight embedding for pgvector-style similarity (no external model required)."""
    vec = [0.0] * dims
    tokens = text.lower().split()
    if not tokens:
        return vec
    for tok in tokens:
        h = hashlib.sha256(tok.encode()).digest()
        for i in range(min(8, dims)):
            idx = (h[i] + h[i + 8]) % dims
            vec[idx] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [round(v / norm, 6) for v in vec]
