"""Quality filtering for collected text."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from pipeline.config import (
    CLEAN_DIR,
    MAX_TEXT_LENGTH,
    MIN_QUALITY_SCORE,
    MIN_TEXT_LENGTH,
    RAW_DIR,
    ensure_dirs,
)
from pipeline.step1_collection.collectors import RawDocument

BOILERPLATE_PATTERNS = (
    r"click here",
    r"subscribe now",
    r"lorem ipsum",
    r"cookie policy",
)
WORD_RE = re.compile(r"[a-zA-Z]{2,}")


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _unique_word_ratio(text: str) -> float:
    words = [w.lower() for w in WORD_RE.findall(text)]
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(ch.isalpha() for ch in text)
    return alpha / len(text)


def score_quality(doc: RawDocument) -> float:
    """Heuristic quality score in [0, 1]."""
    text = doc.text.strip()
    length = len(text)
    if length < MIN_TEXT_LENGTH:
        return 0.0

    length_score = min(1.0, length / 1200)
    diversity = _unique_word_ratio(text)
    alpha = _alpha_ratio(text)
    boilerplate_hits = sum(
        1 for pattern in BOILERPLATE_PATTERNS if re.search(pattern, text, re.I)
    )
    boilerplate_penalty = min(0.4, boilerplate_hits * 0.15)

    score = (0.35 * length_score) + (0.35 * diversity) + (0.30 * alpha)
    return max(0.0, min(1.0, score - boilerplate_penalty))


def filter_document(doc: RawDocument) -> tuple[bool, float, str]:
    text = doc.text.strip()
    if len(text) < MIN_TEXT_LENGTH:
        return False, 0.0, "too_short"
    if len(text) > MAX_TEXT_LENGTH:
        return False, 0.0, "too_long"

    quality = score_quality(doc)
    if quality < MIN_QUALITY_SCORE:
        return False, quality, "low_quality"

    lowered = text.lower()
    for pattern in BOILERPLATE_PATTERNS:
        if re.search(pattern, lowered):
            return False, quality, "boilerplate"

    return True, quality, "accepted"


def dedupe_documents(documents: list[RawDocument]) -> list[RawDocument]:
    seen: set[str] = set()
    unique: list[RawDocument] = []
    for doc in documents:
        digest = doc.content_hash()
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(doc)
    return unique


def load_raw_batches() -> list[RawDocument]:
    ensure_dirs()
    docs: list[RawDocument] = []
    for path in sorted(RAW_DIR.glob("batch_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                docs.append(RawDocument(**payload))
    return docs


def run_filter_pipeline(documents: list[RawDocument]) -> tuple[list[dict], dict]:
    deduped = dedupe_documents(documents)
    accepted: list[dict] = []
    rejected: dict[str, int] = {}

    for doc in deduped:
        ok, quality, reason = filter_document(doc)
        if ok:
            row = asdict(doc)
            row["quality_score"] = round(quality, 4)
            accepted.append(row)
        else:
            rejected[reason] = rejected.get(reason, 0) + 1

    out_path = CLEAN_DIR / "corpus.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return accepted, {
        "input_documents": len(documents),
        "deduped_documents": len(deduped),
        "accepted_documents": len(accepted),
        "rejected_breakdown": rejected,
        "clean_corpus_path": str(out_path),
    }
