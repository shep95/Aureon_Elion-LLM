"""Step 1 — automated data collection from trusted sources."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode
from xml.etree import ElementTree

import requests

from pipeline.config import RAW_DIR, SEEDS_DIR, ensure_dirs

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass
class RawDocument:
    doc_id: str
    source: str
    title: str
    text: str
    url: str = ""
    language: str = "en"
    metadata: dict[str, Any] = field(default_factory=dict)
    collected_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def content_hash(self) -> str:
        payload = f"{self.title}\n{self.text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class Collector(Protocol):
    name: str

    def collect(self, limit: int) -> list[RawDocument]: ...


class SeedCollector:
    """Load bundled primary-source seed texts (offline-safe)."""

    name = "seeds"

    def collect(self, limit: int = 50) -> list[RawDocument]:
        docs: list[RawDocument] = []
        if not SEEDS_DIR.exists():
            return docs
        for path in sorted(SEEDS_DIR.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for item in payload.get("documents", [])[:limit]:
                docs.append(
                    RawDocument(
                        doc_id=item.get("doc_id", str(uuid.uuid4())),
                        source=self.name,
                        title=item["title"],
                        text=item["text"],
                        url=item.get("url", ""),
                        language=item.get("language", "en"),
                        metadata=item.get("metadata", {}),
                    )
                )
            if len(docs) >= limit:
                break
        return docs[:limit]


class ArxivCollector:
    """Pull open research abstracts from arXiv API."""

    name = "arxiv"
    API = "https://export.arxiv.org/api/query"

    def __init__(self, query: str = "cat:cs.AI OR cat:cs.LG") -> None:
        self.query = query

    def collect(self, limit: int = 20) -> list[RawDocument]:
        params = {
            "search_query": self.query,
            "start": 0,
            "max_results": limit,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{self.API}?{urlencode(params)}"
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
        except requests.RequestException:
            return []

        root = ElementTree.fromstring(response.text)
        docs: list[RawDocument] = []
        for entry in root.findall("atom:entry", ATOM_NS):
            title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
            summary = (
                entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or ""
            ).strip()
            link = entry.find("atom:id", ATOM_NS)
            doc_url = link.text.strip() if link is not None and link.text else ""
            doc_id = doc_url.rsplit("/", 1)[-1] if doc_url else str(uuid.uuid4())
            docs.append(
                RawDocument(
                    doc_id=f"arxiv_{doc_id}",
                    source=self.name,
                    title=re.sub(r"\s+", " ", title),
                    text=re.sub(r"\s+", " ", summary),
                    url=doc_url,
                    metadata={"domain": "research", "peer_reviewed": False},
                )
            )
        return docs


class GutenbergCollector:
    """Pull pre-copyright book excerpts from Project Gutenberg."""

    name = "gutenberg"

    SOURCES = [
        (
            "1342",
            "Pride and Prejudice",
            "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
        ),
        (
            "2554",
            "Crime and Punishment",
            "https://www.gutenberg.org/cache/epub/2554/pg2554.txt",
        ),
    ]

    def collect(self, limit: int = 2) -> list[RawDocument]:
        docs: list[RawDocument] = []
        for book_id, title, url in self.SOURCES[:limit]:
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                text = response.text
            except requests.RequestException:
                continue
            start = text.find("*** START")
            end = text.find("*** END")
            if start != -1 and end != -1:
                text = text[start:end]
            excerpt = re.sub(r"\s+", " ", text[:8000]).strip()
            docs.append(
                RawDocument(
                    doc_id=f"gutenberg_{book_id}",
                    source=self.name,
                    title=title,
                    text=excerpt,
                    url=url,
                    metadata={"domain": "literature", "primary_source": True},
                )
            )
        return docs


class LocalFileCollector:
    """Ingest plain-text / markdown files dropped into data/raw/inbox."""

    name = "local_inbox"

    def __init__(self, inbox: Path | None = None) -> None:
        ensure_dirs()
        self.inbox = inbox or (RAW_DIR / "inbox")
        self.inbox.mkdir(parents=True, exist_ok=True)

    def collect(self, limit: int = 100) -> list[RawDocument]:
        docs: list[RawDocument] = []
        patterns = ("*.txt", "*.md", "*.json")
        files: list[Path] = []
        for pattern in patterns:
            files.extend(sorted(self.inbox.glob(pattern)))
        for path in files[:limit]:
            if path.suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                text = payload.get("text", "")
                title = payload.get("title", path.stem)
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")
                title = path.stem
            docs.append(
                RawDocument(
                    doc_id=f"local_{path.stem}",
                    source=self.name,
                    title=title,
                    text=text,
                    url=str(path),
                    metadata={"filename": path.name},
                )
            )
        return docs


def save_raw_batch(documents: list[RawDocument]) -> Path:
    ensure_dirs()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RAW_DIR / f"batch_{stamp}.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for doc in documents:
            handle.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
    return out_path
