#!/usr/bin/env python3
"""Ingest Aureon Files (PDF/TXT) into the document database.

Usage:
  python scripts/ingest_aureon_files.py "C:/Users/kille/Downloads/Aureon Files"
  python scripts/ingest_aureon_files.py ./docs --domain philosophy --subdomain metaphysics
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            print(f"Skip PDF (install pypdf): {path.name}")
            return ""
        except Exception as exc:
            print(f"PDF read failed {path.name}: {exc}")
            return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def ingest_folder(
    folder: Path,
    *,
    domain: str = "philosophy_and_metaphysics",
    subdomain: str = "consciousness_studies",
    micro: str = "philosophical_doctrine",
) -> int:
    from db.models import Document, KnowledgeDomain, KnowledgeMicroSubdomain, KnowledgeSubdomain
    from db.seed import get_micro_subdomain
    from db.session import get_session, init_db

    init_db()
    count = 0
    patterns = ("*.pdf", "*.txt", "*.md")

    with get_session() as session:
        micro_row = get_micro_subdomain(session, domain, subdomain, micro)
        if not micro_row:
            dom = session.query(KnowledgeDomain).filter_by(slug=domain).first()
            if not dom:
                print(f"Unknown domain slug: {domain}")
                return 0
            sub = (
                session.query(KnowledgeSubdomain)
                .filter_by(domain_id=dom.id, slug=subdomain)
                .first()
            )
            if not sub:
                print(f"Unknown subdomain: {subdomain}")
                return 0
            print(f"Unknown micro: {micro}")
            return 0

        for pattern in patterns:
            for path in folder.rglob(pattern):
                text = _read_file(path).strip()
                if len(text) < 80:
                    continue
                digest = hashlib.sha256(text.encode()).hexdigest()
                existing = session.query(Document).filter_by(content_hash=digest).first()
                if existing:
                    continue
                session.add(
                    Document(
                        domain_id=micro_row.domain_id,
                        subdomain_id=micro_row.subdomain_id,
                        micro_subdomain_id=micro_row.id,
                        source="aureon_files",
                        title=path.stem.replace("_", " ")[:200],
                        text=text[:50000],
                        verified=True,
                        quality_score=0.85,
                        content_hash=digest,
                        extra={"path": str(path)},
                    )
                )
                count += 1
        session.commit()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Aureon Files into corpus DB")
    parser.add_argument("folder", type=Path, help="Folder with PDF/TXT/MD files")
    parser.add_argument("--domain", default="philosophy_and_metaphysics")
    parser.add_argument("--subdomain", default="consciousness_studies")
    parser.add_argument("--micro", default="philosophical_doctrine")
    args = parser.parse_args()
    if not args.folder.is_dir():
        raise SystemExit(f"Not a directory: {args.folder}")
    n = ingest_folder(args.folder, domain=args.domain, subdomain=args.subdomain, micro=args.micro)
    print(f"Ingested {n} new documents from {args.folder}")


if __name__ == "__main__":
    main()
