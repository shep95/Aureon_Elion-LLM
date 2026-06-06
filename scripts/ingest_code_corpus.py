#!/usr/bin/env python3
"""CLI wrapper — ingest HumanEval + MBPP into the document database."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.code_corpus_ingest import ingest_code_corpus

if __name__ == "__main__":
    n = ingest_code_corpus()
    print(f"Ingested {n} new code documents.")
