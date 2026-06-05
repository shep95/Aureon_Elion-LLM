#!/usr/bin/env python3
"""Initialize PostgreSQL on Railway and seed the knowledge taxonomy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.cortex import bootstrap_brain, brain_status
from brain.domains.taxonomy import total_subdomains
from db.session import get_database_url, init_db


def main() -> None:
    print(f"Database URL: {get_database_url().split('@')[-1] if '@' in get_database_url() else get_database_url()}")
    init_db()
    stats = bootstrap_brain()
    status = brain_status()
    print(json.dumps({"seed": stats, "status": status, "total_subdomains_in_taxonomy": total_subdomains()}, indent=2))


if __name__ == "__main__":
    main()
