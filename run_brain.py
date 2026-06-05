#!/usr/bin/env python3
"""CLI for the brain micro-algorithm architecture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brain.cortex import bootstrap_brain, brain_status, run_domain_cycle, run_full_brain, run_subdomain_cycle


def main() -> None:
    parser = argparse.ArgumentParser(description="Aureon brain — micro-algorithms per knowledge domain")
    parser.add_argument("--bootstrap", action="store_true", help="Seed DB with domains and micro-agents")
    parser.add_argument("--status", action="store_true", help="Show brain status")
    parser.add_argument("--domain", type=str, help="Run one domain (all subdomains)")
    parser.add_argument("--subdomain", type=str, help="Run one subdomain (requires --domain)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--domain-limit", type=int, default=3)
    parser.add_argument("--subdomain-limit", type=int, default=1)
    args = parser.parse_args()

    if args.bootstrap:
        print(json.dumps(bootstrap_brain(), indent=2))
        return
    if args.status:
        print(json.dumps(brain_status(), indent=2))
        return
    if args.domain and args.subdomain:
        print(json.dumps(run_subdomain_cycle(args.domain, args.subdomain, epochs=args.epochs), indent=2))
        return
    if args.domain:
        print(json.dumps(run_domain_cycle(args.domain, epochs=args.epochs), indent=2))
        return

    print(
        json.dumps(
            run_full_brain(
                epochs=args.epochs,
                domain_limit=args.domain_limit,
                subdomain_limit=args.subdomain_limit,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
