#!/usr/bin/env python3
"""Smoke-test chat: 2 questions per knowledge domain + live-event search."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Fast, bounded local run — override before app imports
os.environ.setdefault("DATABASE_URL", "sqlite:///data/aureon.db")
os.environ.setdefault("AUREON_PREDICT_FAST_EPOCHS", "30")
os.environ.setdefault("AUREON_PREDICT_EPOCHS", "30")
os.environ.setdefault("AUREON_PREDICT_TIMEOUT_SEC", "45")
os.environ.setdefault("AUREON_WEB_SEARCH_ENABLED", "1")
os.environ.setdefault("AUREON_PREDICT_D_MODEL", "64")
os.environ.setdefault("AUREON_PREDICT_LAYERS", "4")
os.environ.setdefault("AUREON_PREDICT_MAX_VOCAB", "8000")
os.environ.setdefault("AUREON_PREDICT_MAX_SEQ", "256")

from brain.domains.taxonomy import all_domain_slugs, taxonomy_catalog
from brain.system_messages import FALLBACK_CORPUS, FALLBACK_TRAINING
from brain.predict_engine import warm_up_predict_brain

LIVE_EVENT_QUESTIONS = [
    "What happened in AI news today?",
    "Who won the latest major US election?",
    "What is the current price of Bitcoin?",
    "What happened in the world in 2026?",
]

FAILURE_MARKERS = (
    "deeper corpus grounding than i can compute",
    "need more corpus in that domain",
    "no production classifier is promoted",
    FALLBACK_CORPUS.lower()[:40],
    FALLBACK_TRAINING.lower()[:40],
)


def _domain_questions() -> list[tuple[str, str, str]]:
    """Return (domain_slug, topic_label, question_text) — two per domain."""
    catalog = taxonomy_catalog()
    rows: list[tuple[str, str, str]] = []
    for domain in all_domain_slugs():
        entry = catalog.get(domain, {})
        topics: list[str] = []
        for sub in entry.get("subdomains", {}).values():
            for micro in sub.get("micro_subdomains", {}).values():
                for topic in micro.get("topics", []):
                    label = str(topic).strip()
                    if label and label not in topics:
                        topics.append(label)
                    if len(topics) >= 2:
                        break
                if len(topics) >= 2:
                    break
            if len(topics) >= 2:
                break
        while len(topics) < 2:
            name = entry.get("name", domain.replace("_", " ").title())
            topics.append(f"{name} fundamentals")
            topics = topics[:2]
        for topic in topics[:2]:
            q = f"What is {topic}?"
            rows.append((domain, topic, q))
    return rows


def _is_failure(reply: str, kind: str) -> str | None:
    text = reply.lower().strip()
    if not text or len(text) < 12:
        return "reply too short"
    if kind in ("echo_detected", "self_echo_detected"):
        return None
    for marker in FAILURE_MARKERS:
        if marker in text:
            return f"fallback marker: {marker[:50]}"
    return None


def main() -> int:
    from app.chat_service import chat

    print("Warming predict brain + RAG...")
    warm = warm_up_predict_brain(run_probe=False)
    print(f"  warm-up: {json.dumps(warm, default=str)[:200]}")

    domain_rows = _domain_questions()
    results: list[dict] = []
    failures: list[dict] = []

    print(f"\n=== Domain smoke test ({len(domain_rows)} questions, 30 domains) ===\n")
    t0 = time.monotonic()
    for i, (domain, topic, question) in enumerate(domain_rows, 1):
        sid = f"domain-{domain}-{i}"
        out = chat(question, session_id=sid)
        reply = str(out.get("reply", ""))
        kind = str(out.get("kind", ""))
        err = _is_failure(reply, kind)
        row = {
            "domain": domain,
            "topic": topic,
            "question": question,
            "kind": kind,
            "ok": err is None,
            "error": err,
            "reply_preview": reply[:120].replace("\n", " "),
        }
        results.append(row)
        if err:
            failures.append(row)
        status = "OK" if err is None else f"FAIL ({err})"
        print(f"[{i:02d}/{len(domain_rows)}] {status} | {domain[:28]:28} | {question[:50]}")
        if err:
            print(f"         -> {reply[:100]}")

    print(f"\n=== Live events ({len(LIVE_EVENT_QUESTIONS)} questions) ===\n")
    for j, question in enumerate(LIVE_EVENT_QUESTIONS, 1):
        sid = f"live-{j}"
        out = chat(question, session_id=sid)
        reply = str(out.get("reply", ""))
        kind = str(out.get("kind", ""))
        err = _is_failure(reply, kind)
        search_ok = kind in (
            "search_opinion",
            "deep_concept_search",
            "search_empty",
            "search_no_opinion",
        ) or "source" in reply.lower()
        if kind == "search_empty" and "no results" in reply.lower():
            err = err or "search returned empty"
        row = {
            "domain": "live_events",
            "topic": "web_search",
            "question": question,
            "kind": kind,
            "ok": err is None and kind == "search_opinion",
            "error": err,
            "search_routed": search_ok,
            "reply_preview": reply[:120].replace("\n", " "),
        }
        results.append(row)
        if not row["ok"]:
            failures.append(row)
        status = "OK" if row["ok"] else f"FAIL ({err or kind})"
        print(f"[{j}] {status} | {question}")
        print(f"    kind={kind} | {reply[:100]}")

    elapsed = time.monotonic() - t0
    domain_ok = sum(1 for r in results if r["domain"] != "live_events" and r["ok"])
    live_ok = sum(1 for r in results if r["domain"] == "live_events" and r["ok"])

    print("\n=== Summary ===")
    print(f"Domain questions: {domain_ok}/{len(domain_rows)} passed")
    print(f"Live events:      {live_ok}/{len(LIVE_EVENT_QUESTIONS)} passed")
    print(f"Elapsed:          {elapsed:.1f}s")
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures[:15]:
            print(f"  - [{f['domain']}] {f['question']}: {f.get('error') or f['kind']}")
        if len(failures) > 15:
            print(f"  ... and {len(failures) - 15} more")

    out_path = ROOT / "data" / "audit" / "domain-smoke-test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"results": results, "summary": {
        "domain_pass": domain_ok,
        "domain_total": len(domain_rows),
        "live_pass": live_ok,
        "live_total": len(LIVE_EVENT_QUESTIONS),
        "elapsed_sec": round(elapsed, 2),
    }}, indent=2), encoding="utf-8")
    print(f"\nFull report: {out_path}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
