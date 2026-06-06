#!/usr/bin/env python3
"""Scored curiosity pipeline test — prototypes, security gates, approval flow."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _mock_search(query: str, *, max_results: int = 5):
    return [
        {
            "type": "web",
            "text": f"Advanced AI security math design algorithms for {query[:50]}.",
            "source": "mock.example",
        }
    ]


def main() -> int:
    from app.curiosity_proposals import approve_proposal, get_proposal
    from app.curiosity_sandbox import (
        deploy_proposal,
        push_github_brief,
        run_curiosity_cycle,
        verify_sandbox_proposal,
    )
    from app.self_evolve import push_fork
    from brain.curiosity_engine import is_curiosity_enabled

    checks: list[tuple[str, bool, str]] = []
    score = 100

    def record(name: str, ok: bool, detail: str = "") -> None:
        nonlocal score
        checks.append((name, ok, detail))
        if not ok:
            score -= 10

    if not is_curiosity_enabled():
        print("FAIL: AUREON_CURIOSITY_ENABLED is off")
        return 1

    cycle = run_curiosity_cycle(search_fn=_mock_search)
    record("curiosity_cycle", cycle.get("ok") is True, str(cycle.get("error", "")))

    pid = (cycle.get("proposal") or {}).get("id", "")
    record("proposal_created", bool(pid))
    record(
        "sandbox_built",
        bool((cycle.get("sandbox") or {}).get("sandbox_built")),
    )
    record(
        "pending_approval",
        (cycle.get("proposal") or {}).get("status") == "pending_approval",
    )

    if pid:
        verify = verify_sandbox_proposal(pid)
        record("prototype_verify", verify.get("ok") is True, f"modules={verify.get('module_count')}")
        for mod in verify.get("modules") or []:
            name = mod.get("path", "module")
            record(f"runtime:{Path(name).name}", (mod.get("runtime") or {}).get("ok") is True)

        proposal = get_proposal(pid)
        gh_block = push_github_brief(proposal, approved=False)
        record("github_gate", gh_block.get("pushed") is False)

        fork_block = push_fork(branch="aureon/test", approved=False)
        record("fork_push_gate", fork_block.get("pushed") is False)

        approve_proposal(pid, approved=False, reviewer="runner")
        rejected_deploy = deploy_proposal(pid, approve_github=True)
        record("reject_blocks_deploy", rejected_deploy.get("error") == "rejected")

    score = max(score, 0)
    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    pct = round(100 * passed / total) if total else 0

    out = ROOT / "data" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "score_pct": pct,
        "passed": passed,
        "total": total,
        "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
    }
    path = out / "curiosity-score.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Curiosity test: {passed}/{total} checks ({pct}%)")
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        line = f"  [{mark}] {name}"
        if detail:
            line += f" — {detail}"
        print(line)
    print(f"Report: {path}")
    return 0 if pct >= 90 else 1


if __name__ == "__main__":
    raise SystemExit(main())
