#!/usr/bin/env python3
"""Run self-audit and print scored summary (CI-friendly)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brain.self_audit import format_self_audit_report, run_self_audit  # noqa: E402


def main() -> int:
    audit = run_self_audit()
    report = format_self_audit_report(audit)
    print(report.encode("ascii", errors="replace").decode("ascii"))
    print()

    high = sum(1 for f in audit["security_findings"] if f.get("severity") == "high")
    medium_wf = sum(1 for w in audit["workflow_findings"] if w.get("severity") == "medium")
    score = 100
    real_high = sum(
        1 for f in audit["security_findings"]
        if f.get("severity") == "high" and "AST" in f.get("detail", "")
    )
    config_medium = sum(
        1 for f in audit["security_findings"]
        if f.get("severity") == "medium" and "skip" in f.get("detail", "").lower()
    )
    score -= min(real_high * 10, 40)
    score -= min(medium_wf * 3, 15)
    score -= min(config_medium * 2, 10)
    score = max(score, 0)

    out = ROOT / "data" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "self-audit-score.json"
    payload = {
        "score_pct": score,
        "high_security": high,
        "medium_workflow": medium_wf,
        "files_scanned": audit["inventory"]["file_count"],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Self-audit score: {score}% (high sec={high}, medium workflow={medium_wf})")
    print(f"Report saved: {path}")
    return 0 if score >= 70 else 1


if __name__ == "__main__":
    raise SystemExit(main())
