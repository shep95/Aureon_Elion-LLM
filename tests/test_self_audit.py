"""Self-audit — full codebase introspection tests."""

from __future__ import annotations

from app.chat_service import chat
from brain.self_audit import (
    format_self_audit_report,
    is_self_audit_request,
    run_self_audit,
)


def test_is_self_audit_request_detects_phrases():
    assert is_self_audit_request("give the algorithm a copy of itself and see improvements")
    assert is_self_audit_request("/self-audit")
    assert is_self_audit_request("/evolve audit security workflow")
    assert not is_self_audit_request("what is DNA?")


def test_run_self_audit_returns_structure():
    audit = run_self_audit(max_files=30)
    assert audit["mode"] == "read_only_self_audit"
    assert "inventory" in audit
    assert "security_findings" in audit
    assert "workflow_findings" in audit
    assert "logical_improvements" in audit
    assert "new_software" in audit
    assert "fix_plan" in audit
    assert audit["inventory"]["file_count"] >= 1


def test_format_report_non_empty():
    audit = run_self_audit(max_files=25)
    report = format_self_audit_report(audit)
    assert "Self-audit complete" in report
    assert "Logical improvements" in report
    assert "Security" in report
    assert len(report) > 500


def test_chat_self_audit_command():
    result = chat("/self-audit")
    assert result["kind"] == "self_audit"
    assert "Self-audit complete" in result["reply"]
    assert "audit" in result


def test_chat_natural_self_audit():
    msg = (
        "final test, give the algorithm a copy of itself and see what improvements "
        "it would do, security flaws and workflow flaws"
    )
    result = chat(msg)
    assert result["kind"] == "self_audit"
    assert "Security findings" in result["reply"]
