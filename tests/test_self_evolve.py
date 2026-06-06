"""Self-evolve fork system tests."""

from __future__ import annotations

import pytest

from app.self_evolve import (
    analyze_file_for_task,
    commit_evolution,
    list_source_files,
    plan_evolution,
    repo_status,
    run_tests_before_commit,
    validate_repo_path,
    verify_before_commit,
)


def test_validate_repo_path_allows_brain():
    p = validate_repo_path("brain/code_master.py")
    assert p.name == "code_master.py"


def test_validate_repo_path_rejects_env():
    with pytest.raises(ValueError):
        validate_repo_path(".env")


def test_validate_repo_path_rejects_traversal():
    with pytest.raises(ValueError):
        validate_repo_path("../../../etc/passwd")


def test_plan_evolution_suggests_files():
    plan = plan_evolution("improve philosophy god routing")
    assert "brain/philosophy_handler.py" in plan["suggested_files"]


def test_repo_status():
    status = repo_status()
    assert "current_branch" in status
    assert status["policy"]


def test_list_source_files():
    files = list_source_files(limit=10)
    assert len(files) >= 1
    assert any("chat_service" in f for f in files)


def test_analyze_file_for_task_finds_functions():
    analysis = analyze_file_for_task("app/self_evolve.py", "commit evolution fork")
    assert analysis["path"] == "app/self_evolve.py"
    names = {f["name"] for f in analysis["functions"]}
    assert "commit_evolution" in names
    assert isinstance(analysis["task_relevant_functions"], list)
    assert len(analysis["task_relevant_functions"]) >= 1


def test_verify_before_commit_passes_valid_python():
    result = verify_before_commit(["app/self_evolve.py"])
    assert result["ok"] is True
    assert "app/self_evolve.py" in result["passed"]


def test_verify_before_commit_fails_invalid_python(tmp_path, monkeypatch):
    bad_path = "tests/_self_evolve_syntax_probe.py"
    probe = tmp_path / "_self_evolve_syntax_probe.py"
    probe.write_text("def broken(\n", encoding="utf-8")

    def _read(rel_path: str) -> dict:
        if rel_path == bad_path:
            return {"path": rel_path, "content": probe.read_text(encoding="utf-8")}
        raise FileNotFoundError(rel_path)

    monkeypatch.setattr("app.self_evolve.read_source", _read)
    result = verify_before_commit([bad_path])
    assert result["ok"] is False
    assert result["failed"][0]["path"] == bad_path


def test_plan_evolution_includes_analysis():
    plan = plan_evolution("improve philosophy god routing")
    assert "analysis" in plan
    assert len(plan["analysis"]) >= 1
    assert "capabilities" in plan


def test_analyze_file_includes_imports_and_recommendations():
    analysis = analyze_file_for_task("app/self_evolve.py", "verify commit syntax tests")
    assert analysis["path"] == "app/self_evolve.py"
    assert analysis.get("imports")
    assert "verify_before_commit" in analysis["task_relevant_functions"]
    assert analysis.get("recommendations")


def test_commit_evolution_blocks_on_syntax_failure(monkeypatch):
    bad_path = "tests/_commit_block_probe.py"

    def _fake_verify(paths: list[str]) -> dict:
        return {
            "ok": False,
            "passed": [],
            "failed": [{"path": bad_path, "error": "invalid syntax"}],
            "skipped": [],
        }

    monkeypatch.setattr("app.self_evolve.verify_before_commit", _fake_verify)
    monkeypatch.setattr(
        "app.self_evolve._run_git",
        lambda args, check=True: type("R", (), {"stdout": " M tests/_commit_block_probe.py", "returncode": 0})(),
    )
    monkeypatch.setattr("app.self_evolve.skip_syntax_verify", lambda: False)
    monkeypatch.setattr("app.self_evolve.skip_test_gate", lambda: True)

    result = commit_evolution("test commit", paths=[bad_path])
    assert result["committed"] is False
    assert result["reason"] == "syntax verification failed"
    assert result["verification"]["ok"] is False


def test_run_tests_before_commit():
    result = run_tests_before_commit(timeout=180)
    assert "passed" in result
    assert "output" in result
