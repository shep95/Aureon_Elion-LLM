"""Self-evolve fork system tests."""

from __future__ import annotations

import pytest

from app.self_evolve import (
    list_source_files,
    plan_evolution,
    repo_status,
    validate_repo_path,
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
