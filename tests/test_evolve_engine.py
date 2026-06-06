"""Algorithmic evolve engine tests."""

from __future__ import annotations

from brain.evolve_engine import (
    classify_evolve_task,
    propose_evolution_writes,
    propose_file_patch,
)


def test_classify_evolve_task():
    assert classify_evolve_task("create new module for sentiment") == "new_module"
    assert classify_evolve_task("implement helper function for parsing") == "code"
    assert classify_evolve_task("improve philosophy routing") == "routing"


def test_propose_file_patch_annotate(monkeypatch):
    monkeypatch.setattr(
        "brain.evolve_engine.read_source",
        lambda p: {"path": p, "content": '"""module"""\n\ndef handle() -> str:\n    return "ok"\n'},
    )
    monkeypatch.setattr(
        "brain.evolve_engine.analyze_file_for_task",
        lambda p, t: {
            "path": p,
            "task": t,
            "functions": [{"name": "handle", "task_relevant": True}],
            "task_relevant_functions": ["handle"],
            "issues": [],
        },
    )
    monkeypatch.setattr(
        "brain.evolve_engine.reason_about_task",
        lambda task, analysis=None: {"hint": "route faith queries", "confidence": 0.7, "abstained": False},
    )
    result = propose_file_patch("brain/philosophy_handler.py", "improve philosophy routing", strategy="routing")
    assert result["patched"] is True
    assert "handle" in result["content"] or "SOLIA auto-evolve" in result["content"]


def test_propose_evolution_writes(monkeypatch):
    monkeypatch.setattr(
        "app.self_evolve.plan_evolution",
        lambda task: {
            "task": task,
            "suggested_files": ["brain/philosophy_handler.py"],
            "analysis": [],
        },
    )
    monkeypatch.setattr(
        "brain.evolve_engine.propose_file_patch",
        lambda path, task, strategy=None: {
            "path": path,
            "patched": True,
            "content": '"""patched"""\n',
            "method": "annotate_stamp",
            "strategy": strategy or "routing",
        },
    )
    result = propose_evolution_writes("improve philosophy routing")
    assert result["brain"] == "algorithmic (predict + code_master + AST)"
    assert len(result["writes"]) == 1
    assert result["writes"][0]["path"] == "brain/philosophy_handler.py"
