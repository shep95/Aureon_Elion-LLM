"""Autonomous self-evolve agent — fork-only cycle without main push."""

from __future__ import annotations

from app.self_evolve_agent import get_history, run_autonomous_evolution


def _stub_evolve(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "app.self_evolve.create_evolution_branch",
        lambda desc: {"branch": "aureon/self-evolve-test-1", "description": desc},
    )
    monkeypatch.setattr(
        "app.self_evolve.read_source",
        lambda p: {"path": p, "content": '"""module"""\n\npass\n'},
    )

    written: list[str] = []

    def _write(path: str, content: str) -> dict:
        written.append(path)
        return {"path": path, "written": True}

    monkeypatch.setattr("app.self_evolve.write_source", _write)
    monkeypatch.setattr(
        "app.self_evolve.commit_evolution",
        lambda msg, paths=None: {"committed": True, "message": msg, "paths": paths},
    )
    monkeypatch.setattr(
        "app.self_evolve.push_fork",
        lambda branch=None, approved=False: {
            "pushed": approved,
            "branch": branch,
            "remote": "houseofasher",
        },
    )
    monkeypatch.setattr(
        "app.self_evolve.repo_status",
        lambda: {"current_branch": "aureon/self-evolve-test-1", "policy": "fork-only"},
    )
    monkeypatch.setattr("app.self_evolve_agent.HISTORY_PATH", tmp_path / "history.jsonl")
    return written


def test_run_autonomous_evolution_pushes_fork(monkeypatch, tmp_path):
    written = _stub_evolve(monkeypatch, tmp_path)
    result = run_autonomous_evolution("improve chat routing", auto_push_fork=True)
    assert result["ok"] is True
    assert result["autonomous"] is True
    assert result["branch"] == "aureon/self-evolve-test-1"
    assert result["push"]["pushed"] is True
    assert result["history_entry"]["main_blocked"] is True
    assert written


def test_run_autonomous_evolution_skips_push_when_disabled(monkeypatch, tmp_path):
    _stub_evolve(monkeypatch, tmp_path)
    result = run_autonomous_evolution("docstring stamp only", auto_push_fork=False)
    assert result["push"]["pushed"] is False


def test_get_history_records_events(monkeypatch, tmp_path):
    _stub_evolve(monkeypatch, tmp_path)
    run_autonomous_evolution("history test task")
    history = get_history(limit=5)
    assert len(history) >= 1
    assert history[0]["action"] == "autonomous_evolution"
    assert history[0]["task"] == "history test task"
