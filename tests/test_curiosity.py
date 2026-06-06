"""Curiosity market research — self-directed web research, sandbox, human approval."""

from __future__ import annotations

import pytest

from app.chat_service import chat
from app.curiosity_proposals import approve_proposal, get_proposal, list_proposals
from app.curiosity_sandbox import (
    build_sandbox,
    deploy_proposal,
    push_github_brief,
    run_curiosity_cycle,
    verify_prototype_code,
    verify_sandbox_proposal,
)
from brain.curiosity_engine import (
    generate_market_queries,
    is_curiosity_enabled,
    is_curiosity_request,
    run_market_research,
)


def _mock_search(query: str, *, max_results: int = 5):
    return [
        {
            "type": "web",
            "text": f"Advanced AI algorithms in {query[:40]} include security math and design tools.",
            "source": "mock.example",
            "url": "https://mock.example/research",
        }
    ]


@pytest.fixture
def proposals_file(tmp_path, monkeypatch):
    path = tmp_path / "proposals.jsonl"
    monkeypatch.setattr("app.curiosity_proposals.PROPOSALS_DIR", tmp_path)
    monkeypatch.setattr("app.curiosity_proposals.PROPOSALS_PATH", path)
    monkeypatch.setattr("app.curiosity_sandbox.SANDBOX_ROOT", tmp_path / "sandbox")
    monkeypatch.setattr("app.curiosity_sandbox.RAILWAY_SECTIONS", tmp_path / "railway")
    return path


def test_is_curiosity_enabled_default():
    assert is_curiosity_enabled() is True


def test_is_curiosity_request():
    assert is_curiosity_request("/curious")
    assert is_curiosity_request("gets curious about what it is and does market research")
    assert not is_curiosity_request("what is 2+2")


def test_generate_market_queries():
    queries = generate_market_queries()
    assert len(queries) >= 2
    domains = {q["domain"] for q in queries}
    assert "identity" in domains
    assert "market" in domains


def test_run_market_research_mocked(proposals_file):
    payload = run_market_research(search_fn=_mock_search)
    assert payload["ok"] is True
    assert payload["self_intro"]
    assert len(payload["research"]) >= 2
    assert payload["curiosity_reflection"]


def test_curiosity_cycle_creates_proposal_and_sandbox(proposals_file):
    result = run_curiosity_cycle(search_fn=_mock_search)
    assert result["ok"] is True
    proposal = result["proposal"]
    assert proposal["id"]
    assert proposal["status"] == "pending_approval"
    assert result["sandbox"]["sandbox_built"] is True
    assert result["sandbox"]["sandbox_path"]
    assert "approve" in result["report"].lower()


def test_build_sandbox_idempotent(proposals_file):
    result = run_curiosity_cycle(search_fn=_mock_search)
    pid = result["proposal"]["id"]
    sandbox = build_sandbox(pid)
    assert sandbox["ok"] is True
    assert sandbox["sandbox_built"] is True
    assert any("research-brief" in f for f in sandbox["sandbox_files"])


def test_github_push_requires_approval(proposals_file):
    result = run_curiosity_cycle(search_fn=_mock_search)
    proposal = get_proposal(result["proposal"]["id"])
    blocked = push_github_brief(proposal, approved=False)
    assert blocked["pushed"] is False


def test_prototype_modules_verify_and_run(proposals_file):
    result = run_curiosity_cycle(search_fn=_mock_search)
    pid = result["proposal"]["id"]
    verification = verify_sandbox_proposal(pid)
    assert verification["ok"] is True
    assert verification["module_count"] >= 1
    for mod in verification["modules"]:
        assert mod["verification"]["ok"] is True
        assert mod["verification"]["security_valid"] is True
        assert mod["runtime"]["ok"] is True
        assert mod["runtime"]["result"]["status"] == "sandbox_prototype"


def test_verify_prototype_rejects_unsafe_code():
    bad = "import os\n\ndef run():\n    os.system('echo hi')\n    return {'status': 'sandbox_prototype'}\n"
    check = verify_prototype_code(bad)
    assert check["ok"] is False
    assert check["security_valid"] is False


def test_deploy_blocked_when_rejected(proposals_file):
    result = run_curiosity_cycle(search_fn=_mock_search)
    pid = result["proposal"]["id"]
    approve_proposal(pid, approved=False, reviewer="test")
    deploy = deploy_proposal(pid, approve_github=True)
    assert deploy["ok"] is False
    assert deploy["error"] == "rejected"


def test_deploy_blocked_when_already_deployed(monkeypatch, proposals_file):
    result = run_curiosity_cycle(search_fn=_mock_search)
    pid = result["proposal"]["id"]
    _mock_deploy_git(monkeypatch)
    first = deploy_proposal(pid, approve_github=True, reviewer="test")
    assert first["ok"] is True
    second = deploy_proposal(pid, approve_github=True, reviewer="test")
    assert second["ok"] is False
    assert second["error"] == "already_deployed"


def test_deploy_blocked_on_commit_failure(monkeypatch, proposals_file):
    result = run_curiosity_cycle(search_fn=_mock_search)
    pid = result["proposal"]["id"]
    _mock_deploy_git(monkeypatch)
    monkeypatch.setattr(
        "app.self_evolve.commit_evolution",
        lambda msg, paths=None: {"committed": False, "reason": "test suite failed"},
    )
    deploy = deploy_proposal(pid, approve_github=False, reviewer="test")
    assert deploy["ok"] is False
    assert deploy["error"] == "commit_failed"
    assert get_proposal(pid)["status"] != "deployed"


def test_fork_push_requires_explicit_approval(proposals_file):
    result = run_curiosity_cycle(search_fn=_mock_search)
    proposal = get_proposal(result["proposal"]["id"])
    from app.self_evolve import push_fork

    blocked = push_fork(branch="aureon/test", approved=False)
    assert blocked["pushed"] is False


def test_sandbox_isolated_from_repo_brain(proposals_file, monkeypatch):
    writes: list[str] = []

    def track_write(path, content):
        writes.append(path)
        return {"path": path, "written": True}

    monkeypatch.setattr("app.curiosity_sandbox.write_source", track_write)
    run_curiosity_cycle(search_fn=_mock_search)
    assert writes == []


def _mock_deploy_git(monkeypatch):
    monkeypatch.setattr(
        "app.curiosity_sandbox.create_evolution_branch",
        lambda task: {"branch": "aureon/test-curiosity-branch", "description": task},
    )
    monkeypatch.setattr(
        "app.curiosity_sandbox.write_source",
        lambda path, content: {"path": path, "written": True},
    )
    monkeypatch.setattr(
        "app.self_evolve.commit_evolution",
        lambda msg, paths=None: {"committed": True, "sha": "abc123"},
    )
    monkeypatch.setattr(
        "app.curiosity_sandbox.push_fork",
        lambda **kwargs: {"pushed": False, "reason": "test"},
    )
    monkeypatch.setattr(
        "app.curiosity_sandbox.push_github_brief",
        lambda proposal, approved=False: {
            "pushed": True,
            "branch": "curiosity/test",
            "repos": ["mock/repo"],
        },
    )


def test_approve_without_git(monkeypatch, proposals_file):
    result = run_curiosity_cycle(search_fn=_mock_search)
    pid = result["proposal"]["id"]
    _mock_deploy_git(monkeypatch)
    deploy = deploy_proposal(pid, approve_github=True, approve_push=False, reviewer="test")
    assert deploy["ok"] is True
    assert deploy["deploy"]["sandbox_verification"]["ok"] is True
    updated = get_proposal(pid)
    assert updated["status"] == "deployed"


def test_list_proposals(proposals_file):
    run_curiosity_cycle(search_fn=_mock_search)
    listed = list_proposals()
    assert listed["count"] >= 1


def test_chat_curious_command(proposals_file, monkeypatch):
    import app.curiosity_sandbox as cs

    monkeypatch.setattr(
        cs,
        "run_curiosity_cycle",
        lambda **kwargs: {
            "ok": True,
            "report": "**Curiosity market research complete**\n\nTest report.",
            "proposal": {"id": "test-id-1234"},
        },
    )
    result = chat("/curious")
    assert result["kind"] == "curiosity"
    assert "Curiosity market research complete" in result["reply"]


def test_chat_natural_curiosity(proposals_file, monkeypatch):
    import app.curiosity_sandbox as cs

    monkeypatch.setattr(
        cs,
        "run_curiosity_cycle",
        lambda **kwargs: {
            "ok": True,
            "report": "**Curiosity market research complete**\n\nNatural language test.",
        },
    )
    msg = "the algorithm gets curious about what it is and does market research on the web"
    result = chat(msg)
    assert result["kind"] == "curiosity"
