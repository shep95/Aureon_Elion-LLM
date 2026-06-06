"""GitHub learning sync tests."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.learning_export import build_export_files
from app.learning_github_sync import (
    GitHubSyncConfig,
    is_github_sync_enabled,
    run_github_sync,
    sync_repo,
)


def test_github_sync_disabled_without_token(monkeypatch):
    monkeypatch.delenv("AUREON_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("AUREON_GITHUB_SYNC", "1")
    assert is_github_sync_enabled() is False


def test_github_sync_enabled_with_token(monkeypatch):
    monkeypatch.setenv("AUREON_GITHUB_SYNC", "1")
    monkeypatch.setenv("AUREON_GITHUB_TOKEN", "ghp_test")
    assert is_github_sync_enabled() is True
    cfg = GitHubSyncConfig.from_env()
    assert "houseofasher/SOLIA" in cfg.repos
    assert "ZorakCorp/Aureon-LLM" in cfg.repos
    assert cfg.on_startup is True
    assert cfg.interval_sec == 3600


def test_build_export_files_has_corpus_prefix():
    files = build_export_files()
    assert "learning-corpus/snapshot.json" in files
    assert "learning-corpus/README.md" in files
    assert "learning-corpus/learned_corpus.jsonl" in files
    assert "learning-corpus/documents.jsonl" in files
    assert "learning-corpus/document_labels.json" in files
    assert "learning-corpus/graduation_summary.json" in files
    snapshot = json.loads(files["learning-corpus/snapshot.json"].decode("utf-8"))
    assert "auto_learn" in snapshot
    assert "exported_at" in snapshot
    assert snapshot["github_sync"]["export_version"] == 2


@patch("app.learning_github_sync.requests.put")
@patch("app.learning_github_sync.requests.post")
@patch("app.learning_github_sync.requests.get")
def test_sync_repo_uploads_files(mock_get, mock_post, mock_put):
    mock_get.return_value = MagicMock(status_code=200, json=lambda: {"sha": "abc"})
    mock_put.return_value = MagicMock(status_code=200, raise_for_status=lambda: None)
    mock_post.return_value = MagicMock(status_code=404)

    files = {"learning-corpus/README.md": b"# test"}
    result = sync_repo("ZorakCorp", "Aureon-LLM", branch="learning-data", token="tok", files=files)
    assert result["files_uploaded"] == 1
    assert mock_put.called


@patch("app.learning_github_sync.requests.put")
@patch("app.learning_github_sync.requests.post")
@patch("app.learning_github_sync.requests.get")
def test_sync_repo_git_refs_404_falls_back_to_contents_api(mock_get, mock_post, mock_put):
    """git/refs 404 should not abort sync — Contents API creates the branch."""
    branch_ref = MagicMock(status_code=404)
    main_ref = MagicMock(
        status_code=200,
        json=lambda: {"object": {"sha": "abc123" * 5 + "abcd"}},
        raise_for_status=lambda: None,
    )
    file_ref = MagicMock(status_code=404, raise_for_status=lambda: None)
    repo_meta = MagicMock(
        status_code=200,
        json=lambda: {"default_branch": "main"},
        raise_for_status=lambda: None,
    )
    mock_get.side_effect = [branch_ref, repo_meta, main_ref, file_ref]
    mock_post.return_value = MagicMock(status_code=404)
    mock_put.return_value = MagicMock(status_code=201, raise_for_status=lambda: None)

    files = {"learning-corpus/README.md": b"# test"}
    result = sync_repo("shep95", "Aureon_Elion-LLM", branch="learning-data", token="tok", files=files)
    assert result["files_uploaded"] == 1
    assert mock_put.called


@patch("app.learning_github_sync.build_export_files")
@patch("app.learning_github_sync.sync_repo")
def test_run_github_sync_partial_success(mock_sync_repo, mock_build, monkeypatch):
    monkeypatch.setenv("AUREON_GITHUB_SYNC", "1")
    monkeypatch.setenv("AUREON_GITHUB_TOKEN", "ghp_test")
    monkeypatch.setenv(
        "AUREON_GITHUB_REPOS",
        "houseofasher/SOLIA,shep95/Aureon_Elion-LLM",
    )
    mock_build.return_value = {"learning-corpus/README.md": b"# test"}

    def _sync(owner, repo, **kwargs):
        if owner == "shep95":
            raise RuntimeError("404 Client Error: Not Found for url: git/refs")
        return {"repo": f"{owner}/{repo}", "files_uploaded": 1, "branch": "learning-data", "paths": []}

    mock_sync_repo.side_effect = _sync
    result = run_github_sync(reason="test")
    assert result["ok"] is True
    assert result["partial"] is True
    assert len(result["repos"]) == 1
    assert len(result["repos_failed"]) == 1


def test_run_github_sync_requires_token(monkeypatch):
    monkeypatch.delenv("AUREON_GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("AUREON_GITHUB_SYNC", "0")
    result = run_github_sync(reason="test")
    assert result["ok"] is False
