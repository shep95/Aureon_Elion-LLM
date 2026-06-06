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
    assert "houseofasher/Aureon-LLM" in cfg.repos
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


def test_run_github_sync_requires_token(monkeypatch):
    monkeypatch.delenv("AUREON_GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("AUREON_GITHUB_SYNC", "0")
    result = run_github_sync(reason="test")
    assert result["ok"] is False
