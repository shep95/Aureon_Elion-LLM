"""Activity logging tests."""

from __future__ import annotations

import json
import logging

import pytest

from app import activity_log


@pytest.fixture(autouse=True)
def reset_activity_env(monkeypatch):
    monkeypatch.delenv("AUREON_ACTIVITY_LOG", raising=False)
    monkeypatch.delenv("AUREON_LOG_JSON", raising=False)
    monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)


def test_activity_log_disabled_locally(monkeypatch, caplog):
    monkeypatch.setenv("AUREON_ACTIVITY_LOG", "0")
    with caplog.at_level(logging.INFO, logger="aureon.ai"):
        activity_log.log_ai_activity("test_action", foo="bar")
    assert not any("test_action" in r.message for r in caplog.records)


def test_activity_log_enabled_explicit(monkeypatch, caplog):
    monkeypatch.setenv("AUREON_ACTIVITY_LOG", "1")
    monkeypatch.setenv("AUREON_LOG_JSON", "1")
    with caplog.at_level(logging.INFO, logger="aureon.ai"):
        activity_log.log_ai_activity("test_action", foo="bar")
    assert len(caplog.records) == 1
    payload = json.loads(caplog.records[0].message)
    assert payload["action"] == "test_action"
    assert payload["foo"] == "bar"
    assert payload["component"] == "aureon-ai"


def test_cycle_id_thread_local():
    activity_log.clear_cycle_id()
    assert activity_log.current_cycle_id() is None
    cid = activity_log.new_cycle_id("t")
    assert activity_log.current_cycle_id() == cid
    assert cid.startswith("t-")
    activity_log.clear_cycle_id()
    assert activity_log.current_cycle_id() is None


def test_railway_defaults_on(monkeypatch):
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
    assert activity_log.is_activity_logging_enabled() is True
    assert activity_log.use_json_format() is True
