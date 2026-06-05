"""Auto-learn scheduler tests."""

from __future__ import annotations

import os

from app.auto_learn import AutoLearnConfig, AutoLearnScheduler, _env_bool


def test_env_bool():
    assert _env_bool("MISSING", default=True) is True
    assert _env_bool("MISSING", default=False) is False


def test_auto_learn_config_defaults_off_local(monkeypatch):
    monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
    monkeypatch.delenv("RAILWAY_SERVICE_ID", raising=False)
    monkeypatch.setenv("AUREON_AUTO_LEARN", "0")
    cfg = AutoLearnConfig.from_env()
    assert cfg.enabled is False


def test_auto_learn_config_railway_default(monkeypatch):
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
    monkeypatch.delenv("AUREON_AUTO_LEARN", raising=False)
    cfg = AutoLearnConfig.from_env()
    assert cfg.enabled is True


def test_auto_learn_config_railway_via_postgres(monkeypatch):
    monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
    monkeypatch.delenv("RAILWAY_SERVICE_ID", raising=False)
    monkeypatch.delenv("AUREON_AUTO_LEARN", raising=False)
    monkeypatch.setenv("PORT", "8080")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host/db")
    cfg = AutoLearnConfig.from_env()
    assert cfg.enabled is True


def test_is_railway(monkeypatch):
    from app.startup import is_railway

    monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert is_railway() is False
    monkeypatch.setenv("PORT", "8080")
    monkeypatch.setenv("DATABASE_URL", "postgresql://x:y@host/db")
    assert is_railway() is True


def test_scheduler_status_when_disabled():
    scheduler = AutoLearnScheduler(AutoLearnConfig(enabled=False))
    status = scheduler.status()
    assert status["enabled"] is False
    assert status["running"] is False
