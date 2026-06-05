"""Auto-learn scheduler tests."""

from __future__ import annotations

from app.auto_learn import AutoLearnConfig, AutoLearnScheduler, _env_bool, _env_limit
from brain.cortex import iter_training_targets
from brain.domains.taxonomy import total_micro_subdomains


def test_env_bool():
    assert _env_bool("MISSING", default=True) is True
    assert _env_bool("MISSING", default=False) is False


def test_env_limit_all():
    assert _env_limit("MISSING", 5, maximum=30) == 5


def test_env_limit_zero_means_all(monkeypatch):
    monkeypatch.setenv("AUREON_AUTO_LEARN_DOMAIN_LIMIT", "0")
    assert _env_limit("AUREON_AUTO_LEARN_DOMAIN_LIMIT", 5, maximum=30) is None


def test_auto_learn_config_defaults_off_local(monkeypatch):
    monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
    monkeypatch.delenv("RAILWAY_SERVICE_ID", raising=False)
    monkeypatch.setenv("AUREON_AUTO_LEARN", "0")
    cfg = AutoLearnConfig.from_env()
    assert cfg.enabled is False


def test_auto_learn_config_railway_default(monkeypatch):
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
    monkeypatch.delenv("AUREON_AUTO_LEARN", raising=False)
    monkeypatch.delenv("AUREON_AUTO_LEARN_ALL", raising=False)
    cfg = AutoLearnConfig.from_env()
    assert cfg.enabled is True
    assert cfg.domain_limit == 30
    assert cfg.subdomain_limit == 8
    assert cfg.micro_limit == 17


def test_auto_learn_config_train_all(monkeypatch):
    monkeypatch.setenv("AUREON_AUTO_LEARN_ALL", "1")
    cfg = AutoLearnConfig.from_env()
    assert cfg.train_all is True
    assert cfg.domain_limit is None
    assert cfg.subdomain_limit is None
    assert cfg.micro_limit is None


def test_auto_learn_config_railway_via_postgres(monkeypatch):
    monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
    monkeypatch.delenv("RAILWAY_SERVICE_ID", raising=False)
    monkeypatch.delenv("AUREON_AUTO_LEARN", raising=False)
    monkeypatch.setenv("PORT", "8080")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host/db")
    cfg = AutoLearnConfig.from_env()
    assert cfg.enabled is True


def test_iter_training_targets_full_taxonomy():
    targets = iter_training_targets(
        domain_limit=None,
        subdomain_limit=None,
        micro_subdomain_limit=None,
    )
    assert len(targets) == total_micro_subdomains()


def test_iter_training_targets_single_micro():
    targets = iter_training_targets(domain_limit=1, subdomain_limit=1, micro_subdomain_limit=1)
    assert len(targets) == 1


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
