"""Auto-learn scheduler tests."""

from __future__ import annotations

from app.auto_learn import (
    AutoLearnConfig,
    AutoLearnScheduler,
    _env_bool,
    _env_limit,
    load_target_cursor,
    save_target_cursor,
    select_batch_targets,
)
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


def test_auto_learn_config_railway_default_off(monkeypatch):
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
    monkeypatch.delenv("AUREON_AUTO_LEARN", raising=False)
    monkeypatch.delenv("AUREON_AUTO_LEARN_ALL", raising=False)
    monkeypatch.delenv("AUREON_AUTO_LEARN_BATCH_SIZE", raising=False)
    monkeypatch.delenv("AUREON_AUTO_LEARN_INTERVAL_SEC", raising=False)
    monkeypatch.delenv("AUREON_AUTO_LEARN_CONTINUOUS", raising=False)
    cfg = AutoLearnConfig.from_env()
    assert cfg.enabled is False
    assert cfg.on_startup is False
    assert cfg.train_all is False
    assert cfg.continuous is False
    assert cfg.interval_sec == 3600
    assert cfg.domain_limit == 30
    assert cfg.batch_size is None


def test_auto_learn_config_explicit_enable_on_railway(monkeypatch):
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
    monkeypatch.setenv("AUREON_AUTO_LEARN", "1")
    monkeypatch.setenv("AUREON_AUTO_LEARN_ON_STARTUP", "1")
    monkeypatch.setenv("AUREON_AUTO_LEARN_ALL", "1")
    monkeypatch.setenv("AUREON_AUTO_LEARN_CONTINUOUS", "1")
    monkeypatch.setenv("AUREON_AUTO_LEARN_BATCH_SIZE", "25")
    cfg = AutoLearnConfig.from_env()
    assert cfg.enabled is True
    assert cfg.on_startup is True
    assert cfg.train_all is True
    assert cfg.continuous is True
    assert cfg.interval_sec == 0
    assert cfg.domain_limit is None
    assert cfg.batch_size == 25


def test_auto_learn_config_explicit_interval_disables_continuous(monkeypatch):
    monkeypatch.setenv("AUREON_AUTO_LEARN_CONTINUOUS", "1")
    monkeypatch.setenv("AUREON_AUTO_LEARN_INTERVAL_SEC", "600")
    cfg = AutoLearnConfig.from_env()
    assert cfg.continuous is False
    assert cfg.interval_sec == 600


def test_auto_learn_config_interval_zero_enables_continuous(monkeypatch):
    monkeypatch.setenv("AUREON_AUTO_LEARN_INTERVAL_SEC", "0")
    cfg = AutoLearnConfig.from_env()
    assert cfg.continuous is True
    assert cfg.interval_sec == 0


def test_auto_learn_config_batch_size_zero_means_all(monkeypatch):
    monkeypatch.setenv("AUREON_AUTO_LEARN_BATCH_SIZE", "0")
    cfg = AutoLearnConfig.from_env()
    assert cfg.batch_size is None


def test_select_batch_targets_wraps():
    targets = [(f"d{i}", "s", "m") for i in range(10)]
    chunk, nxt = select_batch_targets(targets, cursor=8, batch_size=5)
    assert len(chunk) == 2
    assert nxt == 0


def test_target_cursor_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("AUREON_DATA_DIR", str(tmp_path))
    save_target_cursor(25, total=862)
    assert load_target_cursor() == 25


def test_auto_learn_config_train_all(monkeypatch):
    monkeypatch.setenv("AUREON_AUTO_LEARN_ALL", "1")
    cfg = AutoLearnConfig.from_env()
    assert cfg.train_all is True
    assert cfg.domain_limit is None
    assert cfg.subdomain_limit is None
    assert cfg.micro_limit is None


def test_auto_learn_config_railway_defaults_do_not_train_all(monkeypatch):
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
    monkeypatch.delenv("AUREON_AUTO_LEARN_ALL", raising=False)
    cfg = AutoLearnConfig.from_env()
    assert cfg.train_all is False
    assert cfg.domain_limit == 30


def test_auto_learn_config_railway_via_postgres(monkeypatch):
    monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
    monkeypatch.delenv("RAILWAY_SERVICE_ID", raising=False)
    monkeypatch.delenv("AUREON_AUTO_LEARN", raising=False)
    monkeypatch.setenv("PORT", "8080")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host/db")
    cfg = AutoLearnConfig.from_env()
    assert cfg.enabled is False


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
