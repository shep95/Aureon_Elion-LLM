"""Model registry promotion tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.step3_training.registry import ModelRegistry


@pytest.fixture
def registry(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("pipeline.step3_training.registry.REGISTRY_DIR", tmp_path)
    monkeypatch.setattr("pipeline.step3_training.registry.MODELS_DIR", tmp_path / "models")
    (tmp_path / "models").mkdir()
    return ModelRegistry()


def test_should_promote_with_train_accuracy_only(registry: ModelRegistry):
    registry.promote(
        {
            "run_id": "base",
            "metrics": {"train_accuracy": 0.7},
            "artifact_path": "models/base.json",
        }
    )
    assert registry.should_promote({"train_accuracy": 0.75}) is True
    assert registry.should_promote({"train_accuracy": 0.65}) is False


def test_should_promote_prefers_val_accuracy(registry: ModelRegistry):
    registry.promote(
        {
            "run_id": "base",
            "metrics": {"val_accuracy": 0.8, "train_accuracy": 0.95},
            "artifact_path": "models/base.json",
        }
    )
    assert registry.should_promote({"val_accuracy": 0.85, "train_accuracy": 0.5}) is True
    assert registry.should_promote({"val_accuracy": 0.75, "train_accuracy": 0.99}) is False
