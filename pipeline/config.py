"""Pipeline configuration and data paths."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("PIPELINE_DATA_DIR", ROOT / "data"))

RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
LABELED_DIR = DATA_DIR / "labeled"
REVIEW_DIR = DATA_DIR / "review_queue"
MODELS_DIR = DATA_DIR / "models"
REGISTRY_DIR = DATA_DIR / "registry"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"
PREFERENCES_DIR = DATA_DIR / "preferences"
QUEUE_DIR = DATA_DIR / "queue"
SEEDS_DIR = ROOT / "data" / "seeds"

MIN_TEXT_LENGTH = 120
MAX_TEXT_LENGTH = 50_000
MIN_QUALITY_SCORE = 0.45
ACTIVE_LEARNING_REVIEW_RATE = 0.08
TRAINING_EPOCHS = 300
LEARNING_RATE = 0.3
ACCURACY_GATE = 0.55
REASONING_GATE = 0.33
CONSISTENCY_GATE = 0.50
ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "")


def ensure_dirs() -> None:
    for path in (
        RAW_DIR,
        CLEAN_DIR,
        LABELED_DIR,
        REVIEW_DIR,
        MODELS_DIR,
        REGISTRY_DIR,
        BENCHMARKS_DIR,
        PREFERENCES_DIR,
        QUEUE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
