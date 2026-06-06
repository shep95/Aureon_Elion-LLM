"""Pytest defaults — keep predict-brain training fast in CI/local tests."""

from __future__ import annotations

import os

os.environ.setdefault("AUREON_PREDICT_EPOCHS", "60")
os.environ.setdefault("AUREON_PREDICT_TRAIN_CHUNK", "64")
os.environ.setdefault("AUREON_PREDICT_MAX_SEQ", "128")
os.environ.setdefault("AUREON_PREDICT_MAX_VOCAB", "2000")
os.environ.setdefault("AUREON_PREDICT_D_MODEL", "48")
os.environ.setdefault("AUREON_PREDICT_LAYERS", "4")
