"""Load promoted per-scope brain classifiers for MoE chat routing."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select

from db.models import TrainingRun
from db.session import get_session
from pipeline.step4_evaluation.benchmarks import _load_production_model
from src.neural_network import NeuralNetwork
from src.text_features import TextFeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class ScopedClassifier:
    scope: str
    run_id: str
    network: NeuralNetwork
    labels: list[str]
    extractor: TextFeatureExtractor
    source: str


def _load_brain_artifact(artifact_path: str) -> ScopedClassifier | None:
    path = Path(artifact_path)
    if not path.is_file():
        return None
    meta_path = path.parent / "metadata.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    network = NeuralNetwork.load(path)
    extractor = TextFeatureExtractor.from_dict(meta["feature_extractor"])
    return ScopedClassifier(
        scope=str(meta.get("scope", path.parent.name)),
        run_id=path.parent.name,
        network=network,
        labels=list(meta.get("labels", [])),
        extractor=extractor,
        source="brain",
    )


def load_promoted_brain_classifiers(*, limit: int = 32) -> list[ScopedClassifier]:
    models: list[ScopedClassifier] = []
    try:
        with get_session() as session:
            runs = session.scalars(
                select(TrainingRun)
                .where(TrainingRun.promoted.is_(True), TrainingRun.artifact_path.isnot(None))
                .order_by(TrainingRun.created_at.desc())
                .limit(limit)
            ).all()
        seen_scopes: set[str] = set()
        for run in runs:
            if not run.artifact_path:
                continue
            loaded = _load_brain_artifact(run.artifact_path)
            if not loaded or loaded.scope in seen_scopes:
                continue
            seen_scopes.add(loaded.scope)
            models.append(loaded)
    except Exception:
        logger.debug("Brain classifier load skipped", exc_info=True)
    return models


def load_pipeline_classifier() -> ScopedClassifier | None:
    loaded = _load_production_model()
    if not loaded:
        return None
    network, labels, extractor = loaded
    return ScopedClassifier(
        scope="pipeline_production",
        run_id="pipeline",
        network=network,
        labels=labels,
        extractor=extractor,
        source="pipeline",
    )


def classify_moe(text: str) -> dict[str, Any] | None:
    """Pick highest-confidence label across brain specialists + pipeline production."""
    candidates: list[ScopedClassifier] = load_promoted_brain_classifiers()
    pipeline = load_pipeline_classifier()
    if pipeline:
        candidates.append(pipeline)

    if not candidates:
        return None

    min_conf = float(os.environ.get("AUREON_MOE_MIN_CONFIDENCE", "0.35"))
    best: dict[str, Any] | None = None

    for model in candidates:
        if len(model.labels) < 2:
            continue
        try:
            x = model.extractor.transform([text])
            proba = model.network.predict_proba(x)[0]
            idx = int(np.argmax(proba))
            confidence = float(proba[idx])
            label = model.labels[idx] if idx < len(model.labels) else model.labels[0]
        except Exception:
            continue

        entry = {
            "label": label,
            "confidence": round(confidence, 4),
            "labels_available": model.labels,
            "model": f"{model.source}_classifier",
            "scope": model.scope,
            "run_id": model.run_id,
        }
        if best is None or confidence > best["confidence"]:
            best = entry

    if best and best["confidence"] >= min_conf:
        best["routing"] = "mixture_of_experts"
        return best
    return None
