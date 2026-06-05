"""Step 3 — train text classifier and auto-promote if improved."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from pipeline.config import (
    LABELED_DIR,
    LEARNING_RATE,
    MODELS_DIR,
    TRAINING_EPOCHS,
    ensure_dirs,
)
from pipeline.queue import PipelineEvent, get_queue
from pipeline.step3_training.registry import ModelRegistry
from src.neural_network import NeuralNetwork
from src.text_features import TextFeatureExtractor, load_clean_corpus


def _prepare_training_data(labeled_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], TextFeatureExtractor]:
    rows = load_clean_corpus(labeled_path)
    if not rows:
        raise ValueError("No labeled data available")

    labels = sorted({row["label"] for row in rows})
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    texts = [f"{row['title']} {row['text']}" for row in rows]
    y = np.array([label_to_idx[row["label"]] for row in rows])

    extractor = TextFeatureExtractor(max_features=min(256, max(32, len(rows) * 4)))
    x = extractor.fit_transform(texts)
    return x, y, labels, extractor


def run_step3(epochs: int | None = None) -> dict[str, Any]:
    ensure_dirs()
    queue = get_queue()
    registry = ModelRegistry()
    epochs = epochs or TRAINING_EPOCHS

    labeled_path = LABELED_DIR / "labeled.jsonl"
    if not labeled_path.exists():
        return {
            "step": 3,
            "name": "automated_training",
            "status": "skipped",
            "reason": "no labeled data — run step 2 first",
        }

    x, y, labels, extractor = _prepare_training_data(labeled_path)
    num_classes = len(labels)

    if num_classes < 2:
        return {
            "step": 3,
            "name": "automated_training",
            "status": "skipped",
            "reason": "need at least 2 classes to train",
        }

    stratify = y if len(set(y.tolist())) > 1 and len(y) >= num_classes * 4 else None
    if stratify is None or len(y) < 12:
        x_train, y_train = x, y
        x_val, y_val = x[: max(1, len(y) // 5)], y[: max(1, len(y) // 5)]
    else:
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=stratify
        )

    hidden = min(128, max(16, x.shape[1] // 2))
    network = NeuralNetwork(
        layer_sizes=[x.shape[1], hidden, num_classes],
        learning_rate=LEARNING_RATE,
        seed=42,
        output_activation="softmax",
    )
    network.train(x_train, y_train, epochs=epochs, verbose_every=0)
    train_metrics = network.evaluate(x_train, y_train)
    val_metrics = network.evaluate(x_val, y_val)

    run_id = str(uuid.uuid4())[:8]
    artifact_dir = MODELS_DIR / f"run_{run_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "classifier.json"
    network.save(model_path)

    meta = {
        "run_id": run_id,
        "labels": labels,
        "feature_extractor": extractor.to_dict(),
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    metrics = {
        "train_accuracy": round(train_metrics["accuracy"], 4),
        "val_accuracy": round(val_metrics["accuracy"], 4),
    }
    run = registry.log_run(
        run_id=run_id,
        metrics=metrics,
        artifact_path=str(model_path),
        params={"epochs": epochs, "hidden_size": hidden, "num_classes": num_classes},
    )

    promoted = False
    promotion = None
    if registry.should_promote(metrics):
        promotion = registry.promote(run)
        promoted = True

    event = PipelineEvent(
        step="3_training",
        event_type="training_complete",
        payload={"run_id": run_id, "metrics": metrics, "promoted": promoted},
    )
    queue.publish("pipeline.training.complete", event)

    return {
        "step": 3,
        "name": "automated_training",
        "run_id": run_id,
        "labels": labels,
        "metrics": metrics,
        "artifact_path": str(model_path),
        "promoted_to_production": promoted,
        "production": promotion,
        "epochs": epochs,
    }
