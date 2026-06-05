"""Step 5 — RLHF approximation via reward model."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from pipeline.config import MODELS_DIR, PREFERENCES_DIR, ensure_dirs
from pipeline.step3_training.registry import ModelRegistry
from src.neural_network import NeuralNetwork
from src.text_features import TextFeatureExtractor

# Preference pairs: (prompt_context, preferred_response, rejected_response)
DEFAULT_PREFERENCES = [
    {
        "context": "Explain backpropagation simply.",
        "preferred": "Backpropagation adjusts network weights by propagating prediction errors backward using the chain rule.",
        "rejected": "Backpropagation is magic that makes AI conscious and all-knowing.",
    },
    {
        "context": "What makes training data good?",
        "preferred": "Good training data is clean, labeled, verifiable, and matched to a measurable goal.",
        "rejected": "Good training data is whatever gets the most clicks regardless of accuracy.",
    },
    {
        "context": "Describe edge cases in ML systems.",
        "preferred": "Edge cases are inputs outside the training distribution that cause unpredictable model failures.",
        "rejected": "Edge cases never matter if the model is large enough.",
    },
    {
        "context": "How should sources be chosen?",
        "preferred": "Use diverse primary sources with peer review and domain specificity rather than any single geography.",
        "rejected": "Only use one institution because geography determines intelligence.",
    },
]


def _build_preference_dataset(
    preferences: list[dict],
    extractor: TextFeatureExtractor,
) -> tuple[np.ndarray, np.ndarray]:
    texts: list[str] = []
    labels: list[int] = []

    for pair in preferences:
        preferred_text = f"{pair['context']} {pair['preferred']}"
        rejected_text = f"{pair['context']} {pair['rejected']}"
        texts.extend([preferred_text, rejected_text])
        labels.extend([1, 0])

    x = extractor.fit_transform(texts)
    y = np.array(labels)
    return x, y


def train_reward_model(
    preferences: list[dict] | None = None,
    epochs: int = 400,
) -> tuple[NeuralNetwork, TextFeatureExtractor, dict[str, float]]:
    preferences = preferences or DEFAULT_PREFERENCES
    extractor = TextFeatureExtractor(max_features=128)
    x, y = _build_preference_dataset(preferences, extractor)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    network = NeuralNetwork(
        layer_sizes=[x.shape[1], 32, 1],
        learning_rate=0.3,
        seed=42,
        output_activation="sigmoid",
    )
    network.train(x_train, y_train, epochs=epochs, verbose_every=0)
    val_metrics = network.evaluate(x_val, y_val)
    return network, extractor, val_metrics


def score_response(
    reward_model: NeuralNetwork,
    extractor: TextFeatureExtractor,
    context: str,
    response: str,
) -> float:
    text = f"{context} {response}"
    x = extractor.transform([text])
    return float(reward_model.predict_proba(x)[0, 0])


def run_step5(epochs: int = 400) -> dict[str, Any]:
    ensure_dirs()
    registry = ModelRegistry()
    production = registry.get_production()

    if not production:
        return {
            "step": 5,
            "name": "rlhf_approximation",
            "status": "skipped",
            "reason": "no production model — run steps 3-4 first",
        }

    prefs_path = PREFERENCES_DIR / "preferences.json"
    if prefs_path.exists():
        preferences = json.loads(prefs_path.read_text(encoding="utf-8"))
    else:
        preferences = DEFAULT_PREFERENCES
        prefs_path.write_text(json.dumps(preferences, indent=2), encoding="utf-8")

    reward_model, extractor, val_metrics = train_reward_model(preferences, epochs=epochs)

    run_id = str(uuid.uuid4())[:8]
    artifact_dir = MODELS_DIR / f"reward_{run_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "reward_model.json"
    reward_model.save(model_path)
    (artifact_dir / "extractor.json").write_text(
        json.dumps(extractor.to_dict(), indent=2), encoding="utf-8"
    )

    # Score sample outputs — preferred should rank higher than rejected
    ranking_checks = []
    for pair in preferences[:4]:
        preferred_score = score_response(reward_model, extractor, pair["context"], pair["preferred"])
        rejected_score = score_response(reward_model, extractor, pair["context"], pair["rejected"])
        ranking_checks.append(
            {
                "context": pair["context"][:60],
                "preferred_score": round(preferred_score, 4),
                "rejected_score": round(rejected_score, 4),
                "correct_ranking": preferred_score > rejected_score,
            }
        )

    ranking_accuracy = sum(int(r["correct_ranking"]) for r in ranking_checks) / max(
        len(ranking_checks), 1
    )

    return {
        "step": 5,
        "name": "rlhf_approximation",
        "reward_model_path": str(model_path),
        "reward_val_accuracy": round(val_metrics["accuracy"], 4),
        "ranking_accuracy": round(ranking_accuracy, 4),
        "ranking_checks": ranking_checks,
        "note": (
            "Reward model replaces human raters for scoring outputs. "
            "Production classifier can be retrained weighting high-reward samples."
        ),
    }
