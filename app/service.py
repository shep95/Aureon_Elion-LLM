"""Training orchestration returning JSON-serializable reports for the API."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.face_matcher import (
    build_identity_classifier_data,
    build_match_pairs,
    load_face_database,
)
from src.neural_network import NeuralNetwork
from src.synthetic_faces import FEATURE_NAMES, generate_synthetic_faces


def _clamp_epochs(epochs: int, maximum: int = 500) -> int:
    return max(1, min(epochs, maximum))


def concepts() -> dict[str, Any]:
    return {
        "title": "Supervised Machine Learning",
        "summary": (
            "AI in practice is supervised machine learning: you provide labeled inputs "
            "and measurable outputs, and backpropagation discovers the weights."
        ),
        "traditional_vs_ml": {
            "traditional": "You write the algorithm: output = f(input), e.g. 1 + 1 = 2",
            "supervised_ml": (
                "You provide inputs AND correct labels; the computer learns weights "
                "via backpropagation (called 'deep learning' at scale)."
            ),
        },
        "constraints": [
            {"name": "clean_data", "description": "Correct structured labels, not opinions"},
            {"name": "measurable_goal", "description": "Yes/no or class ID — not 'what is good?'"},
            {"name": "defined_parameters", "description": "A bounded labeled database"},
        ],
        "terminology": {
            "neural_network": "A weighting system",
            "backpropagation": "How weights are learned from labeled errors",
            "deep_learning": "Backpropagation with many layers",
            "black_box": "Learned weights are not human-readable",
            "edge_cases": "Inputs outside training data break predictions",
        },
    }


def run_synthetic_demo(epochs: int = 200, seed: int = 42) -> dict[str, Any]:
    epochs = _clamp_epochs(epochs)
    num_people = 5

    x, y = generate_synthetic_faces(num_people=num_people, seed=seed)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    split = int(len(y) * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    network = NeuralNetwork(
        layer_sizes=[len(FEATURE_NAMES), 16, num_people],
        learning_rate=0.5,
        seed=seed,
        output_activation="softmax",
    )
    history = network.train(x_train, y_train, epochs=epochs, verbose_every=0)

    train_metrics = network.evaluate(x_train, y_train)
    test_metrics = network.evaluate(x_test, y_test)
    avg_weights = np.abs(network.weights[0]).mean(axis=1)
    feature_weights = [
        {"feature": name, "weight": round(float(weight), 4)}
        for name, weight in sorted(
            zip(FEATURE_NAMES, avg_weights), key=lambda item: -item[1]
        )
    ]

    return {
        "demo": "synthetic_face_features",
        "description": "Eye/nose/chin-style weights discovered by backpropagation",
        "epochs": epochs,
        "features": FEATURE_NAMES,
        "people_in_database": num_people,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "learnable_weights": network.num_weights,
        "train_accuracy": round(train_metrics["accuracy"], 4),
        "test_accuracy": round(test_metrics["accuracy"], 4),
        "feature_weights": feature_weights,
        "final_loss": round(history[-1].loss, 6) if history else None,
    }


def run_match_demo(
    epochs: int = 200,
    people: int = 40,
    seed: int = 42,
) -> dict[str, Any]:
    epochs = _clamp_epochs(epochs)
    people = max(2, min(people, 40))
    num_pairs = min(4000, people * 100)

    dataset = load_face_database(num_people=people, seed=seed)
    pairs, labels = build_match_pairs(dataset.features, dataset.labels, num_pairs, seed=seed)

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        pairs, labels, test_size=0.2, random_state=seed, stratify=labels
    )

    network = NeuralNetwork(
        layer_sizes=[pairs.shape[1], 64, 1],
        learning_rate=0.3,
        seed=seed,
        output_activation="sigmoid",
    )
    history = network.train(x_train, y_train, epochs=epochs, verbose_every=0)
    metrics = network.evaluate(x_test, y_test)

    return {
        "demo": "binary_face_matching",
        "description": "Measurable yes/no goal: do two faces belong to the same person?",
        "epochs": epochs,
        "people_in_database": people,
        "faces_in_database": dataset.num_faces,
        "train_pairs": len(y_train),
        "test_pairs": len(y_test),
        "input_features": pairs.shape[1],
        "learnable_weights": network.num_weights,
        "test_accuracy": round(metrics["accuracy"], 4),
        "final_loss": round(history[-1].loss, 6) if history else None,
    }


def run_identify_demo(
    epochs: int = 200,
    people: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    epochs = _clamp_epochs(epochs)
    people = max(2, min(people, 10))

    dataset = load_face_database(num_people=people, seed=seed)
    x_train, x_test, y_train, y_test = build_identity_classifier_data(dataset, seed=seed)

    network = NeuralNetwork(
        layer_sizes=[dataset.num_features, 128, people],
        learning_rate=0.3,
        seed=seed,
        output_activation="softmax",
    )
    history = network.train(x_train, y_train, epochs=epochs, verbose_every=0)
    metrics = network.evaluate(x_test, y_test)
    edge_case = _edge_case_report(network, dataset)

    return {
        "demo": "person_identification",
        "description": "Classify which person a face belongs to (harder multi-class task)",
        "epochs": epochs,
        "people_in_database": people,
        "faces_in_database": dataset.num_faces,
        "feature_dimensions": dataset.num_features,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "learnable_weights": network.num_weights,
        "test_accuracy": round(metrics["accuracy"], 4),
        "final_loss": round(history[-1].loss, 6) if history else None,
        "edge_case": edge_case,
    }


def _edge_case_report(network: NeuralNetwork, dataset) -> dict[str, Any]:
    normal_idx = 0
    normal_face = dataset.features[normal_idx : normal_idx + 1]
    normal_label = int(dataset.labels[normal_idx])
    inverted_face = 1.0 - normal_face

    pred_normal = int(network.predict(normal_face))
    pred_inverted = int(network.predict(inverted_face))
    changed = pred_normal != pred_inverted

    return {
        "scenario": "Inverted pixel values (outside training distribution)",
        "actual_person": normal_label,
        "predicted_normal": pred_normal,
        "predicted_inverted": pred_inverted,
        "prediction_changed": changed,
        "note": (
            "Edge case changed the prediction — the black box failed on unseen patterns."
            if changed
            else "Model was robust here, but edge cases often break real systems."
        ),
    }
