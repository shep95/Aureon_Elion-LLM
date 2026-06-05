"""
Synthetic face features — mirrors the lecture's explanation directly.

The lecture says: you have weights for eyes, nose, chin, etc. You know these
features matter but not HOW MUCH they matter. Supervised learning discovers
the weights via backpropagation.
"""

from __future__ import annotations

import numpy as np

from .neural_network import NeuralNetwork

FEATURE_NAMES = ["eye_width", "nose_length", "chin_width", "cheekbone", "mouth_width"]


def generate_synthetic_faces(
    num_people: int = 5,
    samples_per_person: int = 40,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a face database where each person has distinct feature ranges.

    Clean data: numeric measurements, not opinions.
    Defined parameters: 5 facial features per face.
    """
    rng = np.random.default_rng(seed)
    features_list: list[np.ndarray] = []
    labels_list: list[int] = []

    for person_id in range(num_people):
        # Each person has a unique "face signature" in feature space
        base = rng.normal(person_id * 2.0, 0.3, len(FEATURE_NAMES))
        samples = rng.normal(base, 0.25, (samples_per_person, len(FEATURE_NAMES)))
        features_list.append(samples)
        labels_list.extend([person_id] * samples_per_person)

    x = np.vstack(features_list)
    y = np.array(labels_list)

    # Normalize to [0, 1] — clean numeric data
    x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0) + 1e-9)
    return x, y


def train_synthetic_face_classifier(
    num_people: int = 5,
    epochs: int = 400,
    learning_rate: float = 0.5,
    seed: int = 42,
) -> tuple[NeuralNetwork, dict[str, float]]:
    """
    Train on synthetic eye/nose/chin-style features.

    This is the lecture's core idea in its purest form: discover how much
    each facial feature weight contributes to identifying a person.
    """
    x, y = generate_synthetic_faces(num_people=num_people, seed=seed)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    split = int(len(y) * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print("Synthetic face features (lecture-style weights)")
    print(f"  Features: {', '.join(FEATURE_NAMES)}")
    print(f"  People in database: {num_people}")
    print(f"  Train: {len(y_train)}  |  Test: {len(y_test)}")
    print()

    network = NeuralNetwork(
        layer_sizes=[len(FEATURE_NAMES), 16, num_people],
        learning_rate=learning_rate,
        seed=seed,
        output_activation="softmax",
    )
    print(f"Neural network weights to learn: {network.num_weights:,}")
    print()
    print("Training — computer discovers how much each feature matters...")

    network.train(x_train, y_train, epochs=epochs, verbose_every=max(epochs // 4, 1))

    train_metrics = network.evaluate(x_train, y_train)
    test_metrics = network.evaluate(x_test, y_test)
    print()
    print(f"Train accuracy: {train_metrics['accuracy']:.1%}")
    print(f"Test accuracy:  {test_metrics['accuracy']:.1%}")

    # Show learned first-layer weights (feature importance — the lecture's "weights")
    print()
    print("Learned feature weights (first layer — how much each feature matters):")
    avg_weights = np.abs(network.weights[0]).mean(axis=1)
    for name, weight in sorted(zip(FEATURE_NAMES, avg_weights), key=lambda t: -t[1]):
        bar = "#" * int(weight * 40)
        print(f"  {name:14s}  {weight:.3f}  {bar}")

    return network, test_metrics
