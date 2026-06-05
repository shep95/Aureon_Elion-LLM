"""
Face matching with supervised machine learning.

Maps directly to the lecture example:
  - Database of faces (defined parameters)
  - Clean labeled images (clean data)
  - Measurable goal: does face A match person B? yes/no (binary classification)

The network learns weights for each pixel/feature — analogous to the lecture's
weights for eyes, nose, chin — until each face maps to a distinct mathematical model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

from .neural_network import NeuralNetwork


@dataclass
class FaceDataset:
    """Labeled face database — satisfies the 'defined parameters' constraint."""

    features: np.ndarray
    labels: np.ndarray
    person_names: list[str]

    @property
    def num_faces(self) -> int:
        return self.features.shape[0]

    @property
    def num_features(self) -> int:
        return self.features.shape[1]

    @property
    def num_people(self) -> int:
        return len(set(self.labels.tolist()))


def _downsample_images(images: np.ndarray, size: int = 16) -> np.ndarray:
    """Reduce 64x64 faces to smaller grids so weights are learnable with small datasets."""
    n = images.shape[0]
    h, w = images.shape[1], images.shape[2]
    step_h, step_w = h // size, w // size
    blocks = images.reshape(n, size, step_h, size, step_w)
    return blocks.mean(axis=(2, 4))


def load_face_database(
    num_people: int = 10,
    feature_size: int = 16,
    seed: int = 42,
) -> FaceDataset:
    """
    Load clean, labeled face images from the Olivetti faces database.

    Constraint 1 — clean data: real grayscale face photos with person IDs,
    not opinions or unstructured text.
    Constraint 3 — defined parameters: fixed-size feature vectors from a face database.
    """
    data = fetch_olivetti_faces(shuffle=True, random_state=seed)
    images = data.images
    targets = data.target

    mask = targets < num_people
    cropped = _downsample_images(images[mask], size=feature_size)
    features = cropped.reshape(-1, feature_size * feature_size)
    labels = targets[mask]

    # Normalize pixel values to [0, 1] — standard preprocessing for clean numeric data
    features = features / 255.0

    person_names = [f"person_{i}" for i in range(num_people)]
    return FaceDataset(features=features, labels=labels, person_names=person_names)


def build_match_pairs(
    features: np.ndarray,
    labels: np.ndarray,
    num_pairs: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create training pairs for the measurable goal: match (1) or no-match (0).

    Constraint 2 — measurable goal: binary yes/no, not "what is beauty?"

    Uses |face_a - face_b| as input — the network learns which weighted
    feature differences signal a match (like the lecture's eye/nose/chin weights).
    """
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    pair_features: list[np.ndarray] = []
    pair_labels: list[int] = []

    for _ in range(num_pairs):
        if rng.random() < 0.5:
            label = int(rng.integers(0, labels.max() + 1))
            indices = np.where(labels == label)[0]
            if len(indices) < 2:
                continue
            i, j = rng.choice(indices, size=2, replace=False)
            pair = np.abs(features[i] - features[j])
            pair_features.append(pair)
            pair_labels.append(1)
        else:
            i, j = rng.integers(0, n, size=2)
            while labels[i] == labels[j]:
                j = int(rng.integers(0, n))
            pair = np.abs(features[i] - features[j])
            pair_features.append(pair)
            pair_labels.append(0)

    return np.array(pair_features), np.array(pair_labels)


def build_identity_classifier_data(
    dataset: FaceDataset,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Person identification: given a face, which person in the database?

    This is the lecture's goal — turn each face into a distinct mathematical model.
    """
    return train_test_split(
        dataset.features,
        dataset.labels,
        test_size=test_size,
        random_state=seed,
        stratify=dataset.labels,
    )


def train_face_matcher(
    num_people: int = 40,
    hidden_size: int = 128,
    epochs: int = 500,
    learning_rate: float = 0.3,
    seed: int = 42,
) -> tuple[NeuralNetwork, dict[str, float]]:
    """
    Train a neural network to answer: do these two faces belong to the same person?

    Uses backpropagation to learn weights — the process the lecture calls
    "deep learning" when applied to many layers.
    """
    dataset = load_face_database(num_people=num_people, seed=seed)
    x_train, x_test, y_train, y_test = build_identity_classifier_data(dataset, seed=seed)

    num_classes = num_people
    input_size = dataset.num_features

    print(f"Database: {dataset.num_faces} faces, {dataset.num_people} people")
    print(f"Feature dimensions (weights): {input_size} per face")
    print(f"Train: {x_train.shape[0]} samples  |  Test: {x_test.shape[0]} samples")
    print()

    network = NeuralNetwork(
        layer_sizes=[input_size, hidden_size, num_classes],
        learning_rate=learning_rate,
        seed=seed,
        output_activation="softmax",
    )
    print(f"Neural network: {network.layer_sizes}")
    print(f"Total learnable weights: {network.num_weights:,}")
    print()
    print("Training via backpropagation (supervised learning)...")

    network.train(x_train, y_train, epochs=epochs, verbose_every=max(epochs // 3, 1))

    metrics = network.evaluate(x_test, y_test)
    print()
    print(f"Test accuracy: {metrics['accuracy']:.1%}")

    return network, metrics


def train_binary_face_matcher(
    num_people: int = 40,
    num_pairs: int = 4000,
    hidden_size: int = 64,
    epochs: int = 500,
    learning_rate: float = 0.3,
    seed: int = 42,
) -> tuple[NeuralNetwork, dict[str, float]]:
    """
    Train on explicit match/no-match pairs — closest to the lecture's
    "does this face match? yes or no" framing.
    """
    dataset = load_face_database(num_people=num_people, seed=seed)
    pairs, labels = build_match_pairs(dataset.features, dataset.labels, num_pairs, seed=seed)

    x_train, x_test, y_train, y_test = train_test_split(
        pairs, labels, test_size=0.2, random_state=seed, stratify=labels
    )

    input_size = pairs.shape[1]  # absolute difference vector

    print("Binary face matching (yes/no — measurable goal)")
    print(f"  Training pairs: {len(y_train)}  |  Test pairs: {len(y_test)}")
    print(f"  Input size: {input_size} (|face_a - face_b| feature differences)")
    print()

    network = NeuralNetwork(
        layer_sizes=[input_size, hidden_size, 1],
        learning_rate=learning_rate,
        seed=seed,
        output_activation="sigmoid",
    )
    print(f"Neural network weights: {network.num_weights:,}")
    print()

    network.train(x_train, y_train, epochs=epochs, verbose_every=max(epochs // 4, 1))
    metrics = network.evaluate(x_test, y_test)
    print()
    print(f"Match/no-match test accuracy: {metrics['accuracy']:.1%}")

    return network, metrics


def demonstrate_edge_case(network: NeuralNetwork, dataset: FaceDataset) -> None:
    """
    Edge cases break supervised systems — the lecture's self-driving car example.

    A face outside the training distribution (e.g. heavily occluded or wrong
    orientation) may be misclassified because the model learned spurious patterns.
    """
    print()
    print("Edge case demo (model fragility):")
    print("  Training data: front-facing, well-lit Olivetti portraits.")
    print("  Edge case: invert pixels (negative image) — outside training distribution.")

    normal_idx = 0
    normal_face = dataset.features[normal_idx : normal_idx + 1]
    normal_label = dataset.labels[normal_idx]

    inverted_face = 1.0 - normal_face
    proba_normal = network.predict_proba(normal_face)
    proba_inverted = network.predict_proba(inverted_face)

    pred_normal = int(np.argmax(proba_normal))
    pred_inverted = int(np.argmax(proba_inverted))

    print(f"  Normal face (person {normal_label}): predicted person {pred_normal}")
    print(f"  Inverted face:                       predicted person {pred_inverted}")

    if pred_normal != pred_inverted:
        print("  -> Edge case changed the prediction. The black box failed on unseen patterns.")
    else:
        print("  -> Model happened to be robust here, but edge cases often break real systems.")
