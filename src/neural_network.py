"""
Neural network with backpropagation — the core of supervised machine learning.

Traditional programming:  you write the algorithm, give input, get output.
  output = algorithm(input)   e.g.  1 + 1 = 2

Supervised machine learning: you control BOTH input AND the correct output (label).
  The computer adjusts internal weights until predictions match your labels.
  This is backpropagation — what the industry rebrands as "deep learning."
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(output: np.ndarray) -> np.ndarray:
    return output * (1.0 - output)


@dataclass
class TrainingMetrics:
    epoch: int
    loss: float
    accuracy: float


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


@dataclass
class NeuralNetwork:
    """
    A weighting system (neural network) that learns by backpropagation.

    Each connection between layers has a weight.  The lecture's facial-recognition
    example uses weights for features like eyes, nose, and chin — here each pixel
    or feature dimension plays that role.  We know these features matter; we do
    not know how much.  Training discovers the weights.
    """

    layer_sizes: list[int]
    learning_rate: float = 0.1
    seed: int = 42
    output_activation: str = "sigmoid"  # "sigmoid" for binary, "softmax" for multi-class
    weights: list[np.ndarray] = field(default_factory=list)
    biases: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.weights:
            rng = np.random.default_rng(self.seed)
            for in_size, out_size in zip(self.layer_sizes, self.layer_sizes[1:]):
                # Xavier-style init helps gradients flow during backpropagation
                scale = np.sqrt(2.0 / in_size)
                self.weights.append(rng.normal(0, scale, (in_size, out_size)))
                self.biases.append(np.zeros((1, out_size)))

    @property
    def num_weights(self) -> int:
        return sum(w.size + b.size for w, b in zip(self.weights, self.biases))

    def _activate_hidden(self, z: np.ndarray) -> np.ndarray:
        return relu(z)

    def _activate_output(self, z: np.ndarray) -> np.ndarray:
        if self.output_activation == "softmax":
            return softmax(z)
        return sigmoid(z)

    def forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Run input through the network; return activations and pre-activations."""
        activations = [x]
        pre_activations: list[np.ndarray] = []

        current = x
        num_layers = len(self.weights)
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = current @ weight + bias
            pre_activations.append(z)
            if i == num_layers - 1:
                current = self._activate_output(z)
            else:
                current = self._activate_hidden(z)
            activations.append(current)

        return activations, pre_activations

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        activations, _ = self.forward(x)
        return activations[-1]

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(x)
        if self.output_activation == "softmax":
            return np.argmax(proba, axis=1)
        return (proba >= threshold).astype(int).ravel()

    def _one_hot(self, y: np.ndarray, num_classes: int) -> np.ndarray:
        encoded = np.zeros((y.shape[0], num_classes))
        encoded[np.arange(y.shape[0]), y.astype(int)] = 1.0
        return encoded

    def _loss_and_output_delta(
        self, output: np.ndarray, targets: np.ndarray
    ) -> tuple[float, np.ndarray]:
        eps = 1e-9
        if self.output_activation == "softmax":
            loss = float(-np.mean(np.sum(targets * np.log(output + eps), axis=1)))
            delta = (output - targets) / output.shape[0]
        else:
            error = output - targets
            loss = float(np.mean(error**2))
            delta = (error * sigmoid_derivative(output)) / output.shape[0]
        return loss, delta

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 500,
        batch_size: int | None = None,
        verbose_every: int = 100,
    ) -> list[TrainingMetrics]:
        """
        Supervised training loop.

        You provide:
          - x: controlled inputs  (face images / feature vectors)
          - y: controlled outputs (yes/no labels — measurable goals)

        Backpropagation adjusts weights until the network matches your labels.
        """
        num_classes = self.layer_sizes[-1]
        if num_classes == 1:
            targets = y.reshape(-1, 1).astype(float)
        else:
            targets = self._one_hot(y, num_classes)

        batch_size = batch_size or min(32, x.shape[0])
        history: list[TrainingMetrics] = []
        rng = np.random.default_rng(self.seed)

        for epoch in range(1, epochs + 1):
            indices = rng.permutation(x.shape[0])
            epoch_loss = 0.0
            correct = 0
            batches = 0

            for start in range(0, x.shape[0], batch_size):
                batch_idx = indices[start : start + batch_size]
                batch_x = x[batch_idx]
                batch_y = targets[batch_idx]
                batch_labels = y[batch_idx]

                activations, pre_activations = self.forward(batch_x)
                output = activations[-1]

                loss, delta = self._loss_and_output_delta(output, batch_y)
                epoch_loss += loss
                batches += 1

                if num_classes == 1:
                    predictions = (output >= 0.5).astype(int).ravel()
                    correct += int(np.sum(predictions == batch_labels.astype(int)))
                else:
                    correct += int(np.sum(np.argmax(output, axis=1) == batch_labels.astype(int)))

                # Backpropagation — propagate error backward to update weights
                deltas = [delta]

                for layer in reversed(range(len(self.weights) - 1)):
                    delta = deltas[-1] @ self.weights[layer + 1].T
                    delta *= relu_derivative(pre_activations[layer])
                    deltas.append(delta)
                deltas.reverse()

                for layer in range(len(self.weights)):
                    grad_w = activations[layer].T @ deltas[layer]
                    grad_b = np.sum(deltas[layer], axis=0, keepdims=True)
                    self.weights[layer] -= self.learning_rate * grad_w
                    self.biases[layer] -= self.learning_rate * grad_b

            accuracy = correct / x.shape[0]
            metrics = TrainingMetrics(epoch=epoch, loss=epoch_loss / batches, accuracy=accuracy)
            history.append(metrics)

            if verbose_every and epoch % verbose_every == 0:
                print(
                    f"  epoch {epoch:4d}  loss={metrics.loss:.4f}  accuracy={metrics.accuracy:.1%}"
                )

        return history

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        proba = self.predict_proba(x)
        num_classes = self.layer_sizes[-1]

        if num_classes == 1:
            predictions = (proba >= 0.5).astype(int).ravel()
            accuracy = float(np.mean(predictions == y.astype(int)))
        else:
            predictions = np.argmax(proba, axis=1)
            accuracy = float(np.mean(predictions == y.astype(int)))

        return {"accuracy": accuracy, "num_samples": float(x.shape[0])}

    def save(self, path: str | Path) -> None:
        payload = {
            "layer_sizes": self.layer_sizes,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "output_activation": self.output_activation,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> NeuralNetwork:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        net = cls(
            layer_sizes=payload["layer_sizes"],
            learning_rate=payload["learning_rate"],
            seed=payload["seed"],
            output_activation=payload.get("output_activation", "sigmoid"),
        )
        net.weights = [np.array(w) for w in payload["weights"]]
        net.biases = [np.array(b) for b in payload["biases"]]
        return net


def compare_traditional_vs_ml() -> None:
    """Illustrate the lecture's contrast between traditional code and ML."""

    def traditional_add(a: float, b: float) -> float:
        return a + b

    print("Traditional programming:")
    print(f"  algorithm: a + b")
    print(f"  input:  1, 1")
    print(f"  output: {traditional_add(1, 1)}")
    print()
    print("Supervised machine learning:")
    print("  You do NOT write the algorithm.")
    print("  You provide inputs AND the correct outputs (labels).")
    print("  The computer discovers the weights via backpropagation.")
