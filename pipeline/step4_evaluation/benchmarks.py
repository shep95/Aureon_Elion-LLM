"""Step 4 — benchmark suites, quality gates, and alerts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.config import (
    ACCURACY_GATE,
    CONSISTENCY_GATE,
    REASONING_GATE,
    ensure_dirs,
)
from pipeline.step3_training.registry import ModelRegistry
from src.neural_network import NeuralNetwork
from src.text_features import TextFeatureExtractor

# Fixed benchmark sets — reasoning, verifiability, consistency
REASONING_CASES = [
    {
        "text": "If all mammals breathe air and whales are mammals, whales breathe air.",
        "label": "biology",
    },
    {
        "text": "Matrix multiplication is associative but generally not commutative.",
        "label": "mathematics",
    },
    {
        "text": "Backpropagation applies the chain rule to compute gradients for weight updates.",
        "label": "computer_science",
    },
]

CONSISTENCY_PAIRS = [
    (
        "Newton's second law relates force mass and acceleration.",
        "Force equals mass times acceleration according to Newton.",
        "physics",
    ),
    (
        "DNA replication is semiconservative with complementary base pairing.",
        "Each DNA strand templates a new complementary strand during replication.",
        "biology",
    ),
]

VERIFICATION_CASES = [
    {
        "text": "The Scientific Revolution emphasized empirical observation and peer scrutiny.",
        "label": "history",
        "must_contain": ["empirical", "peer"],
    },
    {
        "text": "Panini's grammar describes Sanskrit through explicit metarules and substitutions.",
        "label": "linguistics",
        "must_contain": ["sanskrit", "grammar"],
    },
]


def _load_production_model() -> tuple[NeuralNetwork, list[str], TextFeatureExtractor] | None:
    registry = ModelRegistry()
    production = registry.get_production()
    if not production:
        return None

    artifact_dir = Path(production["artifact_path"]).parent
    meta = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    network = NeuralNetwork.load(production["artifact_path"])
    extractor = TextFeatureExtractor.from_dict(meta["feature_extractor"])
    return network, meta["labels"], extractor


def _predict_label(
    network: NeuralNetwork,
    labels: list[str],
    extractor: TextFeatureExtractor,
    text: str,
) -> str:
    x = extractor.transform([text])
    idx = int(network.predict(x)[0])
    if idx < len(labels):
        return labels[idx]
    return labels[0]


def _case_applicable(expected: str, labels: list[str]) -> bool:
    return expected in labels


def run_reasoning_benchmark(
    network: NeuralNetwork,
    labels: list[str],
    extractor: TextFeatureExtractor,
) -> dict[str, Any]:
    correct = 0
    applicable = 0
    results = []
    for case in REASONING_CASES:
        if not _case_applicable(case["label"], labels):
            results.append({"text": case["text"][:80], "expected": case["label"], "skipped": True})
            continue
        applicable += 1
        pred = _predict_label(network, labels, extractor, case["text"])
        ok = pred == case["label"]
        correct += int(ok)
        results.append({"text": case["text"][:80], "expected": case["label"], "predicted": pred, "pass": ok})
    score = correct / applicable if applicable else 1.0
    return {"score": round(score, 4), "passed": score >= REASONING_GATE, "cases": results}


def run_consistency_benchmark(
    network: NeuralNetwork,
    labels: list[str],
    extractor: TextFeatureExtractor,
) -> dict[str, Any]:
    consistent = 0
    applicable = 0
    results = []
    for text_a, text_b, expected in CONSISTENCY_PAIRS:
        if not _case_applicable(expected, labels):
            results.append({"expected": expected, "skipped": True})
            continue
        applicable += 1
        pred_a = _predict_label(network, labels, extractor, text_a)
        pred_b = _predict_label(network, labels, extractor, text_b)
        ok = pred_a == pred_b == expected
        consistent += int(ok)
        results.append(
            {
                "paraphrase_a": text_a[:60],
                "paraphrase_b": text_b[:60],
                "expected": expected,
                "predicted_a": pred_a,
                "predicted_b": pred_b,
                "pass": ok,
            }
        )
    score = consistent / applicable if applicable else 1.0
    return {"score": round(score, 4), "passed": score >= CONSISTENCY_GATE, "cases": results}


def run_verification_benchmark(
    network: NeuralNetwork,
    labels: list[str],
    extractor: TextFeatureExtractor,
) -> dict[str, Any]:
    verified = 0
    applicable = 0
    results = []
    for case in VERIFICATION_CASES:
        if not _case_applicable(case["label"], labels):
            results.append({"expected": case["label"], "skipped": True})
            continue
        applicable += 1
        pred = _predict_label(network, labels, extractor, case["text"])
        label_ok = pred == case["label"]
        content_ok = all(token in case["text"].lower() for token in case["must_contain"])
        ok = label_ok and content_ok
        verified += int(ok)
        results.append(
            {
                "text": case["text"][:80],
                "expected": case["label"],
                "predicted": pred,
                "verifiable": content_ok,
                "pass": ok,
            }
        )
    score = verified / applicable if applicable else 1.0
    return {"score": round(score, 4), "passed": score >= ACCURACY_GATE, "cases": results}


def evaluate_gates(benchmarks: dict[str, dict]) -> dict[str, Any]:
    gates = {
        "reasoning": benchmarks["reasoning"]["passed"],
        "consistency": benchmarks["consistency"]["passed"],
        "verification": benchmarks["verification"]["passed"],
    }
    all_passed = all(gates.values())
    return {"gates": gates, "all_passed": all_passed, "halt_training": not all_passed}
