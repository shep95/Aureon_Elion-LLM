"""Step 2 — automated labeling via teacher model + active learning."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from pipeline.config import (
    ACTIVE_LEARNING_REVIEW_RATE,
    CLEAN_DIR,
    LABELED_DIR,
    REVIEW_DIR,
    ensure_dirs,
)
from pipeline.queue import PipelineEvent, get_queue
from src.text_features import TextFeatureExtractor, load_clean_corpus

DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "mathematics": ("matrix", "linear", "equation", "theorem", "algebra", "proof"),
    "computer_science": (
        "algorithm",
        "neural",
        "network",
        "backpropagation",
        "complexity",
        "data structure",
        "programming",
    ),
    "physics": ("force", "motion", "newton", "energy", "quantum", "mechanics"),
    "biology": ("dna", "cell", "gene", "protein", "organism", "replication"),
    "engineering": ("control", "feedback", "system", "design", "circuit", "signal"),
    "history": ("century", "revolution", "empire", "war", "institution", "society"),
    "linguistics": ("grammar", "sanskrit", "language", "morphology", "phonological"),
    "literature": ("chapter", "character", "novel", "story", "narrative", "author"),
    "research": ("abstract", "study", "results", "method", "experiment", "analysis"),
}


def _keyword_teacher(text: str, title: str) -> tuple[str, float]:
    """Strong heuristic teacher — baseline when sklearn teacher is uncertain."""
    combined = f"{title} {text}".lower()
    scores: dict[str, float] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in combined)
        scores[domain] = hits / max(len(keywords), 1)
    best_domain = max(scores, key=scores.get)
    confidence = scores[best_domain]
    return best_domain, confidence


class TeacherLabeler:
    """
    Teacher-student labeling: a stronger sklearn model labels data for training.
    Falls back to keyword teacher for cold-start domains.
    """

    def __init__(self) -> None:
        self.vectorizer = TextFeatureExtractor(max_features=256)
        self.model = LogisticRegression(max_iter=1000)
        self.domains: list[str] = []
        self._fitted = False

    def _bootstrap_fit(self, rows: list[dict]) -> None:
        texts = [f"{r['title']} {r['text']}" for r in rows]
        pseudo_labels = []
        for row in rows:
            meta_domain = row.get("metadata", {}).get("domain")
            if meta_domain and meta_domain in DOMAIN_KEYWORDS:
                pseudo_labels.append(meta_domain)
            else:
                domain, _ = _keyword_teacher(row["text"], row["title"])
                pseudo_labels.append(domain)

        self.domains = sorted(set(pseudo_labels))
        x = self.vectorizer.fit_transform(texts)
        y = np.array([self.domains.index(label) for label in pseudo_labels])
        if len(set(y.tolist())) >= 2:
            self.model.fit(x, y)
            self._fitted = True

    def label(self, rows: list[dict]) -> list[dict]:
        if not rows:
            return []

        if not self._fitted:
            self._bootstrap_fit(rows)

        labeled: list[dict] = []
        texts = [f"{r['title']} {r['text']}" for r in rows]

        if self._fitted:
            x = self.vectorizer.transform(texts)
            probabilities = self.model.predict_proba(x)
            predictions = np.argmax(probabilities, axis=1)
            confidences = probabilities.max(axis=1)
        else:
            predictions = []
            confidences = []
            for row in rows:
                domain, conf = _keyword_teacher(row["text"], row["title"])
                predictions.append(self.domains.index(domain) if domain in self.domains else 0)
                confidences.append(conf)
            predictions = np.array(predictions)
            confidences = np.array(confidences)

        for row, pred_idx, confidence in zip(rows, predictions, confidences):
            domain = self.domains[int(pred_idx)] if self.domains else "research"
            keyword_domain, keyword_conf = _keyword_teacher(row["text"], row["title"])

            # Ensemble teacher + keyword for verifiability
            if keyword_conf > float(confidence):
                domain = keyword_domain
                confidence = keyword_conf

            labeled.append(
                {
                    **row,
                    "label": domain,
                    "label_confidence": round(float(confidence), 4),
                    "label_source": "teacher_model",
                }
            )
        return labeled


def active_learning_split(
    labeled_rows: list[dict],
    review_rate: float = ACTIVE_LEARNING_REVIEW_RATE,
) -> tuple[list[dict], list[dict]]:
    """
    Flag uncertain samples for human review — reduces manual labeling to ~5-10%.
    Skipped when the corpus is too small to spare examples.
    """
    if not labeled_rows:
        return [], []
    if len(labeled_rows) <= 12:
        return labeled_rows, []

    uncertainties = []
    for row in labeled_rows:
        confidence = float(row.get("label_confidence", 0.5))
        uncertainties.append(1.0 - confidence)

    threshold = np.quantile(uncertainties, 1.0 - review_rate)
    auto_labeled: list[dict] = []
    for_review: list[dict] = []

    for row, uncertainty in zip(labeled_rows, uncertainties):
        if uncertainty >= threshold:
            review_row = {**row, "review_reason": "high_uncertainty", "uncertainty": round(uncertainty, 4)}
            for_review.append(review_row)
        else:
            auto_labeled.append(row)

    return auto_labeled, for_review


def run_step2() -> dict[str, Any]:
    ensure_dirs()
    queue = get_queue()

    corpus_path = CLEAN_DIR / "corpus.jsonl"
    rows = load_clean_corpus(corpus_path)
    if not rows:
        return {
            "step": 2,
            "name": "automated_labeling",
            "status": "skipped",
            "reason": "no clean corpus — run step 1 first",
        }

    labeler = TeacherLabeler()
    labeled = labeler.label(rows)
    auto_labeled, for_review = active_learning_split(labeled)

    labeled_path = LABELED_DIR / "labeled.jsonl"
    with labeled_path.open("w", encoding="utf-8") as handle:
        for row in auto_labeled:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    review_path = REVIEW_DIR / "pending_review.jsonl"
    with review_path.open("w", encoding="utf-8") as handle:
        for row in for_review:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    event = PipelineEvent(
        step="2_labeling",
        event_type="labels_ready",
        payload={
            "labeled_count": len(auto_labeled),
            "review_count": len(for_review),
            "labeled_path": str(labeled_path),
        },
    )
    queue.publish("pipeline.data.labeled", event)

    return {
        "step": 2,
        "name": "automated_labeling",
        "input_documents": len(rows),
        "auto_labeled": len(auto_labeled),
        "flagged_for_human_review": len(for_review),
        "review_rate": round(len(for_review) / max(len(rows), 1), 4),
        "labeled_path": str(labeled_path),
        "review_path": str(review_path),
        "domains": sorted({row["label"] for row in labeled}),
    }
