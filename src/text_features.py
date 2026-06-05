"""Text feature extraction for the training pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

WORD_RE = re.compile(r"[a-zA-Z]{2,}")


def load_clean_corpus(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class TextFeatureExtractor:
    """TF-IDF bag-of-words features with fixed vocabulary persistence."""

    def __init__(self, max_features: int = 512) -> None:
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        self._fitted = False

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        matrix = self.vectorizer.fit_transform(texts)
        self._fitted = True
        return matrix.toarray()

    def transform(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Vectorizer not fitted")
        return self.vectorizer.transform(texts).toarray()

    def to_dict(self) -> dict:
        vocabulary = {k: int(v) for k, v in self.vectorizer.vocabulary_.items()}
        return {
            "max_features": self.max_features,
            "vocabulary": vocabulary,
            "idf": [float(x) for x in self.vectorizer.idf_.tolist()],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> TextFeatureExtractor:
        obj = cls(max_features=payload["max_features"])
        obj.vectorizer.vocabulary_ = {k: int(v) for k, v in payload["vocabulary"].items()}
        obj.vectorizer.idf_ = np.array(payload["idf"], dtype=float)
        obj.vectorizer.fixed_vocabulary_ = True
        obj._fitted = True
        return obj
