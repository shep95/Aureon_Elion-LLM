"""Step 3 — model registry and automated training loops."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.security import load_json_file_bounded
from pipeline.config import MODELS_DIR, REGISTRY_DIR, ensure_dirs


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelRegistry:
    """
    Lightweight MLflow-style registry using JSON files.
    Tracks runs, metrics, and the current production model.
    """

    def __init__(self) -> None:
        ensure_dirs()
        self.registry_path = REGISTRY_DIR / "registry.json"
        self.production_path = REGISTRY_DIR / "production.json"
        if not self.registry_path.exists():
            self._save_registry({"runs": []})

    def _load_registry(self) -> dict:
        return load_json_file_bounded(self.registry_path)

    def _save_registry(self, payload: dict) -> None:
        self.registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def log_run(
        self,
        run_id: str,
        metrics: dict[str, float],
        artifact_path: str,
        params: dict[str, Any],
        status: str = "completed",
    ) -> dict:
        registry = self._load_registry()
        run = {
            "run_id": run_id,
            "created_at": _utcnow(),
            "metrics": metrics,
            "artifact_path": artifact_path,
            "params": params,
            "status": status,
        }
        registry["runs"].append(run)
        self._save_registry(registry)
        return run

    @staticmethod
    def _metric_score(metrics: dict[str, float], metric: str) -> float:
        if metric in metrics:
            return float(metrics[metric])
        if metric == "val_accuracy" and "train_accuracy" in metrics:
            return float(metrics["train_accuracy"])
        if metric == "train_accuracy" and "val_accuracy" in metrics:
            return float(metrics["val_accuracy"])
        return 0.0

    def get_best_run(self, metric: str = "val_accuracy") -> dict | None:
        registry = self._load_registry()
        runs = [r for r in registry["runs"] if r.get("status") == "completed"]
        if not runs:
            return None
        return max(runs, key=lambda r: self._metric_score(r.get("metrics", {}), metric))

    def get_production(self) -> dict | None:
        if not self.production_path.exists():
            return None
        return load_json_file_bounded(self.production_path)

    def promote(self, run: dict) -> dict:
        payload = {
            "promoted_at": _utcnow(),
            "run_id": run["run_id"],
            "metrics": run["metrics"],
            "artifact_path": run["artifact_path"],
        }
        self.production_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def should_promote(self, new_metrics: dict[str, float], metric: str = "val_accuracy") -> bool:
        current = self.get_production()
        if not current:
            return True
        current_score = self._metric_score(current.get("metrics", {}), metric)
        new_score = self._metric_score(new_metrics, metric)
        return new_score > current_score
