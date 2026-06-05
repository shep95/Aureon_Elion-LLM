"""Run the full 5-step automated training pipeline in order."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.config import REGISTRY_DIR, ensure_dirs
from pipeline.step1_collection.runner import run_step1
from pipeline.step2_labeling.runner import run_step2
from pipeline.step3_training.runner import run_step3
from pipeline.step4_evaluation.runner import run_step4
from pipeline.step5_rlhf.runner import run_step5


def run_full_pipeline(
    epochs: int = 300,
    arxiv_limit: int = 5,
    gutenberg_limit: int = 1,
) -> dict[str, Any]:
    ensure_dirs()
    results: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": [],
    }

    step1 = run_step1(arxiv_limit=arxiv_limit, gutenberg_limit=gutenberg_limit)
    results["steps"].append(step1)

    step2 = run_step2()
    results["steps"].append(step2)
    if step2.get("status") == "skipped":
        results["status"] = "halted"
        results["halt_reason"] = step2.get("reason")
        return _finalize(results)

    step3 = run_step3(epochs=epochs)
    results["steps"].append(step3)
    if step3.get("status") == "skipped":
        results["status"] = "halted"
        results["halt_reason"] = step3.get("reason")
        return _finalize(results)

    step4 = run_step4()
    results["steps"].append(step4)
    halted = bool(step4.get("halt_training"))

    step5 = run_step5(epochs=min(epochs, 400))
    results["steps"].append(step5)

    if halted:
        results["status"] = "completed_with_warnings"
        results["warnings"] = ["Step 4 benchmark gates failed — alert fired, pipeline continued to RLHF"]
    else:
        results["status"] = "completed"
    return _finalize(results)


def _finalize(results: dict[str, Any]) -> dict[str, Any]:
    results["finished_at"] = datetime.now(timezone.utc).isoformat()
    out_path = REGISTRY_DIR / "latest_pipeline_run.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
