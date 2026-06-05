"""Step 4 runner — benchmarks, gates, alerts."""

from __future__ import annotations

import json
from typing import Any

from pipeline.config import BENCHMARKS_DIR, ensure_dirs
from pipeline.queue import PipelineEvent, get_queue
from pipeline.step3_training.registry import ModelRegistry
from pipeline.step4_evaluation.alerts import fire_alert
from pipeline.step4_evaluation.benchmarks import (
    _load_production_model,
    evaluate_gates,
    run_consistency_benchmark,
    run_reasoning_benchmark,
    run_verification_benchmark,
)


def run_step4() -> dict[str, Any]:
    ensure_dirs()
    queue = get_queue()
    registry = ModelRegistry()
    production = registry.get_production()

    if not production:
        return {
            "step": 4,
            "name": "automated_evaluation",
            "status": "skipped",
            "reason": "no production model — run step 3 first",
        }

    loaded = _load_production_model()
    if not loaded:
        return {
            "step": 4,
            "name": "automated_evaluation",
            "status": "failed",
            "reason": "production model artifacts missing",
        }

    network, labels, extractor = loaded
    benchmarks = {
        "reasoning": run_reasoning_benchmark(network, labels, extractor),
        "consistency": run_consistency_benchmark(network, labels, extractor),
        "verification": run_verification_benchmark(network, labels, extractor),
    }
    gate_result = evaluate_gates(benchmarks)

    report = {
        "step": 4,
        "name": "automated_evaluation",
        "production_run_id": production["run_id"],
        "production_metrics": production.get("metrics", {}),
        "benchmarks": benchmarks,
        **gate_result,
    }

    report_path = BENCHMARKS_DIR / "latest_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if gate_result["halt_training"]:
        alert = fire_alert(
            title="Training pipeline halted — benchmark regression",
            details={
                "halt_training": True,
                "gates": gate_result["gates"],
                "production_run_id": production["run_id"],
            },
        )
        report["alert"] = alert

    event = PipelineEvent(
        step="4_evaluation",
        event_type="evaluation_complete",
        payload={
            "all_passed": gate_result["all_passed"],
            "halt_training": gate_result["halt_training"],
        },
    )
    queue.publish("pipeline.evaluation.complete", event)

    return report
