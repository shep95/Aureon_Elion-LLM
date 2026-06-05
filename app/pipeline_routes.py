"""Pipeline API routes."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query

from pipeline.orchestrator import run_full_pipeline
from pipeline.step1_collection.runner import run_step1
from pipeline.step2_labeling.runner import run_step2
from pipeline.step3_training.runner import run_step3
from pipeline.step4_evaluation.runner import run_step4
from pipeline.step5_rlhf.runner import run_step5


def run_pipeline_step(step: int, epochs: int = 300) -> dict[str, Any]:
    runners = {
        1: lambda: run_step1(arxiv_limit=5, gutenberg_limit=0),
        2: lambda: run_step2(),
        3: lambda: run_step3(epochs=epochs),
        4: lambda: run_step4(),
        5: lambda: run_step5(epochs=min(epochs, 400)),
    }
    if step not in runners:
        raise HTTPException(status_code=400, detail="step must be 1-5")
    try:
        return runners[step]()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def run_pipeline_all(epochs: int = 300) -> dict[str, Any]:
    try:
        return run_full_pipeline(epochs=epochs, arxiv_limit=5, gutenberg_limit=0)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
