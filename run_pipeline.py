#!/usr/bin/env python3
"""CLI for the 5-step automated training pipeline."""

from __future__ import annotations

import argparse
import json

from pipeline.orchestrator import run_full_pipeline
from pipeline.step1_collection.runner import run_step1
from pipeline.step2_labeling.runner import run_step2
from pipeline.step3_training.runner import run_step3
from pipeline.step4_evaluation.runner import run_step4
from pipeline.step5_rlhf.runner import run_step5

STEP_RUNNERS = {
    1: lambda args: run_step1(arxiv_limit=args.arxiv_limit, gutenberg_limit=args.gutenberg_limit),
    2: lambda _args: run_step2(),
    3: lambda args: run_step3(epochs=args.epochs),
    4: lambda _args: run_step4(),
    5: lambda args: run_step5(epochs=args.epochs),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="SOLIA automated training pipeline")
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run a single step (default: run all steps in order)",
    )
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs for steps 3 and 5")
    parser.add_argument("--arxiv-limit", type=int, default=5, dest="arxiv_limit")
    parser.add_argument("--gutenberg-limit", type=int, default=1, dest="gutenberg_limit")
    args = parser.parse_args()

    if args.step:
        result = STEP_RUNNERS[args.step](args)
    else:
        result = run_full_pipeline(
            epochs=args.epochs,
            arxiv_limit=args.arxiv_limit,
            gutenberg_limit=args.gutenberg_limit,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
