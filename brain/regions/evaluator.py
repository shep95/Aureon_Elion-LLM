"""Evaluator region — domain benchmarks and quality gates."""

from __future__ import annotations

from sqlalchemy.orm import Session

from brain.base import AgentContext, AgentResult, MicroAgentBase
from db.models import BenchmarkResult
from pipeline.step4_evaluation.benchmarks import (
    _load_production_model,
    evaluate_gates,
    run_consistency_benchmark,
    run_reasoning_benchmark,
    run_verification_benchmark,
)


class EvaluatorAgent(MicroAgentBase):
    region = "evaluator"

    def run(self, session: Session, ctx: AgentContext) -> AgentResult:
        loaded = _load_production_model()
        if not loaded:
            return AgentResult(
                region=self.region,
                status="skipped",
                metrics={"reason": "no production model"},
            )

        network, labels, extractor = loaded
        benchmarks = {
            "reasoning": run_reasoning_benchmark(network, labels, extractor),
            "consistency": run_consistency_benchmark(network, labels, extractor),
            "verification": run_verification_benchmark(network, labels, extractor),
        }
        gates = evaluate_gates(benchmarks)

        for bench_type, result in benchmarks.items():
            session.add(
                BenchmarkResult(
                    domain_id=ctx.domain_id,
                    benchmark_type=bench_type,
                    score=result["score"],
                    passed=result["passed"],
                    details={"cases": result.get("cases", [])},
                )
            )

        return AgentResult(
            region=self.region,
            status="completed" if gates["all_passed"] else "warning",
            metrics={
                "benchmarks": {k: {"score": v["score"], "passed": v["passed"]} for k, v in benchmarks.items()},
                **gates,
            },
        )
