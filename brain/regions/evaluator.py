"""Evaluator region — domain benchmarks and quality gates."""

from __future__ import annotations

from sqlalchemy.orm import Session

from brain.base import AgentContext, AgentResult, MicroAgentBase
from brain.code_evaluator import benchmark_code_pass_rate
from brain.grades import evaluate_code_grade_gates, evaluate_grade_gates, is_code_micro
from db.models import BenchmarkResult
from pipeline.step4_evaluation.benchmarks import (
    _load_production_model,
    run_consistency_benchmark,
    run_reasoning_benchmark,
    run_verification_benchmark,
)


class EvaluatorAgent(MicroAgentBase):
    region = "evaluator"

    def run(self, session: Session, ctx: AgentContext) -> AgentResult:
        if is_code_micro(ctx.micro_subdomain_slug):
            code_bench = benchmark_code_pass_rate(limit=10, use_predict=True)
            if ctx.grade:
                gates = evaluate_code_grade_gates(ctx.grade, code_bench["score"])
            else:
                gates = {
                    "gates": {"code_pass_rate": code_bench["passed"]},
                    "all_passed": code_bench["passed"],
                    "pass_rate": code_bench["score"],
                }
            session.add(
                BenchmarkResult(
                    domain_id=ctx.domain_id,
                    benchmark_type=f"code_pass_rate_{ctx.grade_slug or 'default'}",
                    score=code_bench["score"],
                    passed=code_bench["passed"],
                    details={"cases": code_bench.get("cases", []), "grade": ctx.grade_slug},
                )
            )
            return AgentResult(
                region=self.region,
                status="completed" if gates["all_passed"] else "warning",
                metrics={
                    "benchmarks": {"code_pass_rate": code_bench},
                    **gates,
                },
            )

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
        if ctx.grade:
            gates = evaluate_grade_gates(ctx.grade, benchmarks)
        else:
            from pipeline.step4_evaluation.benchmarks import evaluate_gates

            gates = evaluate_gates(benchmarks)

        for bench_type, result in benchmarks.items():
            session.add(
                BenchmarkResult(
                    domain_id=ctx.domain_id,
                    benchmark_type=f"{bench_type}_{ctx.grade_slug or 'default'}",
                    score=result["score"],
                    passed=result["passed"],
                    details={"cases": result.get("cases", []), "grade": ctx.grade_slug},
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
