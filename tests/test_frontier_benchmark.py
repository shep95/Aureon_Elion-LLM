"""Frontier benchmark harness tests."""

from __future__ import annotations

from brain.deterministic_qa import try_arithmetic_answer
from brain.frontier_benchmark import run_frontier_benchmark


def test_deterministic_cases_in_benchmark():
    from brain.frontier_benchmark import FRONTIER_CASES, _score_case
    from brain.deterministic_qa import try_arithmetic_answer

    for case in FRONTIER_CASES:
        if case["kind"] != "deterministic":
            continue
        det = try_arithmetic_answer(case["question"])
        assert det is not None
        scored = _score_case(case, det["answer"], abstained=False)
        assert scored["passed"]


def test_arithmetic_cases_direct():
    assert try_arithmetic_answer("What is 2+2")["answer"] == "4"
