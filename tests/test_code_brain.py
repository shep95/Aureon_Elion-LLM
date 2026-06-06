"""Tests for code brain region — collector, evaluator, routing, graduation."""

from __future__ import annotations

from app.chat_service import is_code_question
from brain.code_evaluator import (
    benchmark_code_pass_rate,
    check_syntax,
    evaluate_code_response,
    extract_python_code,
)
from brain.grades import CODE_GRADUATION_THRESHOLDS, evaluate_code_grade_gates, get_grade, is_code_micro
from brain.regions.code_collector import CodeCollector


def test_code_collector_loads_humaneval_and_mbpp():
    docs = CodeCollector().collect(limit=2000)
    assert len(docs) == 1138
    sources = {d.source for d in docs}
    assert "humaneval" in sources
    assert "mbpp" in sources
    he = next(d for d in docs if d.source == "humaneval")
    assert he.metadata.get("code_area") == "code_generation"
    assert he.metadata.get("micro_subdomain") == "python_functions"
    assert he.metadata.get("has_tests") is True
    assert "test" in he.metadata


def test_check_syntax_valid_and_invalid():
    assert check_syntax("def f():\n    return 1")["valid"] is True
    bad = check_syntax("def f(\n    return 1")
    assert bad["valid"] is False
    assert bad["error"]


def test_evaluate_code_response_syntax_only():
    result = evaluate_code_response("def add(a, b): return a + b")
    assert result["syntax_valid"] is True
    assert result["score"] == 0.5
    assert result["passed_tests"] is None


def test_evaluate_code_response_with_test():
    code = "def add(a, b):\n    return a + b"
    test = "assert add(2, 3) == 5"
    result = evaluate_code_response(code, test)
    assert result["syntax_valid"] is True
    assert result["passed_tests"] is True
    assert result["score"] == 1.0


def test_extract_python_code_from_answer():
    raw = "think use def keyword therefore answer def add(a, b): return a + b"
    assert extract_python_code(raw).startswith("def add")


def test_is_code_question_triggers():
    assert is_code_question("write a python function to sort a list")
    assert is_code_question("implement binary search")
    assert not is_code_question("what is the capital of france")


def test_code_graduation_thresholds():
    assert CODE_GRADUATION_THRESHOLDS["doctorate"] == 0.90
    grade = get_grade("high_school")
    assert grade is not None
    gates = evaluate_code_grade_gates(grade, 0.70)
    assert gates["all_passed"] is True
    gates_fail = evaluate_code_grade_gates(grade, 0.50)
    assert gates_fail["all_passed"] is False


def test_is_code_micro():
    assert is_code_micro("python_functions")
    assert not is_code_micro("algorithms_and_data_structures")


def test_benchmark_reference_solutions_pass():
    result = benchmark_code_pass_rate(limit=3, use_predict=False)
    assert result["total"] == 3
    assert result["score"] == 1.0


def test_bootstrap_code_keeps_lowercase_def():
    from brain.predict_engine import _bootstrap_answer

    ans = _bootstrap_answer("write a python function to add two numbers")
    assert ans == "def add(a, b): return a + b"
    assert evaluate_code_response(ans)["syntax_valid"] is True
