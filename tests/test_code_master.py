"""Tests for doctorate-level code master pipeline."""

from __future__ import annotations

from brain.code_master import benchmark_humaneval, generate_master_code, get_code_bank


def test_code_bank_loads_problems():
    bank = get_code_bank()
    assert len(bank.problems) >= 1100


def test_retrieval_verified_add_two_numbers():
    result = generate_master_code(
        "write a python function to add two numbers",
        predict_fn=lambda _q: None,
    )
    assert result["answer"]
    assert "def " in result["answer"]
    assert result["code_eval"]["syntax_valid"] is True
    assert result["method"] in ("bootstrap_seed", "retrieval_verified", "retrieval_fallback", "neural_synthesis")


def test_retrieval_reverse_string():
    result = generate_master_code(
        "write a python function to reverse a string",
        predict_fn=lambda _q: None,
    )
    assert result["code_eval"]["syntax_valid"] is True
    assert "reverse" in result["answer"].lower()


def test_humaneval_benchmark_sample():
    report = benchmark_humaneval(limit=10, use_retrieval=True)
    assert report["total"] == 10
    assert report["pass_rate"] >= 0.8
