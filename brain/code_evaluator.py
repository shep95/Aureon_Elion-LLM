"""Code evaluator — syntax check and safe execution with timeout."""

from __future__ import annotations

import ast
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from pipeline.config import ROOT

HUMANEVAL_PATH = ROOT / "data" / "code" / "humaneval-python.jsonl"
MBPP_PATH = ROOT / "data" / "code" / "mbpp.jsonl"


def extract_python_code(text: str) -> str:
    """Pull executable Python from model output — preserves leading imports."""
    if not text:
        return ""
    lowered = text.lower()
    if " answer " in lowered:
        idx = lowered.rfind(" answer ")
        text = text[idx + len(" answer ") :]
    markers = ("from ", "import ", "def ", "class ", "for ", "while ")
    positions = [text.find(m) for m in markers if text.find(m) != -1]
    if positions:
        return text[min(positions) :].strip()
    return text.strip()


def check_syntax(code: str) -> dict[str, Any]:
    """Verify Python syntax without executing."""
    try:
        ast.parse(code)
        return {"valid": True, "error": None}
    except SyntaxError as exc:
        return {"valid": False, "error": str(exc)}


def run_with_timeout(code: str, test: str, timeout: int = 3) -> dict[str, Any]:
    """Execute code + test in isolated subprocess with hard timeout."""
    full_code = code + "\n\n" + test
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as handle:
        handle.write(full_code)
        tmpfile = handle.name
    try:
        result = subprocess.run(
            ["python", tmpfile],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"PATH": os.environ.get("PATH", "")},
        )
        passed = result.returncode == 0
        return {
            "passed": passed,
            "stdout": result.stdout[:500],
            "stderr": result.stderr[:500],
            "timeout": False,
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "timeout": True, "stderr": "Execution timed out"}
    finally:
        try:
            os.unlink(tmpfile)
        except OSError:
            pass


def evaluate_code_response(code: str, test: str | None = None) -> dict[str, Any]:
    """Full evaluation pipeline — syntax first, execution if test available."""
    code = extract_python_code(code)
    syntax = check_syntax(code)
    if not syntax["valid"]:
        return {
            "score": 0.0,
            "syntax_valid": False,
            "error": syntax["error"],
            "passed_tests": False,
        }

    if not test:
        return {
            "score": 0.5,
            "syntax_valid": True,
            "passed_tests": None,
            "note": "syntax valid, no test",
        }

    execution = run_with_timeout(code, test)
    score = 1.0 if execution["passed"] else 0.2
    return {
        "score": score,
        "syntax_valid": True,
        "passed_tests": execution["passed"],
        "timeout": execution.get("timeout", False),
        "stderr": execution.get("stderr", ""),
    }


def _load_code_problems(limit: int) -> list[dict[str, str]]:
    problems: list[dict[str, str]] = []
    for path in (HUMANEVAL_PATH, MBPP_PATH):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if path.name.startswith("humaneval"):
                prompt = item.get("prompt", "")
                body = item.get("canonical_solution", "")
                problems.append(
                    {
                        "question": f"write python code {prompt.strip()}",
                        "reference": f"{prompt}{body}".strip(),
                        "test": item.get("test", ""),
                    }
                )
            else:
                test_list = item.get("test_list") or []
                problems.append(
                    {
                        "question": str(item.get("text", "")),
                        "reference": str(item.get("code", "")),
                        "test": "\n".join(test_list),
                    }
                )
    if len(problems) > limit:
        rng = random.Random(42)
        problems = rng.sample(problems, limit)
    return problems


def benchmark_code_pass_rate(*, limit: int = 5, use_predict: bool = True) -> dict[str, Any]:
    """Measure unit-test pass rate on held-out code problems."""
    problems = _load_code_problems(limit)
    if not problems:
        return {"score": 0.0, "passed": False, "cases": [], "reason": "no code corpus"}

    cases: list[dict[str, Any]] = []
    passed = 0

    for problem in problems:
        test = problem.get("test") or ""
        if use_predict:
            from brain.predict_engine import predict_with_steps

            generated = predict_with_steps(problem["question"], force=True)
            code = (generated or {}).get("answer", "")
            source = "predict"
        else:
            code = problem["reference"]
            source = "reference"

        evaluation = evaluate_code_response(code, test or None)
        ok = bool(evaluation.get("passed_tests"))
        if ok:
            passed += 1
        cases.append(
            {
                "question": problem["question"][:120],
                "source": source,
                "score": evaluation.get("score", 0.0),
                "passed_tests": evaluation.get("passed_tests"),
                "syntax_valid": evaluation.get("syntax_valid"),
            }
        )

    score = passed / len(problems)
    return {
        "score": round(score, 4),
        "passed": score >= 0.5,
        "cases": cases,
        "total": len(problems),
        "passed_count": passed,
    }
