"""Quick assessment of Aureon code understanding + generation."""
from __future__ import annotations

from brain.code_master import benchmark_humaneval, generate_master_code, get_code_bank


def main() -> None:
    bank = get_code_bank()
    he = sum(1 for p in bank.problems if p.source == "humaneval")
    mb = sum(1 for p in bank.problems if p.source == "mbpp")
    print("=== CORPUS ===")
    print(f"Total problems: {len(bank.problems)} (HumanEval={he}, MBPP={mb})")

    print("\n=== SAMPLE PROMPTS (retrieval-only, no neural) ===")
    prompts = [
        "write a python function to add two numbers",
        "write a python function to reverse a string",
        "write a python function to check if a number is prime",
        "implement binary search in python",
        "write a function to find the longest common subsequence",
    ]
    for q in prompts:
        r = generate_master_code(q, predict_fn=lambda _: None)
        ev = r.get("code_eval", {})
        method = str(r.get("method", "?"))[:22]
        print(
            f"  [{method:22}] tests={ev.get('passed_tests')} "
            f"syntax={ev.get('syntax_valid')} conf={r.get('confidence', 0):.2f}"
        )
        print(f"    Q: {q}")

    print("\n=== HUMANEVAL pass@1 (retrieval + verification) ===")
    for n in [10, 20, 50, 100, 164]:
        rep = benchmark_humaneval(limit=n, use_retrieval=True)
        gate = "PASS" if rep["passed_doctorate_gate"] else "below 90%"
        print(f"  n={n:3d} -> {rep['passed']}/{rep['total']} = {rep['pass_rate']*100:.1f}% ({gate})")


if __name__ == "__main__":
    main()
