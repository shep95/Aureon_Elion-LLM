"""Code corpus collector — HumanEval, MBPP, and inline examples."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.config import ROOT
from pipeline.step1_collection.collectors import RawDocument

HUMANEVAL_PATH = ROOT / "data" / "code" / "humaneval-python.jsonl"
MBPP_PATH = ROOT / "data" / "code" / "mbpp.jsonl"

CODE_MICRO_SLUGS = frozenset(
    {
        "python_functions",
        "python_algorithms",
        "python_classes",
        "javascript_functions",
        "sql_queries",
    }
)


class CodeCollector:
    name = "code_corpus"

    def collect(self, limit: int = 2000) -> list[RawDocument]:
        docs: list[RawDocument] = []

        if HUMANEVAL_PATH.exists():
            for line in HUMANEVAL_PATH.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                prompt = item.get("prompt", "")
                solution_body = item.get("canonical_solution", "")
                full_solution = f"{prompt}{solution_body}".strip()
                test = item.get("test", "")
                text = f"question write python code {prompt.strip()} answer {full_solution}"
                docs.append(
                    RawDocument(
                        doc_id=f"humaneval_{item['task_id']}",
                        source="humaneval",
                        title=item["task_id"],
                        text=text,
                        url="",
                        metadata={
                            "domain": "technology_and_engineering",
                            "subdomain": "computer_science",
                            "micro_subdomain": "python_functions",
                            "code_area": "code_generation",
                            "has_tests": True,
                            "test": test,
                            "prompt": prompt,
                            "canonical_solution": full_solution,
                        },
                    )
                )

        if MBPP_PATH.exists():
            for line in MBPP_PATH.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                test_list = item.get("test_list") or []
                test_code = "\n".join(test_list)
                text = f"question {item.get('text', '')} answer {item.get('code', '')}"
                docs.append(
                    RawDocument(
                        doc_id=f"mbpp_{item.get('task_id', '')}",
                        source="mbpp",
                        title=str(item.get("task_id", "")),
                        text=text,
                        url="",
                        metadata={
                            "domain": "technology_and_engineering",
                            "subdomain": "computer_science",
                            "micro_subdomain": "python_algorithms",
                            "code_area": "code_generation",
                            "has_tests": bool(test_list),
                            "test": test_code,
                        },
                    )
                )

        return docs[:limit]
