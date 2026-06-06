"""Academic grade curriculum — teach the algorithm like a child through doctorate."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GradeLevel:
    slug: str
    name: str
    order: int
    phase: str
    description: str
    epoch_factor: float
    accuracy_gate: float
    reasoning_gate: float
    consistency_gate: float
    collection_limit: int
    min_train_accuracy: float


# Preschool → doctorate. Each step must graduate before the next unlocks.
GRADE_CURRICULUM: tuple[GradeLevel, ...] = (
    GradeLevel(
        slug="preschool",
        name="Pre-School",
        order=0,
        phase="early",
        description="First exposure — simple words, patterns, and naming.",
        epoch_factor=0.4,
        accuracy_gate=0.35,
        reasoning_gate=0.20,
        consistency_gate=0.35,
        collection_limit=1,
        min_train_accuracy=0.45,
    ),
    GradeLevel(
        slug="elementary",
        name="Elementary",
        order=1,
        phase="basic",
        description="Foundational concepts and vocabulary in the domain.",
        epoch_factor=0.55,
        accuracy_gate=0.45,
        reasoning_gate=0.25,
        consistency_gate=0.40,
        collection_limit=2,
        min_train_accuracy=0.50,
    ),
    GradeLevel(
        slug="middle_school",
        name="Middle School",
        order=2,
        phase="basic",
        description="Intermediate basics — connecting ideas within the subdomain.",
        epoch_factor=0.7,
        accuracy_gate=0.50,
        reasoning_gate=0.30,
        consistency_gate=0.45,
        collection_limit=2,
        min_train_accuracy=0.55,
    ),
    GradeLevel(
        slug="high_school",
        name="High School",
        order=3,
        phase="intermediate",
        description="Secondary mastery — structured problems and definitions.",
        epoch_factor=0.85,
        accuracy_gate=0.55,
        reasoning_gate=0.33,
        consistency_gate=0.50,
        collection_limit=3,
        min_train_accuracy=0.60,
    ),
    GradeLevel(
        slug="undergraduate",
        name="Undergraduate",
        order=4,
        phase="intermediate",
        description="College-level theory and application.",
        epoch_factor=1.0,
        accuracy_gate=0.60,
        reasoning_gate=0.40,
        consistency_gate=0.55,
        collection_limit=3,
        min_train_accuracy=0.65,
    ),
    GradeLevel(
        slug="masters",
        name="Master's",
        order=5,
        phase="advanced",
        description="Graduate research methods and specialization.",
        epoch_factor=1.15,
        accuracy_gate=0.65,
        reasoning_gate=0.45,
        consistency_gate=0.60,
        collection_limit=4,
        min_train_accuracy=0.70,
    ),
    GradeLevel(
        slug="doctorate",
        name="Doctorate (PhD)",
        order=6,
        phase="advanced",
        description="Doctoral research depth — original synthesis and verification.",
        epoch_factor=1.3,
        accuracy_gate=0.70,
        reasoning_gate=0.50,
        consistency_gate=0.65,
        collection_limit=5,
        min_train_accuracy=0.75,
    ),
)

GRADES_BY_SLUG: dict[str, GradeLevel] = {g.slug: g for g in GRADE_CURRICULUM}

CODE_MICRO_SUBDOMAINS = frozenset(
    {
        "python_functions",
        "python_algorithms",
        "python_classes",
        "javascript_functions",
        "sql_queries",
    }
)

# Code graduation uses unit-test pass rate instead of classification accuracy.
CODE_GRADUATION_THRESHOLDS: dict[str, float] = {
    "preschool": 0.30,
    "elementary": 0.45,
    "middle_school": 0.55,
    "high_school": 0.65,
    "undergraduate": 0.75,
    "masters": 0.82,
    "doctorate": 0.90,
}


def is_code_micro(micro_slug: str | None) -> bool:
    return bool(micro_slug and micro_slug in CODE_MICRO_SUBDOMAINS)


def grade_slugs() -> list[str]:
    return [g.slug for g in GRADE_CURRICULUM]


def get_grade(slug: str) -> GradeLevel | None:
    return GRADES_BY_SLUG.get(slug)


def next_grade(slug: str) -> GradeLevel | None:
    grade = get_grade(slug)
    if not grade:
        return None
    nxt = grade.order + 1
    for g in GRADE_CURRICULUM:
        if g.order == nxt:
            return g
    return None


def first_grade() -> GradeLevel:
    return GRADE_CURRICULUM[0]


def grade_to_dict(grade: GradeLevel) -> dict:
    return {
        "slug": grade.slug,
        "name": grade.name,
        "order": grade.order,
        "phase": grade.phase,
        "description": grade.description,
        "epoch_factor": grade.epoch_factor,
        "accuracy_gate": grade.accuracy_gate,
        "reasoning_gate": grade.reasoning_gate,
        "consistency_gate": grade.consistency_gate,
        "collection_limit": grade.collection_limit,
        "min_train_accuracy": grade.min_train_accuracy,
    }


def curriculum_public() -> list[dict]:
    return [grade_to_dict(g) for g in GRADE_CURRICULUM]


def epochs_for_grade(base_epochs: int, grade: GradeLevel) -> int:
    return max(50, min(500, int(base_epochs * grade.epoch_factor)))


def evaluate_grade_gates(grade: GradeLevel, benchmarks: dict[str, dict]) -> dict:
    """Apply grade-specific thresholds to benchmark results."""
    gates = {
        "reasoning": benchmarks["reasoning"]["score"] >= grade.reasoning_gate,
        "consistency": benchmarks["consistency"]["score"] >= grade.consistency_gate,
        "verification": benchmarks["verification"]["score"] >= grade.accuracy_gate,
    }
    return {
        "gates": gates,
        "all_passed": all(gates.values()),
        "grade_slug": grade.slug,
        "thresholds": {
            "reasoning": grade.reasoning_gate,
            "consistency": grade.consistency_gate,
            "verification": grade.accuracy_gate,
        },
    }


def evaluate_code_grade_gates(grade: GradeLevel, pass_rate: float) -> dict:
    """Apply code-specific graduation thresholds (unit-test pass rate)."""
    threshold = CODE_GRADUATION_THRESHOLDS.get(grade.slug, grade.accuracy_gate)
    passed = pass_rate >= threshold
    return {
        "gates": {"code_pass_rate": passed},
        "all_passed": passed,
        "grade_slug": grade.slug,
        "pass_rate": round(pass_rate, 4),
        "thresholds": {"code_pass_rate": threshold},
    }
