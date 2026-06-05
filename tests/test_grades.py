"""Academic grade curriculum and graduation tests."""

from __future__ import annotations

from brain.grades import GRADE_CURRICULUM, get_grade, next_grade
from brain.cortex import bootstrap_brain
from brain.domains.taxonomy import total_micro_subdomains
from db.session import get_session, init_db


def test_grade_curriculum_order():
    assert len(GRADE_CURRICULUM) == 7
    assert GRADE_CURRICULUM[0].slug == "preschool"
    assert GRADE_CURRICULUM[-1].slug == "doctorate"
    for i, g in enumerate(GRADE_CURRICULUM):
        assert g.order == i


def test_next_grade_chain():
    assert next_grade("preschool").slug == "elementary"
    assert next_grade("doctorate") is None


def test_seed_grade_progress():
    init_db()
    bootstrap_brain()
    with get_session() as session:
        from sqlalchemy import func, select

        from db.models import GradeProgress

        count = session.scalar(select(func.count()).select_from(GradeProgress)) or 0
        assert count >= total_micro_subdomains() * 7
        preschool = session.scalar(
            select(GradeProgress).where(GradeProgress.grade_slug == "preschool").limit(1)
        )
        assert preschool is not None
        assert preschool.status in ("unlocked", "in_progress", "graduated", "failed")


def test_get_grade():
    g = get_grade("undergraduate")
    assert g is not None
    assert g.phase == "intermediate"
