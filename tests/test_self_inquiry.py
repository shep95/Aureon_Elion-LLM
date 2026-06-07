"""Self-inquiry tests."""

from __future__ import annotations

import hashlib

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from brain.self_inquiry import (
    answer_question,
    fetch_learning_context,
    generate_questions,
    is_self_inquiry_enabled,
    recent_inquiries,
    reset_batch_inquiry_budget,
    run_self_inquiry_for_cycle,
)
from db.models import Base, Document, KnowledgeDomain, KnowledgeMicroSubdomain, KnowledgeSubdomain


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _seed_physics_micro(session: Session) -> tuple[str, str, str, int]:
    domain = KnowledgeDomain(slug="science_and_natural_philosophy", name="Science")
    session.add(domain)
    session.flush()
    subdomain = KnowledgeSubdomain(domain_id=domain.id, slug="physics", name="Physics")
    session.add(subdomain)
    session.flush()
    micro = KnowledgeMicroSubdomain(
        domain_id=domain.id,
        subdomain_id=subdomain.id,
        slug="classical_mechanics",
        name="Classical Mechanics",
    )
    session.add(micro)
    session.flush()
    text = "Classical mechanics describes motion using Newton's laws of force and acceleration."
    digest = hashlib.sha256(text.encode()).hexdigest()
    session.add(
        Document(
            domain_id=domain.id,
            subdomain_id=subdomain.id,
            micro_subdomain_id=micro.id,
            source="taxonomy",
            title="Introduction to Classical Mechanics",
            text=text,
            verified=True,
            quality_score=0.9,
            content_hash=digest,
            extra={"topic": "Newton's Laws"},
        )
    )
    session.commit()
    return domain.slug, subdomain.slug, micro.slug, micro.id


def test_generate_questions_for_physics():
    qs = generate_questions(
        domain_slug="science_and_natural_philosophy",
        subdomain_slug="physics",
        micro_slug="classical_mechanics",
        grade_slug="preschool",
        count=2,
    )
    assert len(qs) == 2
    assert all("?" in q for q in qs)


def test_quantum_mechanics_target_asks_seeded_prompt():
    qs = generate_questions(
        domain_slug="science_and_natural_philosophy",
        subdomain_slug="physics",
        micro_slug="quantum_mechanics",
        grade_slug="preschool",
        count=2,
    )
    assert qs[0] == "explain quantum mechanics to me"
    assert len(qs) == 2


def test_self_inquiry_runs_after_cycle(tmp_path, monkeypatch):
    monkeypatch.setenv("AUREON_SELF_INQUIRY", "1")
    monkeypatch.setenv("AUREON_DATA_DIR", str(tmp_path))
    reset_batch_inquiry_budget(limit=10)

    outcome = {
        "domain": "science_and_natural_philosophy",
        "subdomain": "physics",
        "micro_subdomain": "classical_mechanics",
        "grade": "preschool",
        "grade_name": "Pre-School",
        "graduation": {"passed": True, "unlocked_next": "elementary", "train_accuracy": 0.0},
        "regions": [
            {"region": "collector", "status": "completed", "metrics": {}},
            {"region": "trainer", "status": "skipped", "metrics": {"reason": "need at least 2 classes"}},
        ],
    }
    exchanges = run_self_inquiry_for_cycle(outcome)
    assert len(exchanges) == 2
    assert exchanges[0]["question"]
    assert "cycle" in exchanges[0]
    assert len(exchanges[0]["answer"]) <= 120
    assert len(recent_inquiries(5)) == 2


def test_self_inquiry_disabled(monkeypatch):
    monkeypatch.setenv("AUREON_SELF_INQUIRY", "0")
    assert is_self_inquiry_enabled() is False
    reset_batch_inquiry_budget(limit=10)
    outcome = {
        "domain": "science_and_natural_philosophy",
        "subdomain": "physics",
        "micro_subdomain": "classical_mechanics",
        "grade": "preschool",
        "graduation": {"passed": True},
        "regions": [],
    }
    assert run_self_inquiry_for_cycle(outcome) == []


def test_answer_question_short_and_separate_cycle():
    answer, cycle = answer_question(
        "What is classical mechanics?",
        outcome={
            "grade": "preschool",
            "grade_name": "Pre-School",
            "graduation": {"passed": True, "unlocked_next": "elementary"},
            "regions": [{"region": "reward", "status": "completed", "metrics": {}}],
        },
        ctx={"micro_display": "Classical Mechanics", "topic": "Newton's Laws"},
    )
    assert "unlocked elementary" in cycle.lower()
    assert "No verified text" in answer


def test_answer_uses_collected_documents(tmp_path, monkeypatch):
    db_file = tmp_path / "reflect.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file.as_posix()}")

    import db.session as db_session

    db_session._engine = None
    db_session._SessionLocal = None
    db_session.init_db()

    from db.session import get_session

    with get_session() as session:
        domain, subdomain, micro, _micro_id = _seed_physics_micro(session)

    learning = fetch_learning_context(domain, subdomain, micro)
    assert len(learning["documents"]) == 1

    answer, cycle = answer_question(
        "What is Classical Mechanics?",
        outcome={
            "grade": "preschool",
            "grade_name": "Pre-School",
            "graduation": {"passed": True, "unlocked_next": "elementary", "train_accuracy": 1.0},
            "regions": [{"region": "collector", "status": "completed", "metrics": {}}],
        },
        learning=learning,
        ctx={"micro_display": "Classical Mechanics", "topic": "Newton's Laws"},
    )
    assert "Newton" in answer
    assert len(answer) <= 90
    assert "Passed" in cycle


def test_quantum_prompt_uses_grounded_explanation():
    learning = {
        "documents": [
            {
                "title": "Quantum Mechanics — Core Idea",
                "text": (
                    "Quantum mechanics explains matter and energy at atomic and subatomic scales. "
                    "Particles are described by wavefunctions and measurements return probabilities."
                ),
                "source": "seeds",
            }
        ],
        "labels": [],
    }
    answer, _cycle = answer_question(
        "explain quantum mechanics to me",
        outcome={"grade": "elementary", "graduation": {"passed": True}, "regions": []},
        learning=learning,
        ctx={"micro_display": "Quantum Mechanics", "topic": "Quantum Mechanics"},
    )
    assert "Quantum mechanics explains" in answer
    assert len(answer) <= 120


def test_collector_question_lists_document_titles():
    learning = {
        "documents": [
            {"title": "Botany Seeds", "text": "Plants are living organisms.", "source": "taxonomy"},
        ],
        "labels": [],
    }
    answer, _cycle = answer_question(
        "What did my collector region find about Botany?",
        outcome={"grade": "elementary", "graduation": {"passed": True}, "regions": []},
        learning=learning,
        ctx={"micro_display": "Botany", "topic": "Botany"},
    )
    assert "Botany Seeds" in answer
    assert len(answer.split()) <= 12


def test_one_word_answer_from_labels():
    learning = {
        "documents": [],
        "labels": [{"label": "genetics", "count": 2, "avg_confidence": 0.88}],
    }
    answer, _cycle = answer_question(
        "What is one word I would use to describe Genetics?",
        outcome={"grade": "preschool", "graduation": {"passed": True}, "regions": []},
        learning=learning,
        ctx={"topic": "Genetics", "micro_display": "Genetics"},
    )
    assert answer == "genetics"
