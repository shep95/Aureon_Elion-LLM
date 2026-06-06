"""Label review API tests."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.label_review import list_pending_review, resolve_label
from db.models import Base, Document, DocumentLabel, KnowledgeDomain, KnowledgeSubdomain


@pytest.fixture
def review_db(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    @contextmanager
    def _get_session():
        session = factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr("app.label_review.get_session", _get_session)

    with _get_session() as session:
        domain = KnowledgeDomain(slug="mathematics", name="Mathematics")
        session.add(domain)
        session.flush()
        subdomain = KnowledgeSubdomain(domain_id=domain.id, slug="algebra", name="Algebra")
        session.add(subdomain)
        session.flush()
        doc = Document(
            domain_id=domain.id,
            subdomain_id=subdomain.id,
            source="test",
            title="Linear maps",
            text="A linear map preserves vector addition and scalar multiplication.",
            content_hash="abc123",
        )
        session.add(doc)
        session.flush()
        label = DocumentLabel(
            document_id=doc.id,
            domain_id=domain.id,
            subdomain_id=subdomain.id,
            label="linear_algebra",
            confidence=0.42,
            needs_review=True,
        )
        session.add(label)
        session.flush()
        label_id = label.id

    return label_id


def test_list_pending_review(review_db: int):
    result = list_pending_review(limit=10)
    assert result["total_pending"] == 1
    assert result["pending"][0]["label_id"] == review_db
    assert result["pending"][0]["needs_review"] is True


def test_resolve_label_approve(review_db: int):
    approved = resolve_label(review_db, label="vectors", approve=True)
    assert approved["ok"] is True
    assert approved["action"] == "approved"
    assert approved["label"]["proposed_label"] == "vectors"
    assert approved["label"]["needs_review"] is False

    pending = list_pending_review()
    assert pending["total_pending"] == 0
