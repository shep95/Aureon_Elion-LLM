"""Vector RAG tests."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from brain.vector_rag import VectorRAGIndex, retrieve_with_citations
from db.models import Base, Document, KnowledgeDomain


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_rag_retrieves_matching_document(monkeypatch):
    index = VectorRAGIndex()

    def fake_rebuild(self):
        self._hits = []
        from brain.vector_rag import RagHit

        self._hits = [
            RagHit(1, "hash1", "Paris", "Paris is the capital of France in Europe.", "test", 0.0),
            RagHit(2, "hash2", "Berlin", "Berlin is the capital of Germany.", "test", 0.0),
        ]
        corpus = [h.snippet() for h in self._hits]
        self._matrix = self._vectorizer.fit_transform(corpus)
        self._built_at = 1.0
        return 2

    monkeypatch.setattr(VectorRAGIndex, "rebuild", fake_rebuild)

    from brain import vector_rag as vr

    monkeypatch.setattr(vr, "_index", None)
    hits = vr.get_rag_index(force_rebuild=True).retrieve("capital of France", top_k=2)
    assert hits
    assert any("paris" in h.title.lower() or "paris" in h.text.lower() for h in hits)


def test_retrieve_with_citations_returns_metadata(monkeypatch):
    index = VectorRAGIndex()

    def fake_rebuild(self):
        from brain.vector_rag import RagHit

        self._hits = [
            RagHit(9, "abc", "DNA", "DNA stores genetic information in cells.", "biology", 0.0),
        ]
        self._matrix = self._vectorizer.fit_transform([h.snippet() for h in self._hits])
        self._built_at = 1.0
        return 1

    monkeypatch.setattr(VectorRAGIndex, "rebuild", fake_rebuild)
    from brain import vector_rag as vr

    monkeypatch.setattr(vr, "_index", None)
    _ctx, hits, citations = retrieve_with_citations("what is dna", top_k=1)
    assert citations[0]["document_id"] == 9
    assert citations[0]["content_hash"] == "abc"


def test_retrieve_with_citations_domain_locks_paths(monkeypatch):
    def fake_rebuild(self):
        from brain.vector_rag import RagHit

        self._hits = [
            RagHit(
                1,
                "silicon",
                "Silicon Valley Quantum Startups",
                "Quantum computer funding news from Silicon Valley and India.",
                "test",
                0.0,
                {"domain": "technology_and_engineering", "subdomain": "computer_science", "micro_subdomain": "startups"},
            ),
            RagHit(
                2,
                "physics",
                "Quantum Mechanics",
                "Quantum computer qubits use superposition, gates, measurement, and entanglement.",
                "test",
                0.0,
                {
                    "domain": "science_and_natural_philosophy",
                    "subdomain": "physics",
                    "micro_subdomain": "quantum_mechanics",
                },
            ),
        ]
        self._matrix = self._vectorizer.fit_transform([h.snippet() for h in self._hits])
        self._built_at = 1.0
        return 2

    monkeypatch.setattr(VectorRAGIndex, "rebuild", fake_rebuild)
    from brain import vector_rag as vr

    monkeypatch.setattr(vr, "_index", None)
    _ctx, hits, citations = retrieve_with_citations(
        "quantum computer",
        top_k=2,
        paths=["science_and_natural_philosophy.physics.quantum_mechanics"],
    )
    assert len(hits) == 1
    assert hits[0].content_hash == "physics"
    assert citations[0]["domain"] == "science_and_natural_philosophy"
