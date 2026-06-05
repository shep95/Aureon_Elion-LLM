"""Labeler region — teacher model + active learning for human review queue."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from brain.base import AgentContext, AgentResult, MicroAgentBase
from db.models import Document, DocumentLabel
from pipeline.step2_labeling.runner import TeacherLabeler, active_learning_split


class LabelerAgent(MicroAgentBase):
    region = "labeler"

    def run(self, session: Session, ctx: AgentContext) -> AgentResult:
        query = (
            select(Document)
            .where(Document.domain_id == ctx.domain_id, Document.verified.is_(True))
        )
        if ctx.subdomain_id:
            query = query.where(Document.subdomain_id == ctx.subdomain_id)
        docs = session.scalars(query).all()

        if not docs:
            return AgentResult(
                region=self.region,
                status="skipped",
                metrics={"reason": "no verified documents"},
            )

        rows = [
            {
                "title": d.title,
                "text": d.text,
                "metadata": d.extra or {},
            }
            for d in docs
        ]
        labeler = TeacherLabeler()
        labeled = labeler.label(rows)
        auto, review = active_learning_split(labeled)
        review_set = {id(r) for r in review}

        labels_written = 0
        review_count = 0

        for doc, row in zip(docs, labeled):
            existing = session.scalar(
                select(DocumentLabel).where(DocumentLabel.document_id == doc.id)
            )
            if existing:
                continue
            needs_review = id(row) in review_set
            session.add(
                DocumentLabel(
                    document_id=doc.id,
                    domain_id=ctx.domain_id,
                    subdomain_id=ctx.subdomain_id,
                    label=row.get("label", ctx.subdomain_slug or ctx.domain_slug),
                    confidence=float(row.get("label_confidence", 0.5)),
                    label_source=row.get("label_source", "teacher_model"),
                    needs_review=needs_review,
                )
            )
            labels_written += 1
            if needs_review:
                review_count += 1

        return AgentResult(
            region=self.region,
            status="completed",
            metrics={
                "labeled": labels_written,
                "auto_labeled": labels_written - review_count,
                "flagged_for_review": review_count,
                "review_rate": round(review_count / max(labels_written, 1), 4),
            },
        )
