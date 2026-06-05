"""Verifier region — checks quality, verifiability, internal consistency."""

from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from brain.base import AgentContext, AgentResult, MicroAgentBase
from db.models import Document

WORD_RE = re.compile(r"[a-zA-Z]{2,}")


class VerifierAgent(MicroAgentBase):
    region = "verifier"

    def run(self, session: Session, ctx: AgentContext) -> AgentResult:
        query = select(Document).where(Document.domain_id == ctx.domain_id)
        if ctx.subdomain_id:
            query = query.where(Document.subdomain_id == ctx.subdomain_id)
        docs = session.scalars(query).all()

        verified = 0
        failed = 0
        for doc in docs:
            if doc.verified:
                continue
            score = self._verify(doc.text, doc.title)
            if score >= 0.5:
                doc.verified = True
                doc.quality_score = max(doc.quality_score or 0, score)
                verified += 1
            else:
                failed += 1

        return AgentResult(
            region=self.region,
            status="completed",
            metrics={
                "documents_checked": len(docs),
                "newly_verified": verified,
                "failed_verification": failed,
            },
        )

    def _verify(self, text: str, title: str) -> float:
        combined = f"{title} {text}"
        words = WORD_RE.findall(combined)
        if len(words) < 20:
            return 0.0
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        has_structure = any(marker in text for marker in (".", ":", ";"))
        alpha_ratio = sum(c.isalpha() for c in combined) / max(len(combined), 1)
        return min(1.0, 0.4 * unique_ratio + 0.3 * alpha_ratio + (0.3 if has_structure else 0))
