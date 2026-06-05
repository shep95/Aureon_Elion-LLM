"""Reward region — RLHF approximation per domain."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from brain.base import AgentContext, AgentResult, MicroAgentBase
from db.models import PreferencePair
from pipeline.step5_rlhf.runner import DEFAULT_PREFERENCES, score_response, train_reward_model


class RewardAgent(MicroAgentBase):
    region = "reward"

    def run(self, session: Session, ctx: AgentContext) -> AgentResult:
        prefs = list(DEFAULT_PREFERENCES)
        db_prefs = session.scalars(
            select(PreferencePair).where(PreferencePair.domain_id == ctx.domain_id).limit(20)
        ).all()
        for p in db_prefs:
            prefs.append({"context": p.context, "preferred": p.preferred, "rejected": p.rejected})

        if len(prefs) < 2:
            return AgentResult(
                region=self.region,
                status="skipped",
                metrics={"reason": "insufficient preference pairs"},
            )

        reward_model, extractor, val_metrics = train_reward_model(prefs, epochs=min(ctx.epochs, 300))

        ranking_checks = []
        for pair in prefs[:4]:
            pref_score = score_response(reward_model, extractor, pair["context"], pair["preferred"])
            rej_score = score_response(reward_model, extractor, pair["context"], pair["rejected"])
            ranking_checks.append(pref_score > rej_score)

        ranking_accuracy = sum(ranking_checks) / max(len(ranking_checks), 1)

        return AgentResult(
            region=self.region,
            status="completed",
            metrics={
                "reward_val_accuracy": round(val_metrics["accuracy"], 4),
                "ranking_accuracy": round(ranking_accuracy, 4),
                "preference_pairs_used": len(prefs),
            },
        )
