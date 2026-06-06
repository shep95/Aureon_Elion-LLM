"""Trainer region — backpropagation per domain/subdomain."""

from __future__ import annotations

import uuid

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from brain.base import AgentContext, AgentResult, MicroAgentBase
from db.models import Document, DocumentLabel, TrainingRun
from brain.grades import epochs_for_grade
from pipeline.config import LEARNING_RATE, MODELS_DIR, ensure_dirs
from pipeline.step3_training.registry import ModelRegistry
from src.neural_network import NeuralNetwork
from src.text_features import TextFeatureExtractor

class TrainerAgent(MicroAgentBase):
    region = "trainer"

    def run(self, session: Session, ctx: AgentContext) -> AgentResult:
        labels_q = (
            select(DocumentLabel, Document)
            .join(Document, DocumentLabel.document_id == Document.id)
            .where(
                DocumentLabel.domain_id == ctx.domain_id,
                DocumentLabel.needs_review.is_(False),
            )
        )
        if ctx.micro_subdomain_id:
            labels_q = labels_q.where(Document.micro_subdomain_id == ctx.micro_subdomain_id)
        elif ctx.subdomain_id:
            labels_q = labels_q.where(DocumentLabel.subdomain_id == ctx.subdomain_id)

        pairs = session.execute(labels_q).all()
        if len(pairs) < 2:
            return AgentResult(
                region=self.region,
                status="skipped",
                metrics={"reason": "insufficient labeled data", "samples": len(pairs)},
            )

        texts = [f"{doc.title} {doc.text}" for _, doc in pairs]
        label_names = sorted({lbl.label for lbl, _ in pairs})
        label_to_idx = {name: i for i, name in enumerate(label_names)}
        y = np.array([label_to_idx[lbl.label] for lbl, _ in pairs])

        if len(set(y.tolist())) < 2:
            return AgentResult(
                region=self.region,
                status="skipped",
                metrics={"reason": "need at least 2 classes"},
            )

        extractor = TextFeatureExtractor(max_features=min(256, max(32, len(texts) * 4)))
        x = extractor.fit_transform(texts)

        hidden = min(64, max(8, x.shape[1] // 2))
        train_epochs = ctx.epochs
        if ctx.grade:
            train_epochs = epochs_for_grade(ctx.epochs, ctx.grade)
        network = NeuralNetwork(
            layer_sizes=[x.shape[1], hidden, len(label_names)],
            learning_rate=LEARNING_RATE,
            output_activation="softmax",
        )
        perm = np.random.default_rng(42).permutation(len(y))
        val_n = max(1, len(y) // 5) if len(y) >= 5 else 0
        if val_n:
            val_idx = perm[:val_n]
            train_idx = perm[val_n:]
            x_train, y_train = x[train_idx], y[train_idx]
            x_val, y_val = x[val_idx], y[val_idx]
        else:
            x_train, y_train = x, y
            x_val, y_val = x, y

        network.train(x_train, y_train, epochs=train_epochs, verbose_every=0)
        train_metrics = network.evaluate(x_train, y_train)
        val_metrics = network.evaluate(x_val, y_val)

        ensure_dirs()
        run_id = str(uuid.uuid4())[:8]
        scope = ctx.scope_slug
        artifact_dir = MODELS_DIR / f"brain_{scope}_{run_id}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / "classifier.json"
        network.save(model_path)

        import json

        meta = {
            "labels": label_names,
            "feature_extractor": extractor.to_dict(),
            "scope": scope,
            "grade": ctx.grade_slug,
        }
        (artifact_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        registry = ModelRegistry()
        run_metrics = {
            "train_accuracy": round(train_metrics["accuracy"], 4),
            "val_accuracy": round(val_metrics["accuracy"], 4),
        }
        registry.log_run(run_id, run_metrics, str(model_path), {"scope": scope, "epochs": ctx.epochs})
        promoted = registry.should_promote(run_metrics)
        if promoted:
            registry.promote(
                {"run_id": run_id, "metrics": run_metrics, "artifact_path": str(model_path)}
            )

        session.add(
            TrainingRun(
                run_id=run_id,
                domain_id=ctx.domain_id,
                subdomain_id=ctx.subdomain_id,
                metrics=run_metrics,
                artifact_path=str(model_path),
                params={"epochs": train_epochs, "scope": scope, "grade": ctx.grade_slug},
                promoted=promoted,
            )
        )

        return AgentResult(
            region=self.region,
            status="completed",
            metrics={
                "run_id": run_id,
                "samples": len(texts),
                "classes": len(label_names),
                **run_metrics,
                "promoted": promoted,
            },
        )
