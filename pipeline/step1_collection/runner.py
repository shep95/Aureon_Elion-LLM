"""Step 1 runner — collect, filter, publish to queue."""

from __future__ import annotations

from typing import Any

from pipeline.config import ensure_dirs
from pipeline.queue import PipelineEvent, get_queue
from pipeline.step1_collection.collectors import (
    ArxivCollector,
    GutenbergCollector,
    LocalFileCollector,
    SeedCollector,
    save_raw_batch,
)
from brain.regions.code_collector import CodeCollector
from pipeline.step1_collection.filters import load_raw_batches, run_filter_pipeline


def run_step1(
    arxiv_limit: int = 10,
    gutenberg_limit: int = 1,
    seed_limit: int = 20,
) -> dict[str, Any]:
    ensure_dirs()
    queue = get_queue()

    collectors = [
        SeedCollector(),
        CodeCollector(),
        LocalFileCollector(),
        ArxivCollector(),
        GutenbergCollector(),
    ]
    limits = {
        "seeds": seed_limit,
        "code_corpus": 2000,
        "local_inbox": 50,
        "arxiv": arxiv_limit,
        "gutenberg": gutenberg_limit,
    }

    collected = []
    source_counts: dict[str, int] = {}
    for collector in collectors:
        limit = limits.get(collector.name, 20)
        batch = collector.collect(limit=limit)
        source_counts[collector.name] = len(batch)
        collected.extend(batch)

    raw_path = save_raw_batch(collected)
    all_raw = load_raw_batches()
    accepted, filter_stats = run_filter_pipeline(all_raw)

    event = PipelineEvent(
        step="1_collection",
        event_type="corpus_ready",
        payload={
            "raw_batch_path": str(raw_path),
            "source_counts": source_counts,
            **filter_stats,
        },
    )
    queue.publish("pipeline.data.clean", event)

    return {
        "step": 1,
        "name": "automated_data_collection",
        "collected_this_run": len(collected),
        "source_counts": source_counts,
        "raw_batch_path": str(raw_path),
        **filter_stats,
    }
