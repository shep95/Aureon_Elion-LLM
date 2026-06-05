"""Event queue abstraction — file-based locally, Kafka-ready in production."""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.config import QUEUE_DIR, ensure_dirs


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PipelineEvent:
    step: str
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=_utcnow)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> PipelineEvent:
        return cls(**json.loads(raw))


class EventQueue(ABC):
    """Interface compatible with Apache Kafka producers/consumers."""

    @abstractmethod
    def publish(self, topic: str, event: PipelineEvent) -> None: ...

    @abstractmethod
    def consume(self, topic: str, limit: int = 100) -> list[PipelineEvent]: ...


class FileEventQueue(EventQueue):
    """Local queue backed by JSONL files (default for Railway/dev)."""

    def __init__(self, base_dir: Path | None = None) -> None:
        ensure_dirs()
        self.base_dir = base_dir or QUEUE_DIR

    def _topic_path(self, topic: str) -> Path:
        safe = topic.replace("/", "_")
        return self.base_dir / f"{safe}.jsonl"

    def publish(self, topic: str, event: PipelineEvent) -> None:
        path = self._topic_path(topic)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(event.to_json() + "\n")

    def consume(self, topic: str, limit: int = 100) -> list[PipelineEvent]:
        path = self._topic_path(topic)
        if not path.exists():
            return []
        events: list[PipelineEvent] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    events.append(PipelineEvent.from_json(line))
                if len(events) >= limit:
                    break
        return events


class KafkaEventQueue(EventQueue):
    """
    Kafka adapter — activated when KAFKA_BOOTSTRAP_SERVERS is set.
    Falls back to logging if kafka-python is unavailable.
    """

    def __init__(self, bootstrap_servers: str) -> None:
        self.bootstrap_servers = bootstrap_servers
        try:
            from kafka import KafkaConsumer, KafkaProducer  # type: ignore

            self._producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers.split(","),
                value_serializer=lambda v: v.encode("utf-8"),
            )
            self._consumer_cls = KafkaConsumer
            self._available = True
        except Exception:
            self._fallback = FileEventQueue()
            self._available = False

    def publish(self, topic: str, event: PipelineEvent) -> None:
        if not self._available:
            self._fallback.publish(topic, event)
            return
        self._producer.send(topic, event.to_json())
        self._producer.flush()

    def consume(self, topic: str, limit: int = 100) -> list[PipelineEvent]:
        if not self._available:
            return self._fallback.consume(topic, limit=limit)
        consumer = self._consumer_cls(
            topic,
            bootstrap_servers=self.bootstrap_servers.split(","),
            auto_offset_reset="earliest",
            consumer_timeout_ms=2000,
            value_deserializer=lambda v: v.decode("utf-8"),
        )
        events: list[PipelineEvent] = []
        for message in consumer:
            events.append(PipelineEvent.from_json(message.value))
            if len(events) >= limit:
                break
        consumer.close()
        return events


def get_queue() -> EventQueue:
    import os

    servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "").strip()
    if servers:
        return KafkaEventQueue(servers)
    return FileEventQueue()
