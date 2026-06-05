"""Alerting when evaluation gates fail."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import requests

from pipeline.config import ALERT_WEBHOOK_URL, BENCHMARKS_DIR, ensure_dirs


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def fire_alert(title: str, details: dict[str, Any]) -> dict[str, Any]:
    ensure_dirs()
    alert = {
        "title": title,
        "timestamp": _utcnow(),
        "details": details,
        "severity": "critical" if details.get("halt_training") else "warning",
    }

    log_path = BENCHMARKS_DIR / "alerts.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(alert, ensure_ascii=False) + "\n")

    webhook_status = "skipped"
    if ALERT_WEBHOOK_URL:
        try:
            response = requests.post(ALERT_WEBHOOK_URL, json=alert, timeout=10)
            webhook_status = f"sent_{response.status_code}"
        except requests.RequestException as exc:
            webhook_status = f"failed_{exc.__class__.__name__}"

    alert["webhook_status"] = webhook_status
    return alert
