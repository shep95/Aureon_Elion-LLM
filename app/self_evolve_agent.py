"""Autonomous self-coding — algorithmic brain loop on fork branch (main always blocked)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from app.self_evolve import (
    commit_evolution,
    create_evolution_branch,
    plan_evolution,
    push_fork,
    repo_status,
    write_source,
)
from brain.evolve_engine import propose_evolution_writes

logger = logging.getLogger(__name__)

HISTORY_PATH = Path(__file__).resolve().parents[1] / "data" / "self_evolve" / "history.jsonl"


def _record_event(event: dict[str, Any]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    event["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, default=str) + "\n")
    try:
        from app.audit import get_audit_log

        get_audit_log().record(
            "mutating_request",
            detail=f"self_evolve:{event.get('action', 'unknown')} branch={event.get('branch', '')}",
        )
    except Exception:
        pass


def get_history(*, limit: int = 50) -> list[dict[str, Any]]:
    if not HISTORY_PATH.is_file():
        return []
    lines = HISTORY_PATH.read_text(encoding="utf-8").splitlines()
    items: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(items))


def run_autonomous_evolution(
    task: str,
    *,
    auto_push_fork: bool = True,
    max_files: int = 3,
) -> dict[str, Any]:
    """
    Full autonomous cycle: branch → AST analyze → predict/code_master patch → verify → commit → push fork.
    Uses Aureon's algorithm stack — not an external LLM.
    """
    if not task.strip():
        raise ValueError("task required")

    plan = plan_evolution(task)
    branch_info = create_evolution_branch(task)
    branch = branch_info["branch"]

    evolution = propose_evolution_writes(task, plan, max_files=max_files)
    written: list[str] = []
    for item in evolution["writes"]:
        path = item.get("path", "")
        content = item.get("content", "")
        if path and content is not None:
            write_source(path, content)
            written.append(path)

    commit = commit_evolution(f"auto-evolve: {task[:200]}", paths=written or None)
    push = push_fork(branch=branch, approved=auto_push_fork)

    event = {
        "action": "autonomous_evolution",
        "task": task,
        "branch": branch,
        "written": written,
        "strategy": evolution.get("strategy"),
        "proposals": [
            {k: v for k, v in p.items() if k != "content"}
            for p in evolution.get("proposals", [])
        ],
        "commit": commit,
        "push": push,
        "auto_push_fork": auto_push_fork,
        "main_blocked": True,
        "brain": evolution.get("brain"),
    }
    _record_event(event)

    return {
        "ok": commit.get("committed", False) or bool(written),
        "autonomous": True,
        "policy": "Fork push allowed without PR approval; main/master never pushed.",
        "brain": evolution.get("brain"),
        "strategy": evolution.get("strategy"),
        "plan": plan,
        "evolution": evolution,
        "branch": branch,
        "written": written,
        "commit": commit,
        "push": push,
        "repo": repo_status(),
        "history_entry": event,
    }
