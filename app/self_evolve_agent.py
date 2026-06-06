"""Autonomous self-coding — full fork cycle without human PR approval (main always blocked)."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from app.self_evolve import (
    commit_evolution,
    create_evolution_branch,
    plan_evolution,
    push_fork,
    read_source,
    repo_status,
    write_source,
)

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


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower())[:32].strip("-") or "auto"


def _autonomous_patch(path: str, content: str, task: str) -> str | None:
    """Generate a minimal autonomous patch — docstring stamp + optional code from code_master."""
    task_l = task.lower()
    stamp = (
        f"\n\n# SOLIA auto-evolve ({time.strftime('%Y-%m-%d')})\n"
        f"# Task: {task[:200]}\n"
    )

    if path.endswith(".py") and any(k in task_l for k in ("function", "code", "implement", "fix", "module")):
        try:
            from brain.code_master import generate_master_code

            prompt = f"write python code to {task} for file {path}"
            result = generate_master_code(prompt, predict_fn=lambda _: None)
            code = (result.get("answer") or "").strip()
            if code and "def " in code and result.get("code_eval", {}).get("syntax_valid"):
                if "# SOLIA auto-evolve" not in content:
                    return content.rstrip() + stamp + "\n\n" + code + "\n"
        except Exception as exc:
            logger.debug("code_master patch skipped: %s", exc)

    if "# SOLIA auto-evolve" in content:
        return None
    if path.endswith(".py"):
        return content.rstrip() + stamp
    if path.endswith(".md"):
        return content.rstrip() + f"\n\n---\n**SOLIA auto-evolve:** {task[:200]}\n"
    return None


def _maybe_create_module(task: str) -> tuple[str, str] | None:
    """Create a new brain module when task asks for new code."""
    task_l = task.lower()
    if not any(k in task_l for k in ("new module", "new file", "create module", "add module")):
        return None
    slug = _slug(task)
    path = f"brain/auto_{slug}.py"
    try:
        from brain.code_master import generate_master_code

        result = generate_master_code(
            f"write a python module for {task}",
            predict_fn=lambda _: None,
        )
        code = (result.get("answer") or "").strip()
        if not code or "def " not in code:
            code = (
                f'"""Auto-generated module — {task}."""\n\n'
                f"def run() -> str:\n"
                f'    return "SOLIA auto module {slug}"\n'
            )
        header = f'"""SOLIA autonomous module — {task}."""\n\n'
        if not code.startswith('"""'):
            code = header + code
        return path, code
    except Exception:
        return path, (
            f'"""SOLIA autonomous module — {task}."""\n\n'
            f"def run() -> str:\n"
            f'    return "SOLIA auto module {slug}"\n'
        )


def run_autonomous_evolution(
    task: str,
    *,
    auto_push_fork: bool = True,
    max_files: int = 3,
) -> dict[str, Any]:
    """
    Full autonomous cycle: branch → read → patch/write → commit → push fork.
    Never touches main/master — no human PR approval required for fork push.
    """
    if not task.strip():
        raise ValueError("task required")

    plan = plan_evolution(task)
    branch_info = create_evolution_branch(task)
    branch = branch_info["branch"]
    written: list[str] = []

    new_mod = _maybe_create_module(task)
    if new_mod:
        path, code = new_mod
        write_source(path, code)
        written.append(path)

    for path in plan["suggested_files"][:max_files]:
        if path in written:
            continue
        try:
            current = read_source(path)["content"]
        except (ValueError, FileNotFoundError):
            continue
        patched = _autonomous_patch(path, current, task)
        if patched and patched != current:
            write_source(path, patched)
            written.append(path)

    commit = commit_evolution(f"auto-evolve: {task[:200]}", paths=written or None)
    push = push_fork(branch=branch, approved=auto_push_fork)

    event = {
        "action": "autonomous_evolution",
        "task": task,
        "branch": branch,
        "written": written,
        "commit": commit,
        "push": push,
        "auto_push_fork": auto_push_fork,
        "main_blocked": True,
    }
    _record_event(event)

    return {
        "ok": True,
        "autonomous": True,
        "policy": "Fork push allowed without PR approval; main/master never pushed.",
        "plan": plan,
        "branch": branch,
        "written": written,
        "commit": commit,
        "push": push,
        "repo": repo_status(),
        "history_entry": event,
    }
