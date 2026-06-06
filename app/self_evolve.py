"""Self-evolution — read/edit own codebase on a fork branch; never push main without approval."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_BRANCHES = frozenset({"main", "master", "HEAD"})
ALLOWED_PREFIXES = ("app/", "brain/", "src/", "tests/", "scripts/", "pipeline/")
BLOCKED_SEGMENTS = ("..", ".env", "secrets", "credentials", ".git/")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def fork_remote() -> str:
    return _env("AUREON_SELF_EVOLVE_FORK_REMOTE", "houseofasher")


def branch_prefix() -> str:
    return _env("AUREON_SELF_EVOLVE_BRANCH_PREFIX", "aureon/self-evolve-")


def validate_repo_path(rel_path: str) -> Path:
    """Reject path traversal and secrets; allow only source tree files."""
    clean = rel_path.replace("\\", "/").lstrip("/")
    if not clean or any(seg in clean for seg in BLOCKED_SEGMENTS):
        raise ValueError(f"Path not allowed: {rel_path}")
    if not any(clean.startswith(p) for p in ALLOWED_PREFIXES):
        raise ValueError(f"Path outside allowed prefixes: {rel_path}")
    full = (ROOT / clean).resolve()
    if not str(full).startswith(str(ROOT.resolve())):
        raise ValueError("Path traversal rejected")
    return full


def _run_git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "git failed")
    return result


def repo_status() -> dict[str, Any]:
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], check=False).stdout.strip()
    remote = fork_remote()
    dirty = _run_git(["status", "--porcelain"], check=False).stdout.strip()
    return {
        "root": str(ROOT),
        "current_branch": branch,
        "fork_remote": remote,
        "dirty": bool(dirty),
        "allowed_prefixes": list(ALLOWED_PREFIXES),
        "policy": "Fork-only pushes; main/master blocked without explicit approval.",
    }


def list_source_files(*, limit: int = 200) -> list[str]:
    files: list[str] = []
    for prefix in ALLOWED_PREFIXES:
        base = ROOT / prefix.rstrip("/")
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json", ".md"):
                rel = path.relative_to(ROOT).as_posix()
                files.append(rel)
                if len(files) >= limit:
                    return files
    return files


def read_source(rel_path: str) -> dict[str, Any]:
    path = validate_repo_path(rel_path)
    if not path.is_file():
        raise FileNotFoundError(rel_path)
    content = path.read_text(encoding="utf-8")
    return {"path": rel_path, "bytes": len(content.encode("utf-8")), "content": content}


def write_source(rel_path: str, content: str) -> dict[str, Any]:
    if len(content.encode("utf-8")) > 512_000:
        raise ValueError("File too large (max 512KB)")
    path = validate_repo_path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"path": rel_path, "written": True, "bytes": len(content.encode("utf-8"))}


def create_evolution_branch(description: str) -> dict[str, Any]:
    slug = re.sub(r"[^a-z0-9-]+", "-", description.lower())[:40].strip("-") or "upgrade"
    branch = f"{branch_prefix()}{slug}-{int(time.time())}"
    if branch in FORBIDDEN_BRANCHES:
        raise ValueError("Invalid branch name")
    _run_git(["checkout", "-b", branch])
    return {"branch": branch, "description": description}


def commit_evolution(message: str, *, paths: list[str] | None = None) -> dict[str, Any]:
    if paths:
        for p in paths:
            validate_repo_path(p)
        _run_git(["add", *paths])
    else:
        _run_git(["add", "-A"])
    status = _run_git(["status", "--porcelain"], check=False).stdout.strip()
    if not status:
        return {"committed": False, "reason": "nothing to commit"}
    _run_git(["commit", "-m", message[:500]])
    sha = _run_git(["rev-parse", "--short", "HEAD"]).stdout.strip()
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    return {"committed": True, "branch": branch, "sha": sha, "message": message}


def push_fork(*, branch: str | None = None, approved: bool = False) -> dict[str, Any]:
    if not approved:
        return {
            "pushed": False,
            "reason": "Push requires explicit approve_push=true — main is never pushed automatically.",
        }
    remote = fork_remote()
    if not remote:
        return {"pushed": False, "reason": "AUREON_SELF_EVOLVE_FORK_REMOTE not configured"}
    target = branch or _run_git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    if target in FORBIDDEN_BRANCHES:
        return {"pushed": False, "reason": f"Refusing to push protected branch: {target}"}
    _run_git(["push", "-u", remote, target])
    return {
        "pushed": True,
        "remote": remote,
        "branch": target,
        "note": "Open a PR on GitHub to merge into main after review.",
    }


def plan_evolution(task: str) -> dict[str, Any]:
    """Suggest files likely relevant to a self-upgrade task."""
    task_l = task.lower()
    suggestions: list[str] = []
    keywords: dict[str, list[str]] = {
        "app/chat_service.py": ["chat", "routing", "reply", "route"],
        "brain/predict_engine.py": ["predict", "bootstrap", "reasoning", "seed"],
        "brain/code_master.py": ["code", "humaneval", "python", "function"],
        "brain/philosophy_handler.py": ["philosophy", "god", "belief", "faith"],
        "brain/identity_handler.py": ["identity", "who are you", "aureon"],
        "app/self_evolve.py": ["self", "evolve", "fork", "upgrade"],
        "app/organism.py": ["organism", "security", "nomad"],
        "app/main.py": ["api", "endpoint", "route"],
    }
    for path, keys in keywords.items():
        if any(k in task_l for k in keys):
            suggestions.append(path)
    if not suggestions:
        suggestions = ["app/chat_service.py", "brain/predict_engine.py"]
    return {
        "task": task,
        "suggested_files": suggestions,
        "workflow": [
            "1. POST /api/brain/self/read — inspect suggested files",
            "2. POST /api/brain/self/write — apply changes on fork branch",
            "3. POST /api/brain/self/commit — commit locally",
            "4. POST /api/brain/self/push with approve_push=true — push fork only",
            "5. Open GitHub PR for human approval before merging to main",
        ],
    }


def run_evolution_cycle(
    task: str,
    *,
    writes: list[dict[str, str]] | None = None,
    approve_push: bool = False,
) -> dict[str, Any]:
    """Full cycle: branch → optional writes → commit → optional fork push."""
    plan = plan_evolution(task)
    branch_info = create_evolution_branch(task)
    written: list[str] = []
    if writes:
        for item in writes:
            path = item.get("path", "")
            content = item.get("content", "")
            if path and content is not None:
                write_source(path, content)
                written.append(path)
    commit = commit_evolution(f"self-evolve: {task[:200]}", paths=written or None)
    push = push_fork(branch=branch_info["branch"], approved=approve_push)
    return {
        "ok": True,
        "plan": plan,
        "branch": branch_info["branch"],
        "written": written,
        "commit": commit,
        "push": push,
    }
