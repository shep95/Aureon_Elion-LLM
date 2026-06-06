"""Self-evolution — read/edit own codebase on a fork branch; never push main without approval."""

from __future__ import annotations

import ast
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


def skip_syntax_verify() -> bool:
    return _env("AUREON_SELF_EVOLVE_SKIP_VERIFY", "").lower() in ("1", "true", "yes")


def skip_test_gate() -> bool:
    return _env("AUREON_SELF_EVOLVE_SKIP_TESTS", "").lower() in ("1", "true", "yes")


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


def analyze_file_for_task(rel_path: str, task: str) -> dict[str, Any]:
    """Read a file and identify relevant sections for the task (AST-based, not keyword-only)."""
    source = read_source(rel_path)["content"]
    analysis: dict[str, Any] = {
        "path": rel_path,
        "task": task,
        "line_count": len(source.splitlines()),
        "functions": [],
        "classes": [],
        "imports": [],
        "issues": [],
        "recommendations": [],
    }

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        analysis["issues"].append(f"Syntax error: {exc}")
        analysis["recommendations"].append("Fix syntax before any evolve patch.")
        return analysis

    analysis["module_docstring"] = (ast.get_docstring(tree) or "")[:400]

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            analysis["imports"].append({
                "module": "",
                "names": [a.name for a in node.names],
                "line": node.lineno,
            })
        elif isinstance(node, ast.ImportFrom):
            analysis["imports"].append({
                "module": node.module or "",
                "names": [a.name for a in node.names],
                "line": node.lineno,
            })

    task_tokens = {t for t in re.findall(r"[a-z0-9]+", task.lower()) if len(t) > 2}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            doc = ast.get_docstring(node) or ""
            name_l = node.name.lower()
            fn_tokens = set(re.findall(r"[a-z0-9]+", name_l + " " + doc.lower()))
            relevant = bool(task_tokens & fn_tokens)
            entry = {
                "name": node.name,
                "line": node.lineno,
                "end_line": getattr(node, "end_lineno", node.lineno),
                "docstring": doc[:240],
                "task_relevant": relevant,
                "arg_count": len(node.args.args),
            }
            analysis["functions"].append(entry)
            if not doc and relevant:
                analysis["issues"].append(f"Function `{node.name}` (line {node.lineno}) lacks docstring.")
        elif isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node) or ""
            analysis["classes"].append({
                "name": node.name,
                "line": node.lineno,
                "docstring": doc[:240],
            })

    relevant = [f for f in analysis["functions"] if f["task_relevant"]]
    analysis["task_relevant_functions"] = [f["name"] for f in relevant]

    if rel_path.endswith(".py"):
        if not relevant and analysis["functions"]:
            analysis["recommendations"].append(
                "No function names overlap task keywords — review manually or enrich docstrings."
            )
        elif relevant:
            analysis["recommendations"].append(
                f"Start with: {', '.join(f['name'] for f in relevant[:4])}."
            )
        if analysis["issues"]:
            analysis["recommendations"].append(
                "Prefer docstring enrichment or append-only verified helpers — not in-place rewrites."
            )
        else:
            analysis["recommendations"].append(
                "Use code_master retrieval for new helpers; pytest gate runs before commit."
            )

    return analysis


def verify_before_commit(paths: list[str]) -> dict[str, Any]:
    """Run syntax check on modified files before committing."""
    results: dict[str, Any] = {"passed": [], "failed": [], "skipped": []}

    for rel_path in paths:
        if not rel_path.endswith(".py"):
            results["skipped"].append(rel_path)
            continue
        try:
            content = read_source(rel_path)["content"]
            ast.parse(content)
            results["passed"].append(rel_path)
        except SyntaxError as exc:
            results["failed"].append({"path": rel_path, "error": str(exc)})
        except (ValueError, FileNotFoundError) as exc:
            results["failed"].append({"path": rel_path, "error": str(exc)})

    results["ok"] = len(results["failed"]) == 0
    return results


def run_tests_before_commit(*, timeout: int = 120) -> dict[str, Any]:
    """Run the test suite — block commit if tests fail."""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "-x", "-q", "--tb=short"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        return {
            "passed": False,
            "output": output[-2000:],
            "errors": "pytest timed out",
        }

    combined = (result.stdout or "") + (result.stderr or "")
    return {
        "passed": result.returncode == 0,
        "output": combined[-2000:],
        "returncode": result.returncode,
    }


def _paths_from_git_status() -> list[str]:
    status = _run_git(["status", "--porcelain"], check=False).stdout.strip()
    paths: list[str] = []
    for line in status.splitlines():
        if len(line) < 4:
            continue
        rel = line[3:].strip().replace("\\", "/")
        if " -> " in rel:
            rel = rel.split(" -> ", 1)[1]
        if rel:
            paths.append(rel)
    return paths


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

    verify_paths = paths if paths else _paths_from_git_status()
    verification: dict[str, Any] | None = None
    tests: dict[str, Any] | None = None

    if verify_paths and not skip_syntax_verify():
        verification = verify_before_commit(verify_paths)
        if not verification["ok"]:
            return {
                "committed": False,
                "reason": "syntax verification failed",
                "verification": verification,
            }

    if not skip_test_gate():
        tests = run_tests_before_commit()
        if not tests["passed"]:
            return {
                "committed": False,
                "reason": "test suite failed",
                "verification": verification,
                "tests": tests,
            }

    _run_git(["commit", "-m", message[:500]])
    sha = _run_git(["rev-parse", "--short", "HEAD"]).stdout.strip()
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    result: dict[str, Any] = {"committed": True, "branch": branch, "sha": sha, "message": message}
    if verification is not None:
        result["verification"] = verification
    if tests is not None:
        result["tests"] = {"passed": True}
    return result


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

    file_analysis: list[dict[str, Any]] = []
    for path in suggestions[:5]:
        try:
            file_analysis.append(analyze_file_for_task(path, task))
        except (ValueError, FileNotFoundError) as exc:
            file_analysis.append({"path": path, "task": task, "issues": [str(exc)]})

    return {
        "task": task,
        "suggested_files": suggestions,
        "analysis": file_analysis,
        "capabilities": {
            "can": [
                "create git branches on a fork",
                "read/write source files in allowed prefixes",
                "AST-based file analysis (functions, classes, imports)",
                "algorithmic patch proposals (predict + code_master + AST)",
                "syntax verification (ast.parse) before every commit",
                "pytest gate before every commit (disable with AUREON_SELF_EVOLVE_SKIP_TESTS=1)",
                "commit and push to fork with explicit approval gate",
            ],
            "cannot_yet": [
                "autonomously decide architecturally sound refactors from task text alone",
                "generate reliable novel production code at d_model=128 without retrieval",
                "reason about correctness beyond syntax + existing test suite",
            ],
            "brain": "algorithmic (predict + code_master + AST) — scales when a stronger model is wired to write_source()",
        },
        "workflow": [
            "1. POST /api/brain/self/plan — file suggestions + AST analysis",
            "2. POST /api/brain/self/auto — algorithmic patch cycle (predict + code_master)",
            "3. POST /api/brain/self/read — inspect suggested files",
            "4. POST /api/brain/self/write — apply manual overrides if needed",
            "5. POST /api/brain/self/commit — syntax check + pytest, then commit locally",
            "6. POST /api/brain/self/push with approve_push=true — push fork only",
            "7. Open GitHub PR for human approval before merging to main",
        ],
    }


def run_evolution_cycle(
    task: str,
    *,
    writes: list[dict[str, str]] | None = None,
    approve_push: bool = False,
    algorithmic: bool = True,
) -> dict[str, Any]:
    """Full cycle: branch → algorithmic or manual writes → verify → commit → optional fork push."""
    plan = plan_evolution(task)
    branch_info = create_evolution_branch(task)
    written: list[str] = []
    evolution: dict[str, Any] | None = None

    if writes:
        for item in writes:
            path = item.get("path", "")
            content = item.get("content", "")
            if path and content is not None:
                write_source(path, content)
                written.append(path)
    elif algorithmic:
        from brain.evolve_engine import propose_evolution_writes

        evolution = propose_evolution_writes(task, plan)
        for item in evolution.get("writes", []):
            path = item.get("path", "")
            content = item.get("content", "")
            if path and content is not None:
                write_source(path, content)
                written.append(path)

    commit = commit_evolution(f"self-evolve: {task[:200]}", paths=written or None)
    push = push_fork(branch=branch_info["branch"], approved=approve_push)
    result: dict[str, Any] = {
        "ok": commit.get("committed", False) or bool(written),
        "plan": plan,
        "branch": branch_info["branch"],
        "written": written,
        "commit": commit,
        "push": push,
    }
    if evolution is not None:
        result["evolution"] = evolution
    return result
