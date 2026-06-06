"""Curiosity sandbox — isolated prototype build + gated GitHub/Railway deploy."""

from __future__ import annotations

import ast
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from app.curiosity_proposals import (
    create_proposal,
    get_proposal,
    mark_deployed,
    mark_sandbox_ready,
    record_audit,
)
from app.self_evolve import ROOT, create_evolution_branch, push_fork, repo_status, write_source
from brain.curiosity_engine import format_curiosity_report, run_market_research

logger = logging.getLogger(__name__)

SANDBOX_ROOT = ROOT / "data" / "curiosity" / "sandbox"
RAILWAY_SECTIONS = ROOT / "data" / "curiosity" / "railway-sections"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", text.lower())[:32].strip("-") or "proto"


def _path_ref(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sandbox_dir(proposal_id: str) -> Path:
    return SANDBOX_ROOT / proposal_id[:8]


def _sanitize_label(text: str, *, limit: int = 80) -> str:
    clean = re.sub(r"[^\w\s\-./]", "", text or "")[:limit].strip()
    return clean or "general"


def verify_prototype_code(code: str) -> dict[str, Any]:
    """Security + syntax gate for curiosity-generated prototype modules."""
    from brain.code_evaluator import check_forbidden_constructs, check_syntax

    syntax = check_syntax(code)
    if not syntax.get("valid"):
        return {"ok": False, "syntax_valid": False, "security_valid": False, "error": syntax.get("error")}

    forbidden = check_forbidden_constructs(code)
    if not forbidden.get("safe"):
        return {
            "ok": False,
            "syntax_valid": True,
            "security_valid": False,
            "error": forbidden.get("error"),
        }

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"ok": False, "syntax_valid": False, "security_valid": False, "error": str(exc)}

    has_run = any(
        isinstance(n, ast.FunctionDef) and n.name == "run"
        for n in ast.walk(tree)
    )
    if not has_run:
        return {"ok": False, "syntax_valid": True, "security_valid": True, "error": "missing run() entry point"}

    return {"ok": True, "syntax_valid": True, "security_valid": True, "error": None}


def verify_sandbox_proposal(proposal_id: str) -> dict[str, Any]:
    """Execute verification on all sandbox .py prototypes for a proposal."""
    proposal = get_proposal(proposal_id)
    if not proposal:
        return {"ok": False, "error": "proposal_not_found"}

    out_dir = sandbox_dir(proposal_id)
    if not out_dir.is_dir():
        return {"ok": False, "error": "sandbox_missing", "proposal_id": proposal_id}

    modules: list[dict[str, Any]] = []
    all_ok = True
    for path in sorted(out_dir.rglob("*.py")):
        code = path.read_text(encoding="utf-8")
        check = verify_prototype_code(code)
        runtime: dict[str, Any] | None = None
        if check.get("ok"):
            runtime = _run_prototype_file(path)
            if not runtime.get("ok"):
                check = {**check, "ok": False, "error": runtime.get("error")}
                all_ok = False
        else:
            all_ok = False
        modules.append({
            "path": _path_ref(path),
            "verification": check,
            "runtime": runtime,
        })

    return {
        "ok": all_ok and bool(modules),
        "proposal_id": proposal_id,
        "modules": modules,
        "module_count": len(modules),
    }


def _run_prototype_file(path: Path) -> dict[str, Any]:
    """Import sandbox module in isolation and call run() — no side effects expected."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"curiosity_proto_{path.stem}", path)
    if spec is None or spec.loader is None:
        return {"ok": False, "error": "import spec failed"}
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        return {"ok": False, "error": f"import failed: {exc}"}
    run_fn = getattr(module, "run", None)
    if not callable(run_fn):
        return {"ok": False, "error": "run() not callable"}
    try:
        result = run_fn()
    except Exception as exc:
        return {"ok": False, "error": f"run() failed: {exc}"}
    if not isinstance(result, dict):
        return {"ok": False, "error": "run() must return dict"}
    if result.get("status") != "sandbox_prototype":
        return {"ok": False, "error": "unexpected run() status"}
    return {"ok": True, "result": result}


def _build_prototype_module(domain: str, task: str, proposal_id: str) -> tuple[str, str]:
    """Generate a verified Python prototype module for the sandbox."""
    slug = _slug(domain)
    safe_domain = _sanitize_label(domain)
    safe_task = _sanitize_label(task, limit=200)
    rel_path = f"brain/curiosity_proto_{slug}.py"
    code = (
        f'"""Curiosity sandbox prototype — {safe_domain} (proposal {proposal_id[:8]}).\n\n'
        f"Task: {safe_task}\n"
        f'Human approval required before merge to main or Railway deploy.\n"""\n\n'
        f"from __future__ import annotations\n\n"
        f"DOMAIN = {safe_domain!r}\n"
        f"PROPOSAL_ID = {proposal_id!r}\n"
        f"TASK = {safe_task!r}\n\n"
        f"def describe() -> str:\n"
        f'    """Return capability summary for this curiosity prototype."""\n'
        f'    return f"Curiosity prototype {{DOMAIN}}: {{TASK[:120]}}"\n\n'
        f"def run() -> dict:\n"
        f'    """Sandbox entry point — safe to import, no side effects."""\n'
        f"    return {{\n"
        f'        "domain": DOMAIN,\n'
        f'        "proposal_id": PROPOSAL_ID,\n'
        f'        "task": TASK,\n'
        f'        "status": "sandbox_prototype",\n'
        f"    }}\n"
    )
    verified = verify_prototype_code(code)
    if not verified.get("ok"):
        raise ValueError(f"prototype verification failed: {verified.get('error')}")
    return rel_path, code


def _write_research_brief(proposal: dict[str, Any], out_dir: Path) -> Path:
    brief_path = out_dir / "research-brief.json"
    brief = {
        "proposal_id": proposal["id"],
        "self_intro": proposal.get("self_intro"),
        "curiosity_reflection": proposal.get("curiosity_reflection"),
        "research": proposal.get("research", []),
        "advancements": proposal.get("advancements", []),
    }
    brief_path.write_text(json.dumps(brief, indent=2), encoding="utf-8")
    return brief_path


def _railway_manifest(proposal: dict[str, Any], advancement: dict[str, Any]) -> dict[str, Any]:
    section = advancement.get("railway_section") or f"aureon-curiosity-{_slug(advancement.get('domain', 'adv'))}"
    return {
        "service_name": section,
        "description": advancement.get("prototype_task", "Curiosity-driven Aureon prototype"),
        "domain": advancement.get("domain"),
        "proposal_id": proposal["id"],
        "requires_human_approval": True,
        "deploy_model": "Separate Railway service from fork branch after PR merge",
        "healthcheck_path": "/health",
        "suggested_env": {
            "AUREON_CURIOSITY_ENABLED": "1",
            "AUREON_WEB_SEARCH_ENABLED": "1",
            "AUREON_API_KEY": "${AUREON_API_KEY}",
        },
        "github_branch_pattern": f"aureon/curiosity-{proposal['id'][:8]}-*",
    }


def build_sandbox(proposal_id: str) -> dict[str, Any]:
    """Build isolated prototype files — no git writes on main tree."""
    proposal = get_proposal(proposal_id)
    if not proposal:
        return {"ok": False, "error": "proposal_not_found"}

    advancements = proposal.get("advancements") or []
    if not advancements:
        return {
            "ok": True,
            "sandbox_built": False,
            "reason": "no_advancements_identified",
            "proposal_id": proposal_id,
        }

    out_dir = sandbox_dir(proposal_id)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sandbox_files: list[str] = []
    manifests: list[dict[str, Any]] = []

    for adv in advancements[:2]:
        rel_path, code = _build_prototype_module(
            str(adv.get("domain", "general")),
            str(adv.get("prototype_task", "curiosity prototype")),
            proposal_id,
        )
        target = out_dir / rel_path.replace("/", "_")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code, encoding="utf-8")
        sandbox_files.append(_path_ref(target))
        manifests.append(_railway_manifest(proposal, adv))

    brief = _write_research_brief(proposal, out_dir)
    sandbox_files.append(_path_ref(brief))

    RAILWAY_SECTIONS.mkdir(parents=True, exist_ok=True)
    for i, manifest in enumerate(manifests):
        manifest_path = RAILWAY_SECTIONS / f"{proposal_id[:8]}-{i}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        sandbox_files.append(_path_ref(manifest_path))

    railway_section = manifests[0]["service_name"] if manifests else None
    verification = verify_sandbox_proposal(proposal_id)
    if not verification.get("ok"):
        return {
            "ok": False,
            "error": "sandbox_verification_failed",
            "verification": verification,
            "proposal_id": proposal_id,
        }

    updated = mark_sandbox_ready(
        proposal_id,
        sandbox_path=_path_ref(out_dir),
        sandbox_files=sandbox_files,
        railway_section=railway_section,
    )
    record_audit(proposal_id, "sandbox_built", f"files={len(sandbox_files)}")

    return {
        "ok": True,
        "sandbox_built": True,
        "proposal_id": proposal_id,
        "sandbox_path": _path_ref(out_dir),
        "sandbox_files": sandbox_files,
        "railway_section": railway_section,
        "railway_manifests": manifests,
        "verification": verification,
        "proposal": updated,
    }


def _apply_sandbox_to_repo(proposal: dict[str, Any]) -> dict[str, Any]:
    """After human approval — write sandbox prototypes to evolution branch."""
    advancements = proposal.get("advancements") or []
    if not advancements:
        return {"written": [], "branch": None}

    task = str(advancements[0].get("prototype_task", "curiosity prototype"))
    branch_info = create_evolution_branch(f"curiosity-{proposal['id'][:8]}-{task[:40]}")
    branch = branch_info["branch"]
    written: list[str] = []

    for adv in advancements[:2]:
        rel_path, code = _build_prototype_module(
            str(adv.get("domain", "general")),
            str(adv.get("prototype_task", task)),
            proposal["id"],
        )
        write_source(rel_path, code)
        written.append(rel_path)

    from app.self_evolve import commit_evolution

    commit = commit_evolution(
        f"curiosity: {task[:200]} (proposal {proposal['id'][:8]})",
        paths=written,
    )
    return {"written": written, "branch": branch, "commit": commit}


def push_github_brief(
    proposal: dict[str, Any],
    *,
    approved: bool = False,
) -> dict[str, Any]:
    """Push research brief + Railway manifest to GitHub via Contents API (Railway-safe)."""
    if not approved:
        return {"pushed": False, "reason": "GitHub push requires explicit approve_github=true"}

    from app.learning_github_sync import (
        _ensure_branch,
        _upsert_file,
        github_repos,
        github_token,
    )

    token = github_token()
    if not token:
        return {"pushed": False, "reason": "AUREON_GITHUB_TOKEN not configured"}

    branch = f"curiosity/{proposal['id'][:8]}"
    brief = json.dumps({
        "proposal_id": proposal["id"],
        "self_intro": proposal.get("self_intro"),
        "curiosity_reflection": proposal.get("curiosity_reflection"),
        "advancements": proposal.get("advancements"),
        "railway_section": proposal.get("railway_section"),
        "status": proposal.get("status"),
    }, indent=2).encode("utf-8")

    repos_pushed: list[str] = []
    errors: list[str] = []

    for repo_slug in github_repos()[:3]:
        if "/" not in repo_slug:
            continue
        owner, repo = repo_slug.split("/", 1)
        try:
            _ensure_branch(owner, repo, branch, token)
            _upsert_file(
                owner,
                repo,
                branch,
                f"curiosity/proposals/{proposal['id'][:8]}/brief.json",
                brief,
                f"curiosity: research brief {proposal['id'][:8]}",
                token,
            )
            sandbox = sandbox_dir(proposal["id"])
            if sandbox.is_dir():
                for path in sandbox.rglob("*.py"):
                    rel = f"curiosity/proposals/{proposal['id'][:8]}/sandbox/{path.name}"
                    _upsert_file(
                        owner,
                        repo,
                        branch,
                        rel,
                        path.read_bytes(),
                        f"curiosity: sandbox {path.name}",
                        token,
                    )
            repos_pushed.append(repo_slug)
        except Exception as exc:
            logger.warning("curiosity github push failed for %s: %s", repo_slug, exc)
            errors.append(f"{repo_slug}: {exc}")

    return {
        "pushed": bool(repos_pushed),
        "branch": branch,
        "repos": repos_pushed,
        "errors": errors,
    }


def deploy_proposal(
    proposal_id: str,
    *,
    approve_github: bool = False,
    approve_push: bool = False,
    reviewer: str = "human",
) -> dict[str, Any]:
    """
    Deploy after human approval: git branch + optional fork push + GitHub brief.
    Main/master never pushed.
    """
    from app.curiosity_proposals import approve_proposal

    proposal = get_proposal(proposal_id)
    if not proposal:
        return {"ok": False, "error": "proposal_not_found"}

    status = proposal.get("status")
    if status == "deployed":
        return {"ok": False, "error": "already_deployed", "status": status}
    if status == "rejected":
        return {"ok": False, "error": "rejected", "status": status}

    if status == "approved":
        pass
    elif status in ("pending_approval", "sandbox_ready"):
        approval = approve_proposal(proposal_id, approved=True, reviewer=reviewer)
        if not approval.get("ok"):
            return approval
        proposal = approval.get("proposal") or proposal
    else:
        return {"ok": False, "error": "not_ready_for_deploy", "status": status}

    if not proposal.get("advancements"):
        return {"ok": False, "error": "no_advancements", "status": status}

    sandbox_check = verify_sandbox_proposal(proposal_id)
    if not sandbox_check.get("ok"):
        return {"ok": False, "error": "sandbox_verification_failed", "verification": sandbox_check}

    repo_result = _apply_sandbox_to_repo(proposal)
    commit = repo_result.get("commit") or {}
    if repo_result.get("written") and not commit.get("committed"):
        return {
            "ok": False,
            "error": "commit_failed",
            "deploy": {"repo": repo_result},
            "reason": commit.get("reason"),
        }

    github_result = push_github_brief(proposal, approved=approve_github)
    push_result = push_fork(
        branch=repo_result.get("branch"),
        approved=approve_push,
    )

    deploy = {
        "repo": repo_result,
        "github": github_result,
        "push": push_result,
        "sandbox_verification": sandbox_check,
        "repo_status": repo_status(),
        "branch": repo_result.get("branch"),
        "github_branch": github_result.get("branch"),
    }
    mark_deployed(proposal_id, deploy)
    record_audit(proposal_id, "deployed", f"branch={repo_result.get('branch')}")

    return {
        "ok": True,
        "proposal_id": proposal_id,
        "deploy": deploy,
        "policy": "Main never pushed. Railway section deploys after you merge PR and add Railway service.",
        "railway_section": proposal.get("railway_section"),
        "next_steps": [
            f"Review branch `{repo_result.get('branch')}` on GitHub",
            f"Open PR to merge curiosity prototype",
            f"Add Railway service `{proposal.get('railway_section')}` wired to merged branch",
        ],
    }


def run_curiosity_cycle(
    *,
    focus: str | None = None,
    search_fn: Any | None = None,
    auto_sandbox: bool = True,
) -> dict[str, Any]:
    """Full curiosity pipeline: research -> proposal -> sandbox -> pending human approval."""
    research = run_market_research(focus=focus, search_fn=search_fn)
    if not research.get("ok"):
        return research

    proposal = create_proposal(research)
    proposal_id = proposal["id"]
    record_audit(proposal_id, "research_complete")

    sandbox_result: dict[str, Any] | None = None
    if auto_sandbox and proposal.get("advancements"):
        sandbox_result = build_sandbox(proposal_id)

    report = format_curiosity_report({**research, "proposal_id": proposal_id})
    if sandbox_result and sandbox_result.get("sandbox_built"):
        report += (
            f"\n\n**Sandbox:** `{sandbox_result.get('sandbox_path')}` "
            f"({len(sandbox_result.get('sandbox_files', []))} files)\n"
            f"**Railway section (proposed):** `{sandbox_result.get('railway_section')}`\n"
            "Reply `/curious approve " + proposal_id[:8] + "` when ready to deploy."
        )

    return {
        "ok": True,
        "research": research,
        "proposal": get_proposal(proposal_id),
        "sandbox": sandbox_result,
        "report": report,
        "requires_human_approval": True,
    }
