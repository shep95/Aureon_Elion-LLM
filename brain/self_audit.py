"""Full codebase self-audit — improvements, security, workflow, and build proposals."""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any

from app.self_evolve import ROOT, analyze_file_for_task, list_source_files, repo_status

_CORE_MODULES = (
    "app/chat_service.py",
    "app/main.py",
    "app/security.py",
    "app/middleware.py",
    "app/self_evolve.py",
    "app/organism.py",
    "brain/predict_engine.py",
    "brain/evolve_engine.py",
    "brain/code_master.py",
    "brain/file_router.py",
    "brain/philosophy_handler.py",
)

_SECURITY_PATTERNS: tuple[tuple[str, str, str], ...] = (
    (r"\beval\s*\(", "high", "Dynamic eval() — code injection risk if input reaches it"),
    (r"\bexec\s*\(", "high", "Dynamic exec() — arbitrary code execution"),
    (r"pickle\.loads?\(", "medium", "Pickle deserialization — unsafe on untrusted bytes"),
    (r"shell\s*=\s*True", "high", "subprocess with shell=True — command injection"),
    (r"yaml\.load\s*\(", "medium", "Use yaml.safe_load — yaml.load can execute tags"),
    (r"os\.environ\.get\([\"']AUREON_API_KEY[\"'],\s*[\"'][\"']\)", "info", "API key optional when unset — dev-friendly, prod must set key"),
    (r"def skip_test_gate|def skip_syntax_verify", "medium", "Self-evolve can bypass pytest/syntax gates via env helpers"),
    (r"AUREON_SELF_EVOLVE_SKIP_(TESTS|VERIFY)", "medium", "Env flag can disable self-evolve verification gates"),
    (r"subprocess\.run\(", "low", "Subprocess call — verify timeout, no shell, bounded input"),
)

_WORKFLOW_MARKERS: tuple[tuple[str, str], ...] = (
    ("is_own_output", "Echo guard runs first — good"),
    ("is_code_question", "Code routing before web search — good"),
    ("ciper_payload", "Duplicate ciper_payload check after first branch — dead branch"),
    ("plan_evolution", "/evolve chat scaffolds targeted patches; /self-audit runs full tree read"),
    ("PUBLIC_PATHS", "/api/chat is public — agent/code/evolve scaffolds reachable without API key"),
)

_AUDIT_PHRASES = (
    r"\bself[\s-]?audit\b",
    r"\banalyz(e|ing)\s+(yourself|itself|your\s+code|your\s+codebase|the\s+algorithm)\b",
    r"\bcopy\s+of\s+(itself|yourself)\b",
    r"\b(security|workflow)\s+flaws?\b",
    r"\bimprovements?\s+(you\s+would|it\s+would|i\s+would)\b",
    r"\bwhat\s+(would\s+you|would\s+it)\s+(improve|fix|change|build)\b",
    r"\bfinal\s+test\b.*\b(improve|audit|security|workflow|copy)\b",
    r"\bgive\s+(the\s+)?algorithm\s+a\s+copy\s+of\s+itself\b",
    r"\bhow\s+(would|should)\s+(you|it)\s+fix\b",
    r"\bnew\s+software\s+(to\s+)?build\b",
)


def is_self_audit_request(text: str) -> bool:
    """Detect natural-language requests for full codebase introspection."""
    t = (text or "").strip().lower()
    if not t:
        return False
    if t.startswith("/self-audit") or t.startswith("/evolve audit"):
        return True
    if t.startswith("/evolve") and any(
        k in t for k in ("audit", "analyze", "security", "workflow", "copy of", "yourself", "itself")
    ):
        return True
    return any(re.search(p, t, re.I) for p in _AUDIT_PHRASES)


def _read_module(rel_path: str) -> str | None:
    full = ROOT / rel_path.replace("\\", "/")
    if not full.is_file():
        return None
    try:
        return full.read_text(encoding="utf-8")
    except OSError:
        return None


def _scan_file_security(rel_path: str, content: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    lines = content.splitlines()

    def _line_context(line_no: int) -> str:
        if line_no < 1 or line_no > len(lines):
            return ""
        return lines[line_no - 1].strip()

    def _likely_string_literal(line: str, pattern: str) -> bool:
        """Skip forbidden-pattern tuples and regex source lines."""
        stripped = line.strip()
        if stripped.startswith("#"):
            return True
        if stripped.startswith(("(", "[", '"', "'")) and ("'" in stripped or '"' in stripped):
            if pattern.replace("\\", "") in stripped.replace("\\", ""):
                return True
        if "SECURITY_PATTERNS" in line or "_SECURITY_PATTERNS" in line:
            return True
        if re.search(r'r["\']\\b' + pattern[:4], line):
            return True
        return False

    # AST: real eval/exec calls (not string mentions)
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name == "eval":
                findings.append({
                    "file": rel_path,
                    "line": node.lineno,
                    "severity": "high",
                    "detail": "Dynamic eval() call in AST",
                    "snippet": _line_context(node.lineno)[:120],
                })
            elif name == "exec":
                findings.append({
                    "file": rel_path,
                    "line": node.lineno,
                    "severity": "high",
                    "detail": "Dynamic exec() call in AST",
                    "snippet": _line_context(node.lineno)[:120],
                })
    except SyntaxError:
        pass

    for pattern, severity, detail in _SECURITY_PATTERNS:
        if pattern in (r"\beval\s*\(", r"\bexec\s*\("):
            continue  # handled by AST above
        for match in re.finditer(pattern, content):
            line = content[: match.start()].count("\n") + 1
            ctx = _line_context(line)
            if _likely_string_literal(ctx, pattern):
                continue
            if rel_path == "brain/self_audit.py" and (
                "SKIP_" in ctx or "skip_test" in ctx or "skip_syntax" in ctx
            ):
                continue
            findings.append({
                "file": rel_path,
                "line": line,
                "severity": severity,
                "detail": detail,
                "snippet": ctx[:120],
            })

    # Deduplicate same file:line:detail
    seen: set[tuple[str, int, str]] = set()
    unique: list[dict[str, Any]] = []
    for f in findings:
        key = (f["file"], f["line"], f["detail"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(f)
    return unique


def _scan_workflow() -> list[dict[str, Any]]:
    chat = _read_module("app/chat_service.py") or ""
    middleware = _read_module("app/middleware.py") or ""
    findings: list[dict[str, Any]] = []

    rule_count = len(re.findall(r"# Rule \d", chat))
    findings.append({
        "area": "chat_routing",
        "severity": "info",
        "detail": f"chat() uses {rule_count}+ ordered routing rules — order determines precedence",
    })

    if chat.count("ciper_payload = _ciper_chat_payload") >= 2:
        findings.append({
            "area": "chat_routing",
            "severity": "low",
            "detail": "Duplicate `_ciper_chat_payload` invocation — second block is unreachable after first return",
            "fix": "Remove dead second ciper branch or merge into single early exit",
        })

    if '"/api/chat"' in middleware or '"/api/chat/file"' in middleware:
        findings.append({
            "area": "auth_boundary",
            "severity": "medium",
            "detail": "POST /api/chat and /api/chat/file are PUBLIC_PATHS — no API key even when AUREON_API_KEY is set",
            "fix": "Keep rate limits; add optional chat API key tier or cap agent/code subprocess cost per IP",
        })

    evolve = _read_module("app/self_evolve.py") or ""
    if "skip_test_gate" in evolve and "skip_syntax_verify" in evolve:
        findings.append({
            "area": "self_evolve",
            "severity": "medium",
            "detail": "AUREON_SELF_EVOLVE_SKIP_TESTS / SKIP_VERIFY can disable gates on fork commits",
            "fix": "Hard-block skip flags when RAILWAY_ENVIRONMENT or AUREON_ENV=production",
        })

    return findings


def _logical_improvements(inventory: dict[str, Any], security: list[dict], workflow: list[dict]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = [
        {
            "priority": "high",
            "title": "Unified self-introspection pipeline",
            "rationale": "plan_evolution() keyword-matches ~8 files; full audit scans the allowed tree",
            "how": "run_self_audit() -> structured report -> /evolve propose applies verified patches on fork",
        },
        {
            "priority": "high",
            "title": "Production auth hardening",
            "rationale": "Mutating brain/self/* endpoints require API key; public chat can still trigger heavy paths",
            "how": "Set AUREON_API_KEY in prod; add chat cost budget for agent/code subprocess chains",
        },
        {
            "priority": "medium",
            "title": "Routing table refactor",
            "rationale": f"chat_service.py is {inventory.get('lines', {}).get('app/chat_service.py', '?')} lines with implicit rule ordering",
            "how": "Extract ordered RouteRule list with explicit priority + unit tests per rule",
        },
        {
            "priority": "medium",
            "title": "Stronger evolve brain",
            "rationale": "evolve_engine append-only stamps at d_model=128 — not architectural refactors",
            "how": "Wire external LLM or larger local model to write_source() behind same pytest gate",
        },
        {
            "priority": "low",
            "title": "Audit chain durability",
            "rationale": "Organism audit_immune organ is dormant without AUREON_AUDIT_CHAIN_KEY",
            "how": "Set persistent audit key in Railway secrets; alert on chain break",
        },
    ]
    high_sec = [f for f in security if f.get("severity") == "high"]
    if high_sec:
        items.insert(0, {
            "priority": "critical",
            "title": "Review high-severity static matches",
            "rationale": f"{len(high_sec)} high-severity pattern(s) in scanned modules",
            "how": "Triage each match — many are in evaluators/guards; document allowlist or refactor",
        })
    if any(w.get("area") == "chat_routing" and "Duplicate" in w.get("detail", "") for w in workflow):
        items.append({
            "priority": "low",
            "title": "Remove dead ciper branch",
            "rationale": "Reduces confusion when reasoning about chat precedence",
            "how": "Delete unreachable block; add regression test",
        })
    return items


def _new_software_proposals() -> list[dict[str, Any]]:
    return [
        {
            "module": "brain/self_audit.py",
            "purpose": "Read-only full-tree introspection (this module)",
            "status": "implemented",
        },
        {
            "module": "app/chat_route_registry.py",
            "purpose": "Declarative chat routing with priority, tests, and metrics",
            "status": "proposed",
        },
        {
            "module": "app/self_evolve_reviewer.py",
            "purpose": "Map audit findings -> ranked patch tasks -> propose_evolution_writes()",
            "status": "proposed",
        },
        {
            "module": "scripts/run_self_audit_test.py",
            "purpose": "Scored CI gate — security/workflow finding thresholds (like file-upload runner)",
            "status": "implemented",
        },
        {
            "module": "brain/security_static_scanner.py",
            "purpose": "Reusable AST + regex scanner shared by audit and pre-commit hook",
            "status": "proposed",
        },
    ]


def _inventory_modules(paths: list[str]) -> dict[str, Any]:
    lines: dict[str, int] = {}
    functions: dict[str, int] = {}
    for rel in paths:
        content = _read_module(rel)
        if not content:
            continue
        lines[rel] = len(content.splitlines())
        try:
            tree = ast.parse(content)
            functions[rel] = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        except SyntaxError:
            functions[rel] = 0
    return {"lines": lines, "functions": functions, "file_count": len(lines)}


def _organism_snapshot() -> dict[str, Any]:
    try:
        from app.organism import get_organism

        org = get_organism()
        report = org.get_vitals_report()
        return {
            "vital": org.is_vital(),
            "lockdown_reason": report.get("lockdown_reason"),
            "api_key_required": bool(os.environ.get("AUREON_API_KEY", "").strip()),
            "organs_critical": [
                k for k, v in (report.get("organs") or {}).items() if v.get("state") == "critical"
            ],
        }
    except Exception as exc:
        return {"error": str(exc)[:200]}


def run_self_audit(*, max_files: int = 120) -> dict[str, Any]:
    """Analyze a copy of the allowed source tree — read-only, no git writes."""
    files = list_source_files(limit=max_files)
    scan_paths = list(dict.fromkeys([* _CORE_MODULES, *files]))

    security_findings: list[dict[str, Any]] = []
    ast_summaries: list[dict[str, Any]] = []
    for rel in scan_paths:
        content = _read_module(rel)
        if not content or not rel.endswith(".py"):
            continue
        security_findings.extend(_scan_file_security(rel, content))
        if rel in _CORE_MODULES:
            try:
                ast_summaries.append(analyze_file_for_task(rel, "self audit security workflow improvements"))
            except (ValueError, FileNotFoundError):
                pass

    workflow = _scan_workflow()
    inventory = _inventory_modules(scan_paths)
    improvements = _logical_improvements(inventory, security_findings, workflow)
    new_software = _new_software_proposals()

    severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    security_findings.sort(key=lambda f: severity_rank.get(f.get("severity", "info"), 5))

    return {
        "mode": "read_only_self_audit",
        "repo": repo_status(),
        "inventory": inventory,
        "organism": _organism_snapshot(),
        "security_findings": security_findings,
        "security_logic": {
            "layers": [
                "Organism vitals + lockdown on mutating routes (non-public)",
                "API key + optional replay guard (X-Timestamp, X-Nonce)",
                "Path validation on self-evolve read/write (prefix allowlist, no .env)",
                "Syntax + pytest gates before self-evolve commit",
                "Code exec rate limit (AUREON_CODE_EXEC_PER_MINUTE)",
                "Chat rate limit separate from mutating API rate limit",
            ],
            "gaps": [
                w["detail"]
                for w in workflow
                if w.get("severity") in ("medium", "high")
                and "scaffold only" not in w.get("detail", "")
            ],
        },
        "workflow_findings": workflow,
        "logical_improvements": improvements,
        "new_software": new_software,
        "core_ast": ast_summaries[: len(_CORE_MODULES)],
        "fix_plan": _build_fix_plan(improvements, workflow, security_findings),
    }


def _build_fix_plan(
    improvements: list[dict[str, Any]],
    workflow: list[dict[str, Any]],
    security: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for imp in improvements[:6]:
        plan.append({
            "step": len(plan) + 1,
            "action": imp["title"],
            "method": imp.get("how", ""),
            "priority": imp.get("priority", "medium"),
        })
    for wf in workflow:
        if wf.get("fix"):
            plan.append({
                "step": len(plan) + 1,
                "action": wf["detail"][:120],
                "method": wf["fix"],
                "priority": wf.get("severity", "medium"),
            })
    high = [s for s in security if s.get("severity") in ("high", "critical")][:5]
    for item in high:
        plan.append({
            "step": len(plan) + 1,
            "action": f"Review {item['file']}:{item['line']} — {item['detail']}",
            "method": "Confirm trusted context or replace with safe alternative",
            "priority": item["severity"],
        })
    return plan[:12]


def format_self_audit_report(audit: dict[str, Any]) -> str:
    """Human-readable markdown report for chat/API."""
    inv = audit.get("inventory", {})
    repo = audit.get("repo", {})
    org = audit.get("organism", {})
    sec = audit.get("security_findings", [])
    wf = audit.get("workflow_findings", [])
    imp = audit.get("logical_improvements", [])
    new_sw = audit.get("new_software", [])
    fix = audit.get("fix_plan", [])
    logic = audit.get("security_logic", {})

    lines = [
        "**Self-audit complete** — I read my allowed source tree (read-only; no git writes).",
        "",
        f"**Scope:** {inv.get('file_count', 0)} files · branch `{repo.get('current_branch', '?')}` · "
        f"fork `{repo.get('fork_remote', '?')}`",
        f"**Organism:** {'vital' if org.get('vital') else 'NOT VITAL'}"
        + (f" · lockdown: {org.get('lockdown_reason')}" if org.get("lockdown_reason") else "")
        + ("" if org.get("api_key_required") else " · **AUREON_API_KEY unset** (dev mode)"),
        "",
        "## Logical improvements I would make",
    ]
    for item in imp[:8]:
        lines.append(f"- **[{item.get('priority', '?').upper()}]** {item['title']} — {item.get('rationale', '')}")
        if item.get("how"):
            lines.append(f"  - *How:* {item['how']}")

    lines.extend(["", "## Security logic (what protects me today)"])
    for layer in logic.get("layers", []):
        lines.append(f"- {layer}")
    if logic.get("gaps"):
        lines.append("")
        lines.append("**Gaps:**")
        for gap in logic["gaps"][:5]:
            lines.append(f"- {gap}")

    lines.extend(["", "## Security findings (static scan)"])
    if not sec:
        lines.append("- No high-risk patterns in scanned modules.")
    else:
        shown = 0
        for f in sec:
            if shown >= 12:
                lines.append(f"- …and {len(sec) - shown} more (see API payload)")
                break
            lines.append(
                f"- **[{f.get('severity', '?').upper()}]** `{f['file']}:{f['line']}` — {f['detail']}"
            )
            shown += 1

    lines.extend(["", "## Workflow flaws"])
    for w in wf:
        fix_note = f" -> *Fix:* {w['fix']}" if w.get("fix") else ""
        lines.append(f"- [{w.get('severity', '?').upper()}] {w.get('detail', '')}{fix_note}")

    lines.extend(["", "## Fix plan (ordered)"])
    for step in fix[:10]:
        lines.append(f"{step['step']}. **[{step.get('priority', '?').upper()}]** {step['action']}")
        if step.get("method"):
            lines.append(f"   - {step['method']}")

    lines.extend(["", "## New software I would add"])
    for sw in new_sw:
        status = sw.get("status", "proposed")
        lines.append(f"- `{sw['module']}` ({status}) — {sw['purpose']}")

    lines.extend([
        "",
        "**Next steps:** `/self-audit` anytime · `POST /api/brain/self/audit` · "
        "`/evolve <specific task>` for fork patch proposals · "
        "`POST /api/brain/self/auto` for algorithmic cycle (API key required).",
    ])
    return "\n".join(lines)
