"""Algorithmic self-evolve brain — AST analysis + predict + code_master + verification gates."""

from __future__ import annotations

import ast
import logging
import re
import time
from typing import Any

from app.self_evolve import analyze_file_for_task, read_source
from brain.code_evaluator import check_syntax, extract_python_code

logger = logging.getLogger(__name__)

_CODE_TASK_WORDS = frozenset(
    {"function", "implement", "fix", "module", "code", "algorithm", "helper", "def", "class"}
)
_NEW_MODULE_WORDS = frozenset({"new module", "new file", "create module", "add module"})
_STAMP_MARKER = "# SOLIA auto-evolve"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower())[:32].strip("-") or "auto"


def classify_evolve_task(task: str) -> str:
    """Route task to patch strategy — code, new_module, routing, or annotate."""
    task_l = task.lower()
    if any(k in task_l for k in _NEW_MODULE_WORDS):
        return "new_module"
    if any(w in task_l.split() or w in task_l for w in _CODE_TASK_WORDS):
        return "code"
    if any(k in task_l for k in ("routing", "route", "handler", "reply", "classify")):
        return "routing"
    return "annotate"


def reason_about_task(task: str, analysis: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run predict brain over AST context — algorithm layer, not external LLM."""
    funcs = (analysis or {}).get("task_relevant_functions") or []
    path = (analysis or {}).get("path", "")
    context = f"file {path} functions {', '.join(funcs[:6])} " if funcs else ""
    prompt = f"{context}self evolve task {task.strip().lower()} think therefore answer"
    try:
        from brain.predict_engine import predict_with_steps

        result = predict_with_steps(prompt, force=True) or {}
    except Exception as exc:
        logger.debug("predict reasoning skipped: %s", exc)
        result = {}
    return {
        "hint": (result.get("answer") or "").strip()[:400],
        "confidence": float(result.get("confidence") or 0.0),
        "abstained": bool(result.get("abstained")),
        "model": result.get("model", "stacked_attention_lm"),
    }


def _stamp(task: str) -> str:
    return f"\n\n{_STAMP_MARKER} ({time.strftime('%Y-%m-%d')})\n# Task: {task[:200]}\n"


def _merge_parses(source: str, addition: str) -> bool:
    try:
        ast.parse(source.rstrip() + "\n\n" + addition.strip() + "\n")
        return True
    except SyntaxError:
        return False


def _generate_verified_code(task: str, *, path: str = "", use_predict: bool = True) -> dict[str, Any]:
    from brain.code_master import generate_master_code

    prompt = f"write python code to {task}"
    if path:
        prompt += f" for module {path}"
    predict_fn = None
    if use_predict:
        from brain.predict_engine import predict_with_steps

        predict_fn = lambda q: predict_with_steps(q, force=True)
    return generate_master_code(prompt, predict_fn=predict_fn)


def _append_code_patch(content: str, task: str, path: str) -> dict[str, Any] | None:
    if _STAMP_MARKER in content:
        return None
    result = _generate_verified_code(task, path=path, use_predict=True)
    code = extract_python_code(result.get("answer") or "")
    ev = result.get("code_eval") or {}
    if not code or "def " not in code or not ev.get("syntax_valid"):
        return None
    addition = _stamp(task) + "\n" + code + "\n"
    if not _merge_parses(content, addition):
        return None
    return {
        "content": content.rstrip() + addition,
        "method": result.get("method", "code_master"),
        "confidence": float(result.get("confidence") or 0.0),
        "code_eval": ev,
    }


def _enrich_docstrings(content: str, analysis: dict[str, Any], hint: str, task: str) -> dict[str, Any] | None:
    """Append evolve hint to docstrings of task-relevant functions (AST rewrite)."""
    targets = set(analysis.get("task_relevant_functions") or [])
    if not targets and not hint:
        return None
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None

    note = f" [evolve:{task[:80]}]"
    if hint:
        note += f" hint:{hint[:120]}"
    changed = False

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in targets:
            continue
        existing = ast.get_docstring(node) or ""
        if note.strip() in existing:
            continue
        new_doc = (existing + note).strip()
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            node.body[0].value.value = new_doc
        else:
            node.body.insert(0, ast.Expr(value=ast.Constant(value=new_doc)))
        changed = True

    if not changed:
        return None
    try:
        new_content = ast.unparse(tree)
    except Exception:
        return None
    if not check_syntax(new_content).get("valid"):
        return None
    return {"content": new_content + "\n", "method": "docstring_enrich", "confidence": 0.55}


def propose_file_patch(path: str, task: str, *, strategy: str | None = None) -> dict[str, Any]:
    """Algorithmic patch proposal for one file — returns original if no safe change."""
    strategy = strategy or classify_evolve_task(task)
    try:
        content = read_source(path)["content"]
    except (ValueError, FileNotFoundError) as exc:
        return {"path": path, "patched": False, "reason": str(exc)}

    analysis = analyze_file_for_task(path, task)
    reasoning = reason_about_task(task, analysis)

    proposal: dict[str, Any] = {
        "path": path,
        "strategy": strategy,
        "analysis": analysis,
        "reasoning": reasoning,
        "patched": False,
    }

    if path.endswith(".md"):
        if f"SOLIA auto-evolve: {task[:200]}" in content:
            proposal["reason"] = "already stamped"
            return proposal
        proposal["content"] = content.rstrip() + f"\n\n---\n**SOLIA auto-evolve:** {task[:200]}\n"
        proposal["patched"] = True
        proposal["method"] = "markdown_stamp"
        return proposal

    if not path.endswith(".py"):
        proposal["reason"] = "unsupported file type"
        return proposal

    patch: dict[str, Any] | None = None
    if strategy == "code":
        patch = _append_code_patch(content, task, path)
    elif strategy in ("routing", "annotate"):
        patch = _enrich_docstrings(content, analysis, reasoning.get("hint", ""), task)
        if not patch:
            patch = {"content": content.rstrip() + _stamp(task), "method": "annotate_stamp", "confidence": 0.4}
    else:
        patch = {"content": content.rstrip() + _stamp(task), "method": "annotate_stamp", "confidence": 0.4}

    if not patch:
        proposal["reason"] = "no verified patch produced"
        return proposal

    new_content = patch["content"]
    syntax = check_syntax(new_content)
    if not syntax.get("valid"):
        proposal["reason"] = f"syntax invalid after patch: {syntax.get('error')}"
        return proposal

    proposal.update({
        "patched": True,
        "content": new_content,
        "method": patch.get("method"),
        "confidence": patch.get("confidence", reasoning.get("confidence", 0.0)),
        "code_eval": patch.get("code_eval"),
    })
    return proposal


def propose_new_module(task: str) -> dict[str, Any] | None:
    slug = _slug(task)
    path = f"brain/auto_{slug}.py"
    result = _generate_verified_code(f"write a python module for {task}", use_predict=True)
    code = extract_python_code(result.get("answer") or "")
    ev = result.get("code_eval") or {}
    if not code or "def " not in code or not ev.get("syntax_valid"):
        code = (
            f'"""SOLIA algorithmic module — {task}."""\n\n'
            f"def run() -> str:\n"
            f'    """Entry point for auto module {slug}."""\n'
            f'    return "SOLIA auto module {slug}"\n'
        )
    elif not code.startswith('"""'):
        code = f'"""SOLIA algorithmic module — {task}."""\n\n' + code
    return {
        "path": path,
        "content": code,
        "patched": True,
        "method": result.get("method", "fallback_stub"),
        "confidence": float(result.get("confidence") or 0.5),
        "code_eval": ev,
    }


def propose_evolution_writes(task: str, plan: dict[str, Any] | None = None, *, max_files: int = 3) -> dict[str, Any]:
    """Full algorithmic proposal — analyze, reason, patch, verify before write."""
    from app.self_evolve import plan_evolution

    plan = plan or plan_evolution(task)
    strategy = classify_evolve_task(task)
    proposals: list[dict[str, Any]] = []

    if strategy == "new_module":
        mod = propose_new_module(task)
        if mod:
            proposals.append(mod)

    for rel_path in plan.get("suggested_files", [])[:max_files]:
        if any(p["path"] == rel_path for p in proposals):
            continue
        proposals.append(propose_file_patch(rel_path, task, strategy=strategy))

    writes: list[dict[str, str]] = []
    for prop in proposals:
        if prop.get("patched") and prop.get("content") is not None:
            writes.append({"path": prop["path"], "content": prop["content"]})

    return {
        "task": task,
        "strategy": strategy,
        "plan": plan,
        "proposals": proposals,
        "writes": writes,
        "brain": "algorithmic (predict + code_master + AST)",
    }
