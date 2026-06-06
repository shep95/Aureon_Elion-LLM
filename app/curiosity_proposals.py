"""Curiosity proposal queue — human approval before GitHub/Railway deploy."""

from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.self_evolve import ROOT

_lock = threading.Lock()
PROPOSALS_DIR = ROOT / "data" / "curiosity"
PROPOSALS_PATH = PROPOSALS_DIR / "proposals.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_all() -> list[dict[str, Any]]:
    if not PROPOSALS_PATH.is_file():
        return []
    items: list[dict[str, Any]] = []
    for line in PROPOSALS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items


def _save_all(items: list[dict[str, Any]]) -> None:
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    PROPOSALS_PATH.write_text(
        "\n".join(json.dumps(i, default=str) for i in items) + ("\n" if items else ""),
        encoding="utf-8",
    )


def _append(record: dict[str, Any]) -> None:
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    with PROPOSALS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def create_proposal(research_payload: dict[str, Any]) -> dict[str, Any]:
    """Persist a curiosity research result awaiting sandbox + human approval."""
    if not research_payload.get("ok"):
        raise ValueError(research_payload.get("error", "research failed"))

    proposal_id = str(research_payload.get("proposal_id") or uuid.uuid4())
    record: dict[str, Any] = {
        "id": proposal_id,
        "created_at": _now(),
        "status": "pending_sandbox",
        "snapshot": research_payload.get("snapshot", {}),
        "self_intro": research_payload.get("self_intro", ""),
        "curiosity_reflection": research_payload.get("curiosity_reflection", ""),
        "research": research_payload.get("research", []),
        "advancements": research_payload.get("advancements", []),
        "web_search_used": research_payload.get("web_search_used", False),
        "requires_human_approval": research_payload.get("requires_human_approval", True),
        "sandbox_path": None,
        "sandbox_files": [],
        "branch": None,
        "railway_section": None,
        "github_branch": None,
        "deploy": {},
    }
    with _lock:
        _append(record)
    return record


def get_proposal(proposal_id: str) -> dict[str, Any] | None:
    pid = proposal_id.strip()
    for item in reversed(_load_all()):
        if item.get("id") == pid:
            return item
    return None


def list_proposals(*, status: str | None = None, limit: int = 50) -> dict[str, Any]:
    items = _load_all()
    if status:
        items = [i for i in items if i.get("status") == status]
    items = list(reversed(items))[:limit]
    pending = sum(1 for i in _load_all() if i.get("status") in (
        "pending_sandbox",
        "sandbox_ready",
        "pending_approval",
    ))
    return {"proposals": items, "count": len(items), "pending": pending}


def _update_proposal(proposal_id: str, **fields: Any) -> dict[str, Any] | None:
    with _lock:
        items = _load_all()
        updated: dict[str, Any] | None = None
        for i, item in enumerate(items):
            if item.get("id") == proposal_id:
                items[i] = {**item, **fields, "updated_at": _now()}
                updated = items[i]
                break
        if updated:
            _save_all(items)
        return updated


def mark_sandbox_ready(
    proposal_id: str,
    *,
    sandbox_path: str,
    sandbox_files: list[str],
    railway_section: str | None = None,
) -> dict[str, Any] | None:
    return _update_proposal(
        proposal_id,
        status="pending_approval",
        sandbox_path=sandbox_path,
        sandbox_files=sandbox_files,
        railway_section=railway_section,
    )


def approve_proposal(
    proposal_id: str,
    *,
    approved: bool = True,
    reviewer: str = "human",
) -> dict[str, Any]:
    proposal = get_proposal(proposal_id)
    if not proposal:
        return {"ok": False, "error": "proposal_not_found"}
    if proposal.get("status") not in ("pending_approval", "sandbox_ready"):
        return {
            "ok": False,
            "error": "invalid_status",
            "status": proposal.get("status"),
        }
    if not approved:
        updated = _update_proposal(
            proposal_id,
            status="rejected",
            rejected_at=_now(),
            reviewer=reviewer,
        )
        return {"ok": True, "action": "rejected", "proposal": updated}

    updated = _update_proposal(
        proposal_id,
        status="approved",
        approved_at=_now(),
        reviewer=reviewer,
    )
    return {"ok": True, "action": "approved", "proposal": updated}


def mark_deployed(proposal_id: str, deploy: dict[str, Any]) -> dict[str, Any] | None:
    return _update_proposal(
        proposal_id,
        status="deployed",
        deployed_at=_now(),
        deploy=deploy,
        branch=deploy.get("branch"),
        github_branch=deploy.get("github_branch"),
    )


def record_audit(proposal_id: str, event: str, detail: str = "") -> None:
    try:
        from app.audit import get_audit_log

        get_audit_log().record(
            "mutating_request",
            detail=f"curiosity:{event} id={proposal_id} {detail}"[:500],
        )
    except Exception:
        pass
