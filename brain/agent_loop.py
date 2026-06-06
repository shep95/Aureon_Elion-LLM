"""Agent tool loop — multi-step plans with search, calculate, verify gates."""

from __future__ import annotations

import re
from typing import Any, Callable

from brain.brain_classifiers import classify_moe
from brain.deterministic_qa import try_arithmetic_answer
from brain.predict_engine import predict_with_steps
from brain.vector_rag import retrieve_with_citations
from app.session_memory import history_as_context

ToolFn = Callable[..., dict[str, Any]]


def _tool_rag_search(query: str, *, ctx: dict[str, Any]) -> dict[str, Any]:
    context, hits, citations = retrieve_with_citations(query, top_k=6)
    ctx["rag_context"] = context
    ctx["citations"] = list(citations[:5])
    ctx["rag_hits"] = len(hits)
    return {
        "tool": "rag_search",
        "ok": bool(hits),
        "hits": len(hits),
        "context_words": len(context.split()) if context else 0,
        "citations": ctx["citations"],
        "preview": hits[0].snippet(200) if hits else "",
    }


def _tool_calculate(expression: str) -> dict[str, Any]:
    result = try_arithmetic_answer(f"what is {expression.strip()}")
    if not result:
        result = try_arithmetic_answer(expression.strip())
    return {
        "tool": "calculate",
        "ok": result is not None,
        "expression": expression,
        "answer": result["answer"] if result else None,
    }


def _tool_classify(text: str, *, ctx: dict[str, Any]) -> dict[str, Any]:
    result = classify_moe(text)
    ctx["classification"] = result
    return {"tool": "classify", "ok": result is not None, "classification": result}


def _tool_predict(question: str, *, ctx: dict[str, Any], conversation_context: str = "") -> dict[str, Any]:
    rag_block = ctx.get("rag_context") or ""
    conv = f"{conversation_context} {rag_block}".strip()
    pred = predict_with_steps(question, conversation_context=conv)
    if not pred:
        return {"tool": "predict", "ok": False}
    ctx["predict_answer"] = pred.get("answer")
    ctx["predict_confidence"] = pred.get("confidence")
    if pred.get("citations"):
        ctx["citations"] = list(pred["citations"])
    return {
        "tool": "predict",
        "ok": not pred.get("abstained", False),
        "confidence": pred.get("confidence"),
        "abstained": pred.get("abstained", False),
        "answer": pred.get("answer"),
    }


def _tool_verify(citations: list[dict[str, Any]], answer: str) -> dict[str, Any]:
    grounded = bool(citations) and len(answer.strip()) >= 3
    has_hash = all(c.get("content_hash") for c in citations) if citations else False
    return {
        "tool": "verify",
        "ok": grounded and has_hash,
        "grounded": grounded,
        "citation_count": len(citations),
        "verified_hashes": has_hash,
    }


def _extract_math_expression(text: str) -> str | None:
    match = re.search(r"([\d\s+\-*/().%]+)", text)
    if not match:
        return None
    expr = match.group(1).strip()
    return expr if re.search(r"[+\-*/%]", expr) else None


def is_agent_task(text: str) -> bool:
    """True for multi-step or explicit agent requests."""
    lower = text.strip().lower()
    if lower.startswith("/agent"):
        return True
    triggers = (
        "first search",
        "then calculate",
        "step by step",
        "multi-step",
        "use tools",
        "search and",
        "find and explain",
    )
    return any(t in lower for t in triggers)


def run_agent_loop(
    question: str,
    *,
    max_steps: int = 5,
    session_id: str | None = None,
) -> dict[str, Any]:
    """
    Plan → execute tools → verify → synthesize answer.
    Shared ctx threads RAG/classify/predict results across steps.
    """
    from brain.combinatorial_creation import is_creation_request, plan_combinatorial_creation

    q = question.strip()
    if q.lower().startswith("/agent"):
        q = q[6:].strip(" :")

    ctx: dict[str, Any] = {
        "question": q,
        "citations": [],
        "rag_context": "",
        "classification": None,
        "predict_answer": None,
    }
    conv = history_as_context(session_id)
    steps: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    answer: str | None = None
    confidence = 0.0

    if is_creation_request(q):
        plan = plan_combinatorial_creation(q)
        from brain.combinatorial_creation import format_creation_reply

        comb = plan.to_dict()
        steps.append({"tool": "combinatorial_creation", "ok": comb["valid"], "plan": comb})
        answer = format_creation_reply(plan)
        citations = [p.citation for p in plan.precursors if p.citation]
        return {
            "answer": answer,
            "plan": [s["tool"] for s in steps],
            "steps": steps,
            "citations": citations,
            "confidence": 0.85 if comb["valid"] else 0.0,
            "agent": True,
            "combinatorial": comb,
            "max_steps": max_steps,
            "context": {"creation_doctrine": True},
        }

    expr = _extract_math_expression(q)

    rag = _tool_rag_search(q, ctx=ctx)
    steps.append(rag)
    citations = list(ctx.get("citations") or [])

    arith = try_arithmetic_answer(q)
    if arith:
        steps.append(
            {
                "tool": "calculate",
                "ok": True,
                "expression": q,
                "answer": arith["answer"],
            }
        )
        answer = str(arith["answer"])
    elif expr and len(steps) < max_steps:
        calc = _tool_calculate(expr)
        steps.append(calc)
        if calc.get("ok"):
            answer = str(calc["answer"])

    if len(steps) < max_steps:
        clf = _tool_classify(q, ctx=ctx)
        steps.append(clf)

    if not answer and len(steps) < max_steps:
        pred_step = _tool_predict(q, ctx=ctx, conversation_context=conv)
        steps.append(pred_step)
        if pred_step.get("answer"):
            answer = str(pred_step["answer"])
            confidence = float(pred_step.get("confidence") or 0.0)
            citations = list(ctx.get("citations") or citations)

    if not answer:
        answer = "I couldn't complete the agent plan with grounded tools."

    if len(steps) < max_steps:
        verify = _tool_verify(citations, answer)
        steps.append(verify)
        if (
            not verify.get("ok")
            and not try_arithmetic_answer(q)
            and not answer.replace(".", "", 1).isdigit()
        ):
            from brain.system_messages import FALLBACK_TRAINING

            answer = FALLBACK_TRAINING

    return {
        "answer": answer,
        "plan": [s["tool"] for s in steps],
        "steps": steps,
        "citations": citations,
        "confidence": confidence,
        "agent": True,
        "max_steps": max_steps,
        "context": {
            "rag_hits": ctx.get("rag_hits", 0),
            "classification": ctx.get("classification"),
        },
    }
