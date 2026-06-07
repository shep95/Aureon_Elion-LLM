"""Intent classifier tests."""

from __future__ import annotations

from brain.agent_loop import classify_intent, run_agent_loop


def test_classify_intent_knowledge_question():
    assert classify_intent("What type of algorithm is an artificial intelligence algorithm") == "KNOWLEDGE"


def test_classify_intent_code_command():
    assert classify_intent("write a python function to add two numbers") == "CODE"
    assert classify_intent("fix this code: print(") == "CODE"


def test_classify_intent_evolve_command():
    assert classify_intent("change your code to improve routing") == "EVOLVE"


def test_agent_loop_blocks_evolve_intent_from_tool_execution():
    result = run_agent_loop("change your code to improve routing")
    assert result["context"]["intent"] == "EVOLVE"
    assert result["steps"][0]["tool"] == "intent_classifier"
    assert result["citations"] == []

