"""System fallback message tests."""

from __future__ import annotations

from brain.system_messages import FALLBACK_TIMEOUT


def test_timeout_message_does_not_blame_auto_learn():
    text = FALLBACK_TIMEOUT.lower()
    assert "auto-learn" not in text
    assert "compute time limit" in text

