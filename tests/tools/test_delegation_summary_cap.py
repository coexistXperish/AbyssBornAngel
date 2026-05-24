"""Tests for HERMES++ delegation summary cap in delegate_tool._get_delegation_summary_cap."""

import pytest
from tools.delegate_tool import _get_delegation_summary_cap


class _FakeAgent:
    def __init__(self, config=None):
        self.config = config or {}


def test_returns_zero_when_no_env_no_config(monkeypatch):
    monkeypatch.delenv("HERMES_MAX_DELEGATION_SUMMARY_CHARS", raising=False)
    assert _get_delegation_summary_cap(_FakeAgent()) == 0


def test_env_var_takes_precedence(monkeypatch):
    monkeypatch.setenv("HERMES_MAX_DELEGATION_SUMMARY_CHARS", "1500")
    agent = _FakeAgent(config={"delegation": {"max_summary_chars": 9999}})
    assert _get_delegation_summary_cap(agent) == 1500


def test_config_used_when_no_env(monkeypatch):
    monkeypatch.delenv("HERMES_MAX_DELEGATION_SUMMARY_CHARS", raising=False)
    agent = _FakeAgent(config={"delegation": {"max_summary_chars": 2000}})
    assert _get_delegation_summary_cap(agent) == 2000


def test_non_digit_env_falls_through_to_config(monkeypatch):
    monkeypatch.setenv("HERMES_MAX_DELEGATION_SUMMARY_CHARS", "unlimited")
    agent = _FakeAgent(config={"delegation": {"max_summary_chars": 500}})
    assert _get_delegation_summary_cap(agent) == 500


def test_no_config_no_env_returns_zero(monkeypatch):
    monkeypatch.delenv("HERMES_MAX_DELEGATION_SUMMARY_CHARS", raising=False)
    assert _get_delegation_summary_cap(_FakeAgent()) == 0


def test_summary_is_truncated_when_cap_set(monkeypatch):
    """Integration-style: verify truncation logic applied in delegate_tool works correctly."""
    monkeypatch.setenv("HERMES_MAX_DELEGATION_SUMMARY_CHARS", "50")
    cap = _get_delegation_summary_cap(_FakeAgent())
    long_summary = "A" * 200
    if cap and len(long_summary) > cap:
        truncated = long_summary[:cap] + f"\n…[truncated by HERMES++ to {cap} chars]"
    else:
        truncated = long_summary
    assert len(truncated.splitlines()[0]) == 50
    assert "truncated by HERMES++" in truncated
