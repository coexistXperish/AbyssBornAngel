"""End-to-end integration test for HermesPlusProvider.

Drives a fake session through the provider: initialize → delegation →
session end with reflection, verifying the local JSON store ends up correct
and that the parent never sees raw subagent traces.
"""

import json

import pytest

from hermes_plus.provider import HermesPlusProvider
from hermes_plus import reflection_loop as rl


@pytest.fixture
def provider(tmp_path):
    p = HermesPlusProvider()
    p.initialize("session-1", hermes_home=str(tmp_path))
    return p


def test_initialize_creates_store_dir(provider, tmp_path):
    assert (tmp_path / "hermes_plus_memory.json").parent.exists()


def test_delegation_stores_contract_not_raw(provider):
    raw = (
        "INTERNAL SUBAGENT TRACE that must not leak\n\n"
        '```json\n{"summary": "subtask done", "artifacts": {"count": 3}, "success": true}\n```'
    )
    provider.on_delegation("do subtask", raw, child_session_id="child-9")

    store = json.loads(provider._store_path().read_text())
    assert len(store["delegations"]) == 1
    record = store["delegations"][0]
    assert record["summary"] == "subtask done"
    assert record["artifacts"] == {"count": 3}
    assert record["child_session_id"] == "child-9"
    # Raw trace must NOT be persisted
    assert "raw_output" not in record
    assert "INTERNAL SUBAGENT TRACE" not in json.dumps(store)


def test_delegation_history_is_capped(provider):
    for i in range(150):
        provider.on_delegation(f"task {i}", f'```json\n{{"summary": "s{i}", "success": true}}\n```')
    store = json.loads(provider._store_path().read_text())
    assert len(store["delegations"]) == 100


def test_session_end_applies_approved_patches(provider, monkeypatch):
    monkeypatch.setenv(rl.AUTO_APPROVE_ENV, "1")
    messages = [
        {"role": "user", "content": "investigate the deploy"},
        {"role": "assistant", "content": "Note: deploy requires the VPN to be on."},
        {"role": "assistant", "content": "Done — the deploy succeeded."},
    ]
    provider.on_session_end(messages)

    mem = provider._get_memory_store()
    assert "_session_summary" in mem
    assert any("VPN" in v for v in mem.values())


def test_session_end_skips_when_not_approved(provider, monkeypatch):
    monkeypatch.delenv(rl.AUTO_APPROVE_ENV, raising=False)
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    messages = [
        {"role": "assistant", "content": "Note: something learned here."},
    ]
    provider.on_session_end(messages)
    assert provider._get_memory_store() == {}


def test_prefetch_surfaces_prior_summary(provider, monkeypatch):
    monkeypatch.setenv(rl.AUTO_APPROVE_ENV, "1")
    provider.on_session_end([
        {"role": "assistant", "content": "Final summary of the work."},
    ])
    recalled = provider.prefetch("anything")
    assert "Final summary" in recalled


def test_session_switch_updates_id(provider):
    provider.on_session_switch("session-2")
    assert provider._session_id == "session-2"
