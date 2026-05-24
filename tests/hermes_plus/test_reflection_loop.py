"""Tests for hermes_plus.reflection_loop — extraction, patch proposal, gate."""

import pytest

from hermes_plus import reflection_loop as rl


# ---------------------------------------------------------------------------
# extract_session_learnings
# ---------------------------------------------------------------------------

def test_extracts_summary_from_last_assistant_message():
    messages = [
        {"role": "user", "content": "do a thing"},
        {"role": "assistant", "content": "I completed the thing successfully."},
    ]
    result = rl.extract_session_learnings(messages)
    assert "completed the thing" in result.session_summary


def test_extracts_learned_markers():
    messages = [
        {"role": "assistant", "content": "Note: the API rate-limits at 100 req/min."},
        {"role": "assistant", "content": "Remember: deploy needs sudo."},
        {"role": "assistant", "content": "just some normal text"},
    ]
    result = rl.extract_session_learnings(messages)
    assert len(result.learned) == 2


def test_handles_content_block_format():
    messages = [
        {"role": "assistant", "content": [{"type": "text", "text": "block summary here"}]},
    ]
    result = rl.extract_session_learnings(messages)
    assert "block summary" in result.session_summary


# ---------------------------------------------------------------------------
# propose_patches
# ---------------------------------------------------------------------------

def test_new_learning_becomes_add_patch():
    reflection = rl.ReflectionResult(
        session_summary="did stuff",
        learned=["Note: the cache key includes the user id"],
    )
    patches = rl.propose_patches(reflection, existing_memory={})
    add_patches = [p for p in patches if p.action == "add" and p.key != "_session_summary"]
    assert len(add_patches) == 1


def test_changed_learning_becomes_update_patch():
    key = rl._derive_key("Note: the cache ttl is 60s")
    reflection = rl.ReflectionResult(
        session_summary="x",
        learned=["Note: the cache ttl is 60s"],
    )
    patches = rl.propose_patches(reflection, existing_memory={key: "old value"})
    assert any(p.action == "update" and p.key == key for p in patches)


def test_session_summary_always_proposed():
    reflection = rl.ReflectionResult(session_summary="summary text", learned=[])
    patches = rl.propose_patches(reflection, existing_memory={})
    assert any(p.key == "_session_summary" for p in patches)


# ---------------------------------------------------------------------------
# confirmation gate
# ---------------------------------------------------------------------------

def test_auto_approve_env(monkeypatch):
    monkeypatch.setenv(rl.AUTO_APPROVE_ENV, "1")
    patches = [rl.MemoryPatch(action="add", key="k", new_value="v")]
    approved = rl.confirm_patches(patches)
    assert approved == patches


def test_empty_patches_returns_empty(monkeypatch):
    monkeypatch.delenv(rl.AUTO_APPROVE_ENV, raising=False)
    assert rl.confirm_patches([]) == []


def test_cli_gate_respects_yes(monkeypatch):
    monkeypatch.delenv(rl.AUTO_APPROVE_ENV, raising=False)
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    patches = [rl.MemoryPatch(action="add", key="k", new_value="v")]
    assert len(rl.confirm_patches(patches)) == 1


def test_cli_gate_respects_no(monkeypatch):
    monkeypatch.delenv(rl.AUTO_APPROVE_ENV, raising=False)
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    patches = [rl.MemoryPatch(action="add", key="k", new_value="v")]
    assert rl.confirm_patches(patches) == []


# ---------------------------------------------------------------------------
# MemoryPatch.diff_text
# ---------------------------------------------------------------------------

def test_diff_text_add():
    p = rl.MemoryPatch(action="add", key="k", new_value="hello")
    assert p.diff_text().startswith("+")


def test_diff_text_update_shows_unified_diff():
    p = rl.MemoryPatch(action="update", key="k", old_value="line one\n", new_value="line two\n")
    text = p.diff_text()
    assert "line one" in text and "line two" in text
