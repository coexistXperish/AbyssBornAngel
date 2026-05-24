"""Tests for hermes_plus.reflection_loop — extraction, patch proposal, gate."""

import json
import unittest.mock as mock

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


# ---------------------------------------------------------------------------
# Webhook + email notification paths
# ---------------------------------------------------------------------------

def test_webhook_called_when_env_set(monkeypatch):
    monkeypatch.setenv(rl.WEBHOOK_URL_ENV, "http://localhost:5678/webhook/test")
    monkeypatch.delenv(rl.AUTO_APPROVE_ENV, raising=False)
    patches = [rl.MemoryPatch(action="add", key="k", new_value="v")]

    with mock.patch("urllib.request.urlopen") as mock_open:
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = mock.Mock(return_value=False)
        rl.confirm_patches(patches)

    mock_open.assert_called_once()
    req = mock_open.call_args[0][0]
    body = json.loads(req.data.decode())
    assert "patches" in body


def test_webhook_failure_does_not_raise(monkeypatch):
    monkeypatch.setenv(rl.WEBHOOK_URL_ENV, "http://localhost:5678/webhook/test")
    monkeypatch.delenv(rl.AUTO_APPROVE_ENV, raising=False)
    patches = [rl.MemoryPatch(action="add", key="k", new_value="v")]

    with mock.patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        result = rl.confirm_patches(patches)

    assert result == patches


def test_webhook_auto_approves_all(monkeypatch):
    monkeypatch.setenv(rl.WEBHOOK_URL_ENV, "http://localhost:5678/webhook/test")
    monkeypatch.delenv(rl.AUTO_APPROVE_ENV, raising=False)
    patches = [
        rl.MemoryPatch(action="add", key="k1", new_value="v1"),
        rl.MemoryPatch(action="add", key="k2", new_value="v2"),
    ]

    with mock.patch("urllib.request.urlopen") as mock_open:
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = mock.Mock(return_value=False)
        result = rl.confirm_patches(patches)

    assert result == patches


def test_email_called_when_env_set(monkeypatch):
    monkeypatch.delenv(rl.AUTO_APPROVE_ENV, raising=False)
    monkeypatch.delenv(rl.WEBHOOK_URL_ENV, raising=False)
    monkeypatch.setenv(rl.EMAIL_TO_ENV, "test@example.com")
    monkeypatch.setenv(rl.SMTP_HOST_ENV, "localhost")
    patches = [rl.MemoryPatch(action="add", key="k", new_value="v")]

    with mock.patch("smtplib.SMTP") as mock_smtp:
        mock_smtp.return_value.__enter__ = lambda s: s
        mock_smtp.return_value.__exit__ = mock.Mock(return_value=False)
        result = rl.confirm_patches(patches)

    mock_smtp.assert_called_once()
    assert result == patches


def test_email_failure_does_not_raise(monkeypatch):
    monkeypatch.delenv(rl.AUTO_APPROVE_ENV, raising=False)
    monkeypatch.delenv(rl.WEBHOOK_URL_ENV, raising=False)
    monkeypatch.setenv(rl.EMAIL_TO_ENV, "test@example.com")
    monkeypatch.setenv(rl.SMTP_HOST_ENV, "localhost")
    patches = [rl.MemoryPatch(action="add", key="k", new_value="v")]

    with mock.patch("smtplib.SMTP", side_effect=OSError("smtp down")):
        result = rl.confirm_patches(patches)

    assert result == patches


def test_cli_fallback_when_no_env(monkeypatch):
    monkeypatch.delenv(rl.AUTO_APPROVE_ENV, raising=False)
    monkeypatch.delenv(rl.WEBHOOK_URL_ENV, raising=False)
    monkeypatch.delenv(rl.EMAIL_TO_ENV, raising=False)
    monkeypatch.delenv(rl.SMTP_HOST_ENV, raising=False)
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    patches = [rl.MemoryPatch(action="add", key="k", new_value="v")]
    result = rl.confirm_patches(patches)
    assert result == patches


def test_session_id_in_webhook_payload(monkeypatch):
    monkeypatch.setenv(rl.WEBHOOK_URL_ENV, "http://localhost:5678/webhook/test")
    monkeypatch.delenv(rl.AUTO_APPROVE_ENV, raising=False)
    patches = [rl.MemoryPatch(action="add", key="k", new_value="v")]

    with mock.patch("urllib.request.urlopen") as mock_open:
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = mock.Mock(return_value=False)
        rl.confirm_patches(patches, session_id="sess-abc-123")

    req = mock_open.call_args[0][0]
    body = json.loads(req.data.decode())
    assert body["session_id"] == "sess-abc-123"


def test_priority_order_webhook_beats_email(monkeypatch):
    monkeypatch.delenv(rl.AUTO_APPROVE_ENV, raising=False)
    monkeypatch.setenv(rl.WEBHOOK_URL_ENV, "http://localhost:5678/webhook/test")
    monkeypatch.setenv(rl.EMAIL_TO_ENV, "test@example.com")
    monkeypatch.setenv(rl.SMTP_HOST_ENV, "localhost")
    patches = [rl.MemoryPatch(action="add", key="k", new_value="v")]

    with mock.patch("urllib.request.urlopen") as mock_urlopen, \
         mock.patch("smtplib.SMTP") as mock_smtp:
        mock_urlopen.return_value.__enter__ = lambda s: s
        mock_urlopen.return_value.__exit__ = mock.Mock(return_value=False)
        rl.confirm_patches(patches)

    mock_urlopen.assert_called_once()
    mock_smtp.assert_not_called()
