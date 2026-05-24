"""Tests for hermes_plus.output_contracts — contract enforcement + parent isolation."""

from hermes_plus import output_contracts as oc


# ---------------------------------------------------------------------------
# enforce_contract
# ---------------------------------------------------------------------------

def test_json_contract_is_parsed():
    raw = (
        "Lots of reasoning trace here...\n\n"
        '```json\n'
        '{"summary": "Built the widget", "artifacts": {"path": "/tmp/w"}, '
        '"success": true}\n'
        '```'
    )
    result = oc.enforce_contract("build a widget", raw)
    assert result.summary == "Built the widget"
    assert result.artifacts == {"path": "/tmp/w"}
    assert result.success is True


def test_bare_json_contract_is_parsed():
    raw = 'done. {"summary": "ok done", "success": false}'
    result = oc.enforce_contract("task", raw)
    assert result.summary == "ok done"
    assert result.success is False


def test_heuristic_fallback_without_json():
    raw = "First paragraph.\n\nThe final answer is that the build succeeded cleanly."
    result = oc.enforce_contract("task", raw)
    assert "build succeeded" in result.summary


def test_raw_output_is_truncated():
    raw = "x" * 10_000
    result = oc.enforce_contract("task", raw, max_raw_chars=100)
    assert len(result.raw_output) < 200  # 100 + truncation note
    assert "truncated" in result.raw_output


def test_summary_is_capped():
    long_summary = "word " * 500
    raw = f'```json\n{{"summary": "{long_summary}", "success": true}}\n```'
    result = oc.enforce_contract("task", raw, max_summary_chars=50)
    assert len(result.summary) <= 50


# ---------------------------------------------------------------------------
# parent context isolation
# ---------------------------------------------------------------------------

def test_parent_context_excludes_raw_trace():
    raw = "SECRET INTERNAL TRACE\n\n" + ("noise " * 1000) + '\n\n```json\n{"summary": "clean summary", "success": true}\n```'
    result = oc.enforce_contract("task", raw)
    parent_view = result.to_parent_context()
    assert "clean summary" in parent_view
    assert "SECRET INTERNAL TRACE" not in parent_view
    assert "noise noise" not in parent_view


def test_parent_context_flags_failure():
    result = oc.SubagentResult(task="t", summary="s", success=False)
    assert "FAILED" in result.to_parent_context()


# ---------------------------------------------------------------------------
# mixin
# ---------------------------------------------------------------------------

def test_mixin_intercepts_delegation():
    captured = {}

    class P(oc.OutputContractMemoryHook):
        def on_delegation_contract(self, contract, *, child_session_id="", **kw):
            captured["contract"] = contract
            captured["sid"] = child_session_id

    p = P()
    raw = '```json\n{"summary": "did it", "success": true}\n```'
    p.on_delegation("the task", raw, child_session_id="child-123")

    assert captured["contract"].summary == "did it"
    assert captured["sid"] == "child-123"
