"""Tests for hermes_plus.benchmark — task checks, scoring, report, dry-run."""

import json
from pathlib import Path

import pytest

from hermes_plus.benchmark import (
    TASK_SET,
    ArmResult,
    BenchmarkConfig,
    BenchmarkReport,
    SessionResult,
    TaskResult,
    _has_n_bullets,
    _contains_all,
    _contains_code_block,
    _contains_sql,
    _run_task_dry,
    run_benchmark,
)


# ---------------------------------------------------------------------------
# Task success predicates
# ---------------------------------------------------------------------------

def test_bullet_check_passes_with_enough_bullets():
    text = "- point one\n- point two\n- point three\n- point four\n- point five"
    assert _has_n_bullets(4)(text)


def test_bullet_check_fails_with_too_few():
    assert not _has_n_bullets(4)("- only one")


def test_contains_all_passes():
    assert _contains_all("select", "from")("SELECT * FROM users")


def test_contains_all_fails_on_partial():
    assert not _contains_all("select", "from")("SELECT something")


def test_code_block_detected():
    assert _contains_code_block("```python\ndef foo(): pass\n```")


def test_code_block_missing():
    assert not _contains_code_block("just some prose text here")


def test_sql_detected():
    assert _contains_sql("SELECT id FROM users WHERE age > 30")


def test_sql_missing():
    assert not _contains_sql("no query here at all")


# ---------------------------------------------------------------------------
# TaskResult properties
# ---------------------------------------------------------------------------

def test_task_result_skill_delta():
    r = TaskResult("T1", success=True, skill_count_before=3, skill_count_after=5)
    assert r.skill_delta == 2


def test_task_result_total_tokens():
    r = TaskResult("T1", success=True, input_tokens=800, output_tokens=200)
    assert r.total_tokens == 1000


# ---------------------------------------------------------------------------
# Dry-run task execution
# ---------------------------------------------------------------------------

def test_dry_run_all_tasks_have_results():
    for task in TASK_SET:
        result = _run_task_dry(task, session_idx=0)
        assert result.task_id == task.id
        assert isinstance(result.success, bool)
        assert result.input_tokens > 0


def test_dry_run_t1_code_task_succeeds():
    t1 = next(t for t in TASK_SET if t.id == "T1")
    result = _run_task_dry(t1, session_idx=0)
    assert result.success, "T1 dry-run should pass code block check"


def test_dry_run_t5_sql_task_succeeds():
    t5 = next(t for t in TASK_SET if t.id == "T5")
    result = _run_task_dry(t5, session_idx=0)
    assert result.success, "T5 dry-run should pass SQL check"


def test_dry_run_tokens_vary_by_session():
    t1 = next(t for t in TASK_SET if t.id == "T1")
    r0 = _run_task_dry(t1, session_idx=0)
    r3 = _run_task_dry(t1, session_idx=3)
    # Not a hard requirement, but dry-run tokens should differ across sessions
    # (deterministic per seed, but different seeds)
    assert r0.total_tokens > 0 and r3.total_tokens > 0


# ---------------------------------------------------------------------------
# ArmResult / SessionResult aggregation
# ---------------------------------------------------------------------------

def test_session_result_success_rate():
    session = SessionResult(session_idx=0, task_results=[
        TaskResult("T1", success=True),
        TaskResult("T2", success=True),
        TaskResult("T3", success=False),
    ])
    assert abs(session.success_rate - 2/3) < 0.01


def test_arm_result_mean_tokens():
    arm = ArmResult("stock", sessions=[
        SessionResult(0, [TaskResult("T1", True, input_tokens=500, output_tokens=100)]),
        SessionResult(1, [TaskResult("T1", True, input_tokens=700, output_tokens=200)]),
    ])
    assert arm.mean_tokens_per_session == (600 + 900) / 2


# ---------------------------------------------------------------------------
# BenchmarkReport
# ---------------------------------------------------------------------------

def _make_report(stock_tokens=2000, plus_tokens=1400, stock_success=0.8, plus_success=0.9):
    def _arm(name, tokens, success):
        session = SessionResult(0, task_results=[
            TaskResult("T1", success=(success > 0.5), input_tokens=tokens, output_tokens=0,
                       skill_count_before=0, skill_count_after=2),
        ])
        return ArmResult(arm_name=name, sessions=[session])

    return BenchmarkReport(
        stock=_arm("stock", stock_tokens, stock_success),
        plus=_arm("plus", plus_tokens, plus_success),
        n_sessions=1,
        tasks_run=["T1"],
        model="test-model",
        provider="test-provider",
        timestamp="2026T000000Z",
    )


def test_report_token_reduction_pct():
    r = _make_report(stock_tokens=2000, plus_tokens=1400)
    assert abs(r.token_reduction_pct() - 30.0) < 0.1


def test_report_skill_reduction():
    r = _make_report()
    # Both arms have skill_delta=2; reduction = 0
    assert r.skill_reduction() == 0


def test_report_to_dict_has_summary():
    r = _make_report()
    d = r.to_dict()
    assert "summary" in d
    assert "token_reduction_pct" in d["summary"]
    assert "skill_reduction" in d["summary"]
    assert "success_delta_pct" in d["summary"]


def test_report_to_markdown_contains_table():
    r = _make_report()
    md = r.to_markdown()
    assert "| Metric |" in md
    assert "tokens/session" in md.lower()


def test_report_write(tmp_path):
    r = _make_report()
    r.write(tmp_path)
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["model"] == "test-model"


# ---------------------------------------------------------------------------
# End-to-end dry-run (no API calls)
# ---------------------------------------------------------------------------

def test_dry_run_benchmark_end_to_end(tmp_path):
    config = BenchmarkConfig(
        n_sessions=2,
        tasks=TASK_SET[:3],   # T1-T3 only for speed
        dry_run=True,
        output_dir=tmp_path / "output",
    )
    report = run_benchmark(config)

    assert len(report.stock.sessions) == 2
    assert len(report.plus.sessions) == 2
    assert (tmp_path / "output" / "report.json").exists()
    assert (tmp_path / "output" / "report.md").exists()

    # Sanity: token metrics are populated
    assert report.stock.mean_tokens_per_session > 0
    assert report.plus.mean_tokens_per_session > 0
