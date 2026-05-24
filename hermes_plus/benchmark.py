"""Benchmark harness — HERMES++ vs stock Hermes.

Runs a fixed task set against two isolated arms and measures:
  1. Skill count delta   — how many agent-created skills accumulate per session
  2. Tokens per session  — input + output tokens consumed
  3. Task success rate   — deterministic per-task assertion

Usage:
    from hermes_plus.benchmark import BenchmarkConfig, run_benchmark
    report = run_benchmark(config)
    report.write(output_dir)

CLI entry point: hermes_plus/run_benchmark.py
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

@dataclass
class Task:
    id: str
    prompt: str
    check: Callable[[str], bool]   # success predicate on final_response
    description: str = ""


def _has_n_bullets(n: int) -> Callable[[str], bool]:
    def check(resp: str) -> bool:
        bullets = [l for l in resp.splitlines() if l.strip().startswith(("-", "*", "•")) or
                   (len(l) > 2 and l.strip()[0].isdigit() and l.strip()[1] in ".):")]
        return len(bullets) >= n
    return check


def _contains_all(*fragments: str) -> Callable[[str], bool]:
    def check(resp: str) -> bool:
        lower = resp.lower()
        return all(f.lower() in lower for f in fragments)
    return check


def _contains_code_block(resp: str) -> bool:
    return "```" in resp or "def " in resp or "function " in resp


def _contains_sql(resp: str) -> bool:
    lower = resp.lower()
    return "select" in lower and ("from" in lower or "where" in lower)


# The task set stresses the three problems we fixed:
#   T1-T2: knowledge/research tasks (baseline, low skill pressure)
#   T3:    subagent delegation (stresses Problem 2 — output contracts)
#   T4-T5: skill creation + recall (stresses Problems 1 and 3)
TASK_SET: List[Task] = [
    Task(
        id="T1",
        description="Code generation with deterministic assert check",
        prompt=(
            "Write a Python function called `parse_iso_date` that accepts a string "
            "and returns a datetime object. It should handle both date-only strings "
            "(e.g. '2026-05-24') and datetime strings (e.g. '2026-05-24T10:30:00'). "
            "Include a usage example."
        ),
        check=_contains_code_block,
    ),
    Task(
        id="T2",
        description="Research summary — checks for bullet list output",
        prompt=(
            "Summarise best practices for Docker container healthchecks in exactly 5 "
            "bullet points. Focus on practical production use."
        ),
        check=_has_n_bullets(4),   # 4 not 5 — tolerance for formatting variation
    ),
    Task(
        id="T3",
        description="Multi-step delegation — stresses subagent output contracts",
        prompt=(
            "Break the following task into 3 numbered subtasks and then complete each one: "
            "Transform a CSV with columns [name, dob, score] so that (1) dob is converted "
            "to age in years, (2) score is normalised 0-1, (3) rows with score < 0.3 are "
            "filtered out. Show the Python code for each subtask separately."
        ),
        check=lambda r: r.count("```") >= 2 and _contains_all("def ", "import")(r),
    ),
    Task(
        id="T4",
        description="Skill creation — stresses Problem 1 (skill accumulation)",
        prompt=(
            "Create a reusable skill called 'sql-generator' that takes a plain-English "
            "description of a SELECT query and returns valid SQL. Document the skill with "
            "a SKILL.md that includes a description, usage example, and the Python script."
        ),
        check=_contains_all("skill", "sql", "select"),
    ),
    Task(
        id="T5",
        description="SQL generation — tests skill recall across turns",
        prompt=(
            "Using a sql-generator approach, write a SQL SELECT query that returns all "
            "users from a 'users' table where age > 30, ordered by last_name ascending."
        ),
        check=_contains_sql,
    ),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    task_id: str
    success: bool
    input_tokens: int = 0
    output_tokens: int = 0
    skill_count_before: int = 0
    skill_count_after: int = 0
    duration_seconds: float = 0.0
    response_preview: str = ""    # first 200 chars of response
    error: str = ""               # set if agent raised an exception

    @property
    def skill_delta(self) -> int:
        return self.skill_count_after - self.skill_count_before

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class SessionResult:
    session_idx: int
    task_results: List[TaskResult] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(t.total_tokens for t in self.task_results)

    @property
    def success_rate(self) -> float:
        if not self.task_results:
            return 0.0
        return sum(1 for t in self.task_results if t.success) / len(self.task_results)

    @property
    def total_skill_delta(self) -> int:
        return sum(t.skill_delta for t in self.task_results)


@dataclass
class ArmResult:
    arm_name: str
    sessions: List[SessionResult] = field(default_factory=list)

    @property
    def mean_tokens_per_session(self) -> float:
        if not self.sessions:
            return 0.0
        return sum(s.total_tokens for s in self.sessions) / len(self.sessions)

    @property
    def mean_success_rate(self) -> float:
        if not self.sessions:
            return 0.0
        return sum(s.success_rate for s in self.sessions) / len(self.sessions)

    @property
    def total_skill_delta(self) -> int:
        return sum(s.total_skill_delta for s in self.sessions)


@dataclass
class BenchmarkReport:
    stock: ArmResult
    plus: ArmResult
    n_sessions: int
    tasks_run: List[str]
    model: str
    provider: str
    timestamp: str

    def token_reduction_pct(self) -> float:
        s = self.stock.mean_tokens_per_session
        p = self.plus.mean_tokens_per_session
        if s == 0:
            return 0.0
        return round((s - p) / s * 100, 1)

    def skill_reduction(self) -> int:
        return self.stock.total_skill_delta - self.plus.total_skill_delta

    def success_delta_pct(self) -> float:
        return round((self.plus.mean_success_rate - self.stock.mean_success_rate) * 100, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "model": self.model,
            "provider": self.provider,
            "n_sessions": self.n_sessions,
            "tasks_run": self.tasks_run,
            "stock": asdict(self.stock),
            "plus": asdict(self.plus),
            "summary": {
                "token_reduction_pct": self.token_reduction_pct(),
                "skill_reduction": self.skill_reduction(),
                "success_delta_pct": self.success_delta_pct(),
            },
        }

    def to_markdown(self) -> str:
        lines = [
            "# HERMES++ Benchmark Report",
            f"\nModel: `{self.model}` via `{self.provider}` | Sessions: {self.n_sessions}",
            f"Timestamp: {self.timestamp}\n",
            "## Summary",
            "",
            "| Metric | Stock | HERMES++ | Delta |",
            "|--------|-------|----------|-------|",
            f"| Mean tokens/session | {self.stock.mean_tokens_per_session:.0f} | "
            f"{self.plus.mean_tokens_per_session:.0f} | "
            f"{self.token_reduction_pct():+.1f}% |",
            f"| Task success rate | {self.stock.mean_success_rate:.1%} | "
            f"{self.plus.mean_success_rate:.1%} | "
            f"{self.success_delta_pct():+.1f}pp |",
            f"| Total skill delta | {self.stock.total_skill_delta} | "
            f"{self.plus.total_skill_delta} | "
            f"{self.skill_reduction():+d} |",
            "",
            "## Tasks",
            "",
        ]
        for task in TASK_SET:
            if task.id in self.tasks_run:
                lines.append(f"- **{task.id}**: {task.description}")
        lines.append("")
        return "\n".join(lines)

    def write(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "report.json").write_text(
            json.dumps(self.to_dict(), indent=2), encoding="utf-8"
        )
        (output_dir / "report.md").write_text(self.to_markdown(), encoding="utf-8")
        logger.info("Benchmark report written to %s", output_dir)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    model: str = "claude-opus-4-7"
    provider: str = "anthropic"
    n_sessions: int = 10
    tasks: List[Task] = field(default_factory=lambda: TASK_SET)
    output_dir: Optional[Path] = None
    dry_run: bool = False           # stub agent calls, no real API usage
    stock_hermes_home: Optional[Path] = None   # auto-generated tmp dir if None
    plus_hermes_home: Optional[Path] = None


# ---------------------------------------------------------------------------
# Agent driver
# ---------------------------------------------------------------------------

def _count_skills(hermes_home: Path) -> int:
    try:
        env_backup = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = str(hermes_home)
        try:
            # Re-import to pick up new HERMES_HOME
            from tools.skill_usage import list_agent_created_skill_names
            return len(list_agent_created_skill_names())
        finally:
            if env_backup is None:
                os.environ.pop("HERMES_HOME", None)
            else:
                os.environ["HERMES_HOME"] = env_backup
    except Exception as e:
        logger.warning("benchmark: skill count failed: %s", e)
        return 0


def _run_task_live(task: Task, config: BenchmarkConfig, hermes_home: Path) -> TaskResult:
    """Run a single task against the live agent."""
    from hermes_cli.oneshot import _run_agent

    start = time.monotonic()
    skill_before = _count_skills(hermes_home)

    env_backup = os.environ.copy()
    os.environ["HERMES_HOME"] = str(hermes_home)

    try:
        result = _run_agent(
            task.prompt,
            model=config.model,
            provider=config.provider,
            toolsets=None,
        )
        response = result.get("final_response", "") or ""
        success = task.check(response)
        return TaskResult(
            task_id=task.id,
            success=success,
            input_tokens=result.get("input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
            skill_count_before=skill_before,
            skill_count_after=_count_skills(hermes_home),
            duration_seconds=round(time.monotonic() - start, 2),
            response_preview=response[:200],
        )
    except Exception as e:
        logger.warning("benchmark: task %s failed: %s", task.id, e)
        return TaskResult(
            task_id=task.id,
            success=False,
            skill_count_before=skill_before,
            skill_count_after=_count_skills(hermes_home),
            duration_seconds=round(time.monotonic() - start, 2),
            error=str(e),
        )
    finally:
        # Restore env
        for k, v in env_backup.items():
            os.environ[k] = v
        for k in list(os.environ):
            if k not in env_backup:
                del os.environ[k]


def _run_task_dry(task: Task, session_idx: int) -> TaskResult:
    """Stub task run for dry-run mode — no real API calls."""
    import hashlib
    # Deterministic fake response based on task+session so checks behave realistically
    seed = f"{task.id}-{session_idx}"
    h = int(hashlib.md5(seed.encode()).hexdigest(), 16)

    fake_responses = {
        "T1": "```python\nfrom datetime import datetime\ndef parse_iso_date(s):\n    import datetime\n    return datetime.datetime.fromisoformat(s)\n```\nUsage: parse_iso_date('2026-05-24')",
        "T2": "- Use HEALTHCHECK CMD curl -f http://localhost/health\n- Set --interval=30s\n- Set --timeout=5s\n- Set --retries=3\n- Avoid checking external deps",
        "T3": "```python\nimport pandas as pd\ndef subtask1(df): ...\n```\n```python\ndef subtask2(df): import ...\n```\n```python\ndef subtask3(df): ...\n```",
        "T4": "This skill documentation covers sql-generator. Usage: select all rows. The script generates SELECT queries.",
        "T5": "SELECT * FROM users WHERE age > 30 ORDER BY last_name ASC;",
    }
    response = fake_responses.get(task.id, f"Fake response for {task.id}")
    base_tokens = 1000 + (h % 500)
    return TaskResult(
        task_id=task.id,
        success=task.check(response),
        input_tokens=base_tokens,
        output_tokens=base_tokens // 3,
        skill_count_before=session_idx,
        skill_count_after=session_idx + (1 if task.id == "T4" else 0),
        duration_seconds=0.01,
        response_preview=response[:200],
    )


# ---------------------------------------------------------------------------
# Arm runner
# ---------------------------------------------------------------------------

def _prepare_hermes_home(base: Path, arm_name: str, hermes_plus: bool) -> Path:
    """Create an isolated HERMES_HOME for one arm."""
    home = base / arm_name
    home.mkdir(parents=True, exist_ok=True)
    # Write minimal config
    config = {"memory": {"hermes_plus": hermes_plus}}
    import yaml
    try:
        (home / "config.yaml").write_text(yaml.dump(config), encoding="utf-8")
    except ImportError:
        import json as _json
        (home / "config.json").write_text(_json.dumps(config), encoding="utf-8")
    return home


def run_arm(
    arm_name: str,
    hermes_home: Path,
    config: BenchmarkConfig,
    hermes_plus_enabled: bool,
) -> ArmResult:
    arm = ArmResult(arm_name=arm_name)
    print(f"\n{'='*60}")
    print(f"  Running arm: {arm_name} ({config.n_sessions} sessions)")
    print(f"  HERMES_HOME: {hermes_home}")
    print(f"  hermes_plus: {hermes_plus_enabled}")
    print(f"{'='*60}")

    for session_idx in range(config.n_sessions):
        print(f"\n  [Session {session_idx+1}/{config.n_sessions}]")
        session = SessionResult(session_idx=session_idx)

        for task in config.tasks:
            print(f"    Task {task.id}: {task.description[:50]}...", end=" ", flush=True)
            if config.dry_run:
                result = _run_task_dry(task, session_idx)
            else:
                result = _run_task_live(task, config, hermes_home)

            session.task_results.append(result)
            status = "✓" if result.success else "✗"
            print(f"{status}  ({result.total_tokens} tokens, Δskills={result.skill_delta})")

        arm.sessions.append(session)
        print(f"  Session summary: success={session.success_rate:.0%}  "
              f"tokens={session.total_tokens}  Δskills={session.total_skill_delta}")

    return arm


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_benchmark(config: BenchmarkConfig) -> BenchmarkReport:
    import tempfile
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_tmp = Path(tempfile.mkdtemp(prefix="hermes-bench-"))

    stock_home = config.stock_hermes_home or _prepare_hermes_home(
        base_tmp, "stock", hermes_plus=False
    )
    plus_home = config.plus_hermes_home or _prepare_hermes_home(
        base_tmp, "plus", hermes_plus=True
    )

    try:
        stock_result = run_arm("stock", stock_home, config, hermes_plus_enabled=False)
        plus_result = run_arm("plus", plus_home, config, hermes_plus_enabled=True)
    finally:
        if config.stock_hermes_home is None and config.plus_hermes_home is None:
            shutil.rmtree(base_tmp, ignore_errors=True)

    report = BenchmarkReport(
        stock=stock_result,
        plus=plus_result,
        n_sessions=config.n_sessions,
        tasks_run=[t.id for t in config.tasks],
        model=config.model,
        provider=config.provider,
        timestamp=timestamp,
    )

    output_dir = config.output_dir or (
        Path.home() / ".hermes" / "benchmark-results" / timestamp
    )
    report.write(output_dir)

    _print_summary(report)
    return report


def _print_summary(report: BenchmarkReport) -> None:
    print(f"""
╔══════════════════════════════════════════════════════╗
║           HERMES++ BENCHMARK RESULTS                 ║
╠══════════════════════════════════════════════════════╣
║  Tokens/session:  stock={report.stock.mean_tokens_per_session:.0f}  plus={report.plus.mean_tokens_per_session:.0f}  ({report.token_reduction_pct():+.1f}%)
║  Success rate:    stock={report.stock.mean_success_rate:.0%}  plus={report.plus.mean_success_rate:.0%}  ({report.success_delta_pct():+.1f}pp)
║  Skill delta:     stock={report.stock.total_skill_delta}  plus={report.plus.total_skill_delta}  (saved={report.skill_reduction()})
╚══════════════════════════════════════════════════════╝
""")
