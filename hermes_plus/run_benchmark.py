"""CLI entry point for the HERMES++ benchmark.

Usage:
    # Dry-run (no API key needed — verifies harness logic only)
    uv run python -m hermes_plus.run_benchmark --dry-run --sessions 3

    # Real run
    OPENROUTER_API_KEY=... uv run python -m hermes_plus.run_benchmark \\
        --sessions 10 --model claude-opus-4-7 --provider anthropic

    # Specific output dir
    uv run python -m hermes_plus.run_benchmark --dry-run --output /tmp/bench-out
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="HERMES++ benchmark: stock vs ++ over N sessions"
    )
    parser.add_argument("--sessions", type=int, default=10,
                        help="Number of sessions per arm (default: 10)")
    parser.add_argument("--model", default="claude-opus-4-7",
                        help="Model to use for both arms (default: claude-opus-4-7)")
    parser.add_argument("--provider", default="anthropic",
                        help="Provider (default: anthropic)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory for report (default: ~/.hermes/benchmark-results/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stub agent calls — no real API usage, verifies harness logic")
    parser.add_argument("--tasks", nargs="+",
                        help="Subset of task IDs to run (e.g. T1 T3 T5); default: all")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from hermes_plus.benchmark import BenchmarkConfig, TASK_SET, run_benchmark

    tasks = TASK_SET
    if args.tasks:
        task_map = {t.id: t for t in TASK_SET}
        missing = [tid for tid in args.tasks if tid not in task_map]
        if missing:
            print(f"ERROR: Unknown task IDs: {missing}. Valid: {list(task_map)}")
            return 1
        tasks = [task_map[tid] for tid in args.tasks]

    config = BenchmarkConfig(
        model=args.model,
        provider=args.provider,
        n_sessions=args.sessions,
        tasks=tasks,
        output_dir=args.output,
        dry_run=args.dry_run,
    )

    if config.dry_run:
        print("DRY-RUN mode — no real API calls. Results are synthetic.")
    else:
        print(f"LIVE mode — model={config.model} provider={config.provider}")
        print("Make sure your API key is set in the environment.")

    run_benchmark(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
