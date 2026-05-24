"""Output Contracts — HERMES++ fix for Problem 2 (Parent Context Pollution).

Subagents return a typed OutputContract instead of raw text, capping the
amount of context that bleeds back to the parent.

How it works:
  1. SubagentResult wraps raw output with a schema: summary, artifacts, errors.
  2. enforce_contract() trims raw_output to MAX_RAW_CHARS before storage.
  3. The parent receives only the contract (summary + artifacts), never
     the full subagent trace.
  4. OutputContractMemoryHook is a MemoryProvider mixin that intercepts
     on_delegation() calls and rewrites the result through the contract.

Usage in run_agent.py / memory providers:
    from hermes_plus.output_contracts import OutputContractMemoryHook
    class MyProvider(OutputContractMemoryHook, MemoryProvider):
        ...
    # on_delegation is now automatically contract-filtered
"""

from __future__ import annotations

import json
import logging
import textwrap
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Max characters of raw subagent output preserved (rest is dropped).
# The contract summary is always kept in full.
MAX_RAW_CHARS: int = 4_000

# Max characters for the contract summary field.
MAX_SUMMARY_CHARS: int = 800


@dataclass
class SubagentResult:
    """Typed contract for subagent output.

    Fields:
      task        — the original task description sent to the subagent
      summary     — 1–3 sentence human-readable outcome (required)
      artifacts   — key/value outputs the parent may need (paths, counts, etc.)
      errors      — non-fatal issues encountered
      success     — whether the subagent considered itself successful
      raw_output  — truncated trace; NOT forwarded to parent LLM context
    """
    task: str
    summary: str
    artifacts: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    success: bool = True
    raw_output: str = ""

    def to_parent_context(self) -> str:
        """Return the minimal string the parent sees — no raw trace."""
        parts = [f"Task: {self.task}", f"Outcome: {self.summary}"]
        if self.artifacts:
            parts.append("Artifacts: " + json.dumps(self.artifacts, default=str))
        if self.errors:
            parts.append("Errors: " + "; ".join(self.errors))
        if not self.success:
            parts.append("Status: FAILED")
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def enforce_contract(
    task: str,
    raw_output: str,
    *,
    max_raw_chars: int = MAX_RAW_CHARS,
    max_summary_chars: int = MAX_SUMMARY_CHARS,
) -> SubagentResult:
    """Build a SubagentResult from a raw subagent output string.

    Tries to parse structured JSON first (if the subagent was prompted to
    emit a contract block). Falls back to heuristic extraction.
    """
    # Attempt JSON contract parse
    contract = _try_parse_json_contract(task, raw_output)
    if contract is not None:
        contract.raw_output = raw_output[:max_raw_chars]
        contract.summary = contract.summary[:max_summary_chars]
        return contract

    # Heuristic: first non-empty paragraph as summary
    summary = _heuristic_summary(raw_output, max_summary_chars)
    truncated_raw = raw_output[:max_raw_chars]
    if len(raw_output) > max_raw_chars:
        truncated_raw += f"\n[... truncated {len(raw_output) - max_raw_chars} chars]"

    return SubagentResult(
        task=task,
        summary=summary,
        raw_output=truncated_raw,
    )


def _try_parse_json_contract(task: str, text: str) -> Optional[SubagentResult]:
    """Look for a ```json ... ``` block or bare JSON object with contract keys."""
    import re
    pattern = re.compile(r'```json\s*(\{.*?\})\s*```', re.DOTALL)
    m = pattern.search(text)
    if not m:
        # Try bare JSON at end of output
        m2 = re.search(r'\{[^{}]*"summary"[^{}]*\}', text, re.DOTALL)
        if not m2:
            return None
        json_str = m2.group(0)
    else:
        json_str = m.group(1)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if "summary" not in data:
        return None

    return SubagentResult(
        task=data.get("task", task),
        summary=str(data.get("summary", "")),
        artifacts=data.get("artifacts", {}),
        errors=data.get("errors", []),
        success=bool(data.get("success", True)),
    )


def _heuristic_summary(text: str, max_chars: int) -> str:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    # Prefer the last non-trivial paragraph (often the final answer)
    for p in reversed(paragraphs):
        if len(p) >= 20:
            return textwrap.shorten(p, width=max_chars, placeholder="…")
    return textwrap.shorten(text.strip(), width=max_chars, placeholder="…")


# ---------------------------------------------------------------------------
# MemoryProvider mixin
# ---------------------------------------------------------------------------

class OutputContractMemoryHook:
    """Mixin for MemoryProvider subclasses that want contract-filtered delegation.

    Override on_delegation_contract() to receive the filtered result.
    The base on_delegation() is intercepted automatically.
    """

    def on_delegation(
        self,
        task: str,
        result: str,
        *,
        child_session_id: str = "",
        **kwargs: Any,
    ) -> None:
        contract = enforce_contract(task, result)
        logger.debug(
            "output_contracts: task=%r summary=%r artifacts=%s success=%s "
            "raw_len=%d→%d",
            task[:80],
            contract.summary[:80],
            list(contract.artifacts.keys()),
            contract.success,
            len(result),
            len(contract.raw_output),
        )
        self.on_delegation_contract(contract, child_session_id=child_session_id, **kwargs)

    def on_delegation_contract(
        self,
        contract: SubagentResult,
        *,
        child_session_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Override to handle the filtered contract. Default: no-op."""


# ---------------------------------------------------------------------------
# Prompt fragment to instruct subagents to emit a contract
# ---------------------------------------------------------------------------

SUBAGENT_CONTRACT_PROMPT = """\
When you have completed your task, output a JSON block in this exact format
(inside triple-backtick json fences) as your final message:

```json
{
  "summary": "<1-3 sentence outcome>",
  "artifacts": {"<key>": "<value>"},
  "errors": ["<any non-fatal issues>"],
  "success": true
}
```

Do NOT include your full reasoning trace in this block.
"""
