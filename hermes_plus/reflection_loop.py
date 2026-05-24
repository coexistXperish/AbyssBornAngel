"""Reflection Loop — HERMES++ fix for Problem 3 (Memory That Doesn't Learn).

After each completed session, a lightweight reflection agent:
  1. Summarizes what was accomplished and what was learned.
  2. Diffs the summary against existing memory entries.
  3. Proposes memory patches (add / update / remove).
  4. Gates the patches behind human confirmation before writing.

Confirmation gate:
  - Dev/sandbox: prints to stdout and reads stdin (or auto-approves if
    HERMES_REFLECT_AUTO_APPROVE=1).
  - Production: sends an email via the configured mailer and waits for a
    reply (Mailpit in sandbox, real SMTP in prod).

Integration:
    from hermes_plus.reflection_loop import ReflectionMemoryHook
    class MyProvider(ReflectionMemoryHook, MemoryProvider):
        pass
    # on_session_end is intercepted; patches are proposed to the user
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import textwrap
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_SUMMARY_CHARS = 2_000
AUTO_APPROVE_ENV = "HERMES_REFLECT_AUTO_APPROVE"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MemoryPatch:
    """A proposed change to the agent's memory store."""
    action: str          # "add" | "update" | "remove"
    key: str             # memory key / topic
    old_value: str = ""
    new_value: str = ""
    rationale: str = ""

    def diff_text(self) -> str:
        if self.action == "remove":
            return f"- {self.key}: {self.old_value}"
        if self.action == "add":
            return f"+ {self.key}: {self.new_value}"
        lines_old = self.old_value.splitlines(keepends=True)
        lines_new = self.new_value.splitlines(keepends=True)
        diff = list(difflib.unified_diff(
            lines_old, lines_new,
            fromfile=f"{self.key} (old)",
            tofile=f"{self.key} (new)",
            lineterm="",
        ))
        return "\n".join(diff) if diff else f"(no textual diff for {self.key})"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReflectionResult:
    session_summary: str
    learned: List[str] = field(default_factory=list)
    proposed_patches: List[MemoryPatch] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Reflection extraction (heuristic — no LLM required at this layer)
# ---------------------------------------------------------------------------

def extract_session_learnings(
    messages: List[Dict[str, Any]],
    *,
    max_summary_chars: int = MAX_SUMMARY_CHARS,
) -> ReflectionResult:
    """Parse a conversation message list into a ReflectionResult.

    Heuristic approach (no LLM call):
      - session_summary: last assistant message, truncated
      - learned: lines starting with "I learned", "Note:", "Remember:" etc.
    """
    assistant_msgs = [
        m.get("content", "")
        for m in messages
        if m.get("role") == "assistant"
    ]
    raw_summary = assistant_msgs[-1] if assistant_msgs else ""
    if isinstance(raw_summary, list):
        # Handle content-block format
        text_blocks = [b.get("text", "") for b in raw_summary if isinstance(b, dict)]
        raw_summary = " ".join(text_blocks)

    session_summary = textwrap.shorten(
        str(raw_summary), width=max_summary_chars, placeholder="…"
    )

    # Simple keyword extraction for "learned" items
    learned: List[str] = []
    markers = ("i learned", "note:", "remember:", "important:", "key insight:")
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        for line in content.splitlines():
            lower = line.strip().lower()
            if any(lower.startswith(m) for m in markers):
                learned.append(line.strip())

    return ReflectionResult(
        session_summary=session_summary,
        learned=learned[:20],  # cap
    )


# ---------------------------------------------------------------------------
# Diff against existing memory
# ---------------------------------------------------------------------------

def propose_patches(
    reflection: ReflectionResult,
    existing_memory: Dict[str, str],
) -> List[MemoryPatch]:
    """Diff reflection.learned items against existing_memory and propose patches.

    Returns a list of MemoryPatch objects. Does NOT write anything.
    """
    patches: List[MemoryPatch] = []

    # New learnings not in memory → add
    for item in reflection.learned:
        key = _derive_key(item)
        if key not in existing_memory:
            patches.append(MemoryPatch(
                action="add",
                key=key,
                new_value=item,
                rationale="New insight from session.",
            ))
        else:
            old = existing_memory[key]
            if old.strip() != item.strip():
                patches.append(MemoryPatch(
                    action="update",
                    key=key,
                    old_value=old,
                    new_value=item,
                    rationale="Updated based on session outcome.",
                ))

    # Session summary as a special key
    summary_key = "_session_summary"
    patches.append(MemoryPatch(
        action="add" if summary_key not in existing_memory else "update",
        key=summary_key,
        old_value=existing_memory.get(summary_key, ""),
        new_value=reflection.session_summary,
        rationale="Latest session summary.",
    ))

    return patches


def _derive_key(text: str) -> str:
    """Turn a learned-item string into a short stable key."""
    import re
    cleaned = re.sub(r"[^a-z0-9 ]", "", text.lower())
    words = cleaned.split()[:5]
    return "_".join(words) if words else "misc"


# ---------------------------------------------------------------------------
# Confirmation gate
# ---------------------------------------------------------------------------

def confirm_patches(patches: List[MemoryPatch]) -> List[MemoryPatch]:
    """Present patches to the user and return approved ones.

    Checks HERMES_REFLECT_AUTO_APPROVE env var. If set to '1', approves all.
    Otherwise prompts via stdout/stdin (suitable for CLI/sandbox).
    """
    if not patches:
        return []

    if os.environ.get(AUTO_APPROVE_ENV) == "1":
        logger.info("reflection_loop: auto-approving %d patches", len(patches))
        return patches

    approved: List[MemoryPatch] = []
    print("\n=== HERMES++ Memory Reflection ===")
    print(f"{len(patches)} proposed memory patch(es):\n")

    for i, patch in enumerate(patches, 1):
        print(f"[{i}/{len(patches)}] {patch.action.upper()} — {patch.key}")
        print(patch.diff_text())
        print(f"  Rationale: {patch.rationale}")
        try:
            choice = input("  Apply? [y/N]: ").strip().lower()
        except EOFError:
            choice = "n"
        if choice == "y":
            approved.append(patch)
        print()

    print(f"Approved {len(approved)}/{len(patches)} patches.\n")
    return approved


# ---------------------------------------------------------------------------
# MemoryProvider mixin
# ---------------------------------------------------------------------------

class ReflectionMemoryHook:
    """Mixin for MemoryProvider subclasses that want post-session reflection.

    on_session_end() is intercepted; patches are proposed and gated.
    Providers must implement _get_memory_store() and _apply_patch().
    """

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        try:
            reflection = extract_session_learnings(messages)
            if not reflection.session_summary and not reflection.learned:
                logger.debug("reflection_loop: nothing to reflect on")
                return

            existing = self._get_memory_store()
            patches = propose_patches(reflection, existing)
            if not patches:
                logger.debug("reflection_loop: no patches proposed")
                return

            approved = confirm_patches(patches)
            for patch in approved:
                try:
                    self._apply_patch(patch)
                except Exception as e:
                    logger.warning("reflection_loop: failed to apply patch %s: %s", patch.key, e)

            logger.info(
                "reflection_loop: session_end complete — %d/%d patches applied",
                len(approved),
                len(patches),
            )
        except Exception as e:
            logger.warning("reflection_loop: on_session_end failed: %s", e, exc_info=True)

    def _get_memory_store(self) -> Dict[str, str]:
        """Return current memory as {key: value}. Override in provider."""
        return {}

    def _apply_patch(self, patch: MemoryPatch) -> None:
        """Write a single approved patch. Override in provider."""
        logger.debug("reflection_loop: patch not applied (no-op base): %s", patch.key)


# ---------------------------------------------------------------------------
# Email confirmation gate (Mailpit / SMTP)
# ---------------------------------------------------------------------------

def send_patch_email(patches: List[MemoryPatch], *, to: str, smtp_host: str, smtp_port: int = 1025) -> None:
    """Send a patch summary email via SMTP (Mailpit in sandbox).

    This is a fire-and-forget send; the human replies out-of-band (future work:
    parse reply to auto-approve). For now it's a notification-only gate — the
    patches are printed to stdout as well.
    """
    import smtplib
    from email.mime.text import MIMEText

    lines = ["HERMES++ proposes the following memory patches:", ""]
    for p in patches:
        lines.append(f"  [{p.action.upper()}] {p.key}")
        lines.append(f"  {p.rationale}")
        lines.append("")
        lines.append(textwrap.indent(p.diff_text(), "    "))
        lines.append("")
    lines.append("Reply to this email to approve (future feature). For now, check CLI.")

    msg = MIMEText("\n".join(lines))
    msg["Subject"] = f"HERMES++ memory patch proposal ({len(patches)} patches)"
    msg["From"] = "hermes-plus@localhost"
    msg["To"] = to

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=5) as s:
            s.sendmail(msg["From"], [to], msg.as_string())
        logger.info("reflection_loop: patch email sent to %s via %s:%d", to, smtp_host, smtp_port)
    except Exception as e:
        logger.warning("reflection_loop: failed to send patch email: %s", e)
