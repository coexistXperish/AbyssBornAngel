# HERMES++ — Project Guide

This is a fork of NousResearch/hermes-agent with targeted improvements
to skill lifecycle management, subagent output isolation, and adaptive memory.

## Branch layout

| Branch | Purpose |
|---|---|
| `main` | Your stable fork of hermes-agent |
| `claude/fervent-archimedes-rBOQw` | Active development |
| `upstream/tracking` | Mirrors `NousResearch/hermes-agent:main`; never force-pushed |

To pull upstream changes without touching main:
```
git fetch upstream
git checkout upstream/tracking
git merge upstream/main
```

## HERMES++ additions (`hermes_plus/`)

### Problem 1 — Skill Explosion → `hermes_plus/skill_registry.py`
- `confidence_score(record)` — float 0–1 from use_count + recency + age
- `prune_candidates()` — skills below `MIN_CONFIDENCE` for `PRUNE_AFTER_DAYS`
- `merge_candidates()` — trigram Jaccard similarity to detect duplicates
- `record_version()` / `pin()` / `unpin()` — versioning + curator bypass
- Hooks alongside `agent/curator.py`; does not replace it

### Problem 2 — Context Pollution → `hermes_plus/output_contracts.py`
- `SubagentResult` dataclass — typed contract with summary/artifacts/errors
- `enforce_contract(task, raw_output)` — parse or heuristic-extract
- `OutputContractMemoryHook` — MemoryProvider mixin; intercepts `on_delegation()`
- `SUBAGENT_CONTRACT_PROMPT` — inject into subagent system prompt
- Parent LLM only ever sees `contract.to_parent_context()`, not the raw trace

### Problem 3 — Static Memory → `hermes_plus/reflection_loop.py`
- `extract_session_learnings(messages)` — heuristic extraction, no LLM call
- `propose_patches(reflection, existing_memory)` — diff-based proposals
- `confirm_patches(patches)` — CLI gate (or auto-approve via env var)
- `ReflectionMemoryHook` — MemoryProvider mixin; intercepts `on_session_end()`
- `send_patch_email()` — Mailpit (sandbox) / SMTP (prod) notification

## Integrating into a MemoryProvider

```python
from hermes_plus.output_contracts import OutputContractMemoryHook
from hermes_plus.reflection_loop import ReflectionMemoryHook
from agent.memory_provider import MemoryProvider

class HermesPlusProvider(OutputContractMemoryHook, ReflectionMemoryHook, MemoryProvider):
    @property
    def name(self): return "hermes_plus"

    def is_available(self): return True

    def system_prompt_block(self): return ""

    def prefetch(self, query, *, session_id=""): return ""

    def sync_turn(self, user_content, assistant_content, *, session_id=""): pass

    # Reflection hook
    def _get_memory_store(self): return {}   # load from Postgres here
    def _apply_patch(self, patch): pass       # write to Postgres here

    # Contract hook
    def on_delegation_contract(self, contract, *, session_id="", **kw):
        # store contract.to_dict() in Postgres
        pass
```

## Stack

- Orchestration: n8n (Hetzner Docker stack)
- Memory backend: Postgres (same stack)
- Confirmation gate: Mailpit sandbox → real SMTP in prod
- Env var: `HERMES_REFLECT_AUTO_APPROVE=1` to skip CLI confirmation gate

## Phase 1 checklist

- [x] Skill registry — confidence scoring, TTL prune, merge detection, versioning
- [x] Output contracts — SubagentResult, enforce_contract, MemoryProvider mixin
- [x] Reflection loop — session learnings, patch proposals, confirmation gate
- [x] CLAUDE.md
- [ ] Postgres-backed MemoryProvider implementation
- [ ] n8n webhook trigger for reflection gate
- [ ] Benchmark harness (stock Hermes vs Hermes++ over 10 sessions)

## Upstream conflict workflow (future)

Upstream changes land on `upstream/tracking`. Before merging to `main`:
1. GitHub Copilot summarizes the diff (see `.github/workflows/conflict-review.yml`)
2. You get a notification with two options
3. You pick one; Claude handles the merge and opens a PR
