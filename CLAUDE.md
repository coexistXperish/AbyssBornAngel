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

## Wiring (DONE)

`hermes_plus/provider.py` — `HermesPlusProvider` is the concrete
`MemoryProvider` that uses both mixins. It is **wired into the agent**:

- `agent/agent_init.py` registers it when `memory.hermes_plus: true` in config
  (opt-in, default off). It coexists with any external backend.
- `agent/memory_manager.py` exempts `"hermes_plus"` from the single-external-
  provider limit (it's local-first, not a competing backend).

Enable it in `~/.hermes/config.yaml`:
```yaml
memory:
  hermes_plus: true
```

## Storage — local-first (NOT Postgres)

Hermes itself is local-first: SQLite (`hermes_state.py`) + JSON files under
`~/.hermes`. There is no Postgres in the agent/memory/skills layer (the only
`asyncpg` dep is the Matrix chat gateway). HERMES++ follows suit:

- `hermes_plus_memory.json` — reflection memory + delegation contracts
- `.skill_registry.json` — skill versions / pins

Postgres/pgvector is only worth adding later if we want semantic recall or
n8n needs its own backend — a deliberate choice, not a default.

## Stack

- Orchestration: n8n (Hetzner Docker stack)
- Memory backend: local JSON/SQLite (Postgres optional, later)
- Confirmation gate: CLI prompt; Mailpit sandbox → real SMTP in prod
- Env var: `HERMES_REFLECT_AUTO_APPROVE=1` to skip CLI confirmation gate

## Tests

`tests/hermes_plus/` — 37 tests (unit + one end-to-end provider integration).
Run: `uv run pytest tests/hermes_plus/ -q`

## Phase 1 checklist

- [x] Skill registry — confidence scoring, TTL prune, merge detection, versioning
- [x] Output contracts — SubagentResult, enforce_contract, MemoryProvider mixin
- [x] Reflection loop — session learnings, patch proposals, confirmation gate
- [x] HermesPlusProvider — concrete provider, wired into agent_init (opt-in)
- [x] Test suite — 37 tests, wired into CI via tests/hermes_plus/
- [x] CLAUDE.md
- [ ] n8n webhook trigger for reflection gate
- [ ] Benchmark harness (stock Hermes vs Hermes++ over 10 sessions)
- [ ] X.com watcher for Nous/Hermes devs (intel feed)
- [ ] (parked) desktop widget for Option A/B upstream decisions

## Upstream conflict workflow (future)

Upstream changes land on `upstream/tracking`. Before merging to `main`:
1. GitHub Copilot summarizes the diff (see `.github/workflows/conflict-review.yml`)
2. You get a notification with two options
3. You pick one; Claude handles the merge and opens a PR
