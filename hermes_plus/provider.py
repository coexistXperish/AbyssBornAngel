"""HermesPlusProvider — wires the HERMES++ modules into Hermes's memory system.

This is the concrete MemoryProvider that makes the three Phase 1 modules
actually run inside a live agent:

  - Problem 2 (context pollution): OutputContractMemoryHook intercepts
    on_delegation() so the parent stores only a typed contract, never the
    raw subagent trace.
  - Problem 3 (static memory): ReflectionMemoryHook intercepts
    on_session_end() to propose memory patches behind a confirmation gate.

Storage is local-first (matching Hermes's own SQLite/JSON design): a single
JSON file under HERMES_HOME. No external database required. The skill
registry (Problem 1) persists separately via hermes_plus.skill_registry.

Activation (run_agent.py):
    from hermes_plus.provider import HermesPlusProvider
    manager.add_provider(HermesPlusProvider())
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from hermes_plus.output_contracts import OutputContractMemoryHook, SubagentResult
from hermes_plus.reflection_loop import ReflectionMemoryHook, MemoryPatch

logger = logging.getLogger(__name__)

_STORE_FILE = "hermes_plus_memory.json"


class HermesPlusProvider(OutputContractMemoryHook, ReflectionMemoryHook, MemoryProvider):
    """Local-first memory provider implementing HERMES++ Phase 1 hooks."""

    def __init__(self) -> None:
        self._hermes_home: Optional[Path] = None
        self._session_id: str = ""
        self._lock = threading.Lock()

    # -- Required MemoryProvider interface -----------------------------------

    @property
    def name(self) -> str:
        return "hermes_plus"

    def is_available(self) -> bool:
        # Local-only; always available once a home dir can be resolved.
        return True

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._session_id = session_id
        home = kwargs.get("hermes_home")
        if home:
            self._hermes_home = Path(home)
        else:
            try:
                from hermes_constants import get_hermes_home
                self._hermes_home = get_hermes_home()
            except ImportError:
                self._hermes_home = Path.home() / ".hermes"
        self._store_path().parent.mkdir(parents=True, exist_ok=True)
        logger.info("hermes_plus: initialized (store=%s)", self._store_path())

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        # Context-only provider; exposes no tools to the model.
        return []

    def system_prompt_block(self) -> str:
        return ""

    def on_session_switch(self, new_session_id: str, **kwargs: Any) -> None:
        self._session_id = new_session_id

    # -- Local JSON store ----------------------------------------------------

    def _store_path(self) -> Path:
        base = self._hermes_home or (Path.home() / ".hermes")
        return base / _STORE_FILE

    def _load_store(self) -> Dict[str, Any]:
        path = self._store_path()
        if not path.exists():
            return {"memory": {}, "delegations": []}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data.setdefault("memory", {})
            data.setdefault("delegations", [])
            return data
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("hermes_plus: failed to read store: %s", e)
            return {"memory": {}, "delegations": []}

    def _save_store(self, data: Dict[str, Any]) -> None:
        path = self._store_path()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)

    # -- Problem 2: contract-filtered delegation -----------------------------

    def on_delegation_contract(
        self,
        contract: SubagentResult,
        *,
        child_session_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Store only the typed contract — never the raw subagent trace."""
        with self._lock:
            data = self._load_store()
            record = contract.to_dict()
            # Drop raw_output before persisting; keep store lean and the
            # parent context clean. raw_output already truncated upstream.
            record.pop("raw_output", None)
            record["child_session_id"] = child_session_id
            data["delegations"].append(record)
            data["delegations"] = data["delegations"][-100:]  # cap history
            self._save_store(data)
        logger.debug("hermes_plus: stored delegation contract (%s)", contract.summary[:60])

    # -- Problem 3: reflection patch persistence -----------------------------

    def _get_memory_store(self) -> Dict[str, str]:
        return dict(self._load_store().get("memory", {}))

    def _apply_patch(self, patch: MemoryPatch) -> None:
        with self._lock:
            data = self._load_store()
            mem = data.setdefault("memory", {})
            if patch.action == "remove":
                mem.pop(patch.key, None)
            else:  # add | update
                mem[patch.key] = patch.new_value
            self._save_store(data)
        logger.debug("hermes_plus: applied patch %s %s", patch.action, patch.key)

    # -- Optional recall: surface latest summary back into context -----------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        summary = self._get_memory_store().get("_session_summary", "")
        return f"Prior session summary: {summary}" if summary else ""
