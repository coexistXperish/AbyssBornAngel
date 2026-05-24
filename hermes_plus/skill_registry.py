"""Skill Registry — HERMES++ fix for Problem 1 (Skill Explosion).

Adds confidence scoring, TTL-based expiry, similarity-based merge detection,
and a versioned registry on top of the existing skill_usage store.

Confidence score (0.0–1.0) is computed from:
  - use_count   → frequency signal
  - age         → decay over time
  - last_used   → recency signal

Skills below MIN_CONFIDENCE for longer than PRUNE_AFTER_DAYS become
prune candidates; a human confirmation is required before actual deletion.

Merge detection uses trigram Jaccard similarity on skill names + descriptions.
Skills with similarity > MERGE_THRESHOLD are flagged as merge candidates.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Tunables
MIN_CONFIDENCE: float = 0.15
PRUNE_AFTER_DAYS: int = 60
MERGE_THRESHOLD: float = 0.55
MAX_USE_COUNT_CAP: int = 100  # sigmoid saturation point

_REGISTRY_FILE = "skills/.skill_registry.json"


# ---------------------------------------------------------------------------
# Registry I/O
# ---------------------------------------------------------------------------

def _registry_path() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / _REGISTRY_FILE
    except ImportError:
        return Path.home() / ".hermes" / _REGISTRY_FILE


def load_registry() -> Dict[str, Any]:
    path = _registry_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("skill_registry: failed to load %s: %s", path, e)
        return {}


def save_registry(data: Dict[str, Any]) -> None:
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def get_entry(skill_name: str) -> Dict[str, Any]:
    reg = load_registry()
    entry = reg.get(skill_name, {})
    entry.setdefault("versions", [])
    entry.setdefault("pinned", False)
    entry.setdefault("merge_target", None)
    return entry


def set_entry(skill_name: str, entry: Dict[str, Any]) -> None:
    reg = load_registry()
    reg[skill_name] = entry
    save_registry(reg)


def record_version(skill_name: str, content_hash: str, description: str = "") -> None:
    """Append a version snapshot when a skill is patched."""
    entry = get_entry(skill_name)
    entry["versions"].append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "hash": content_hash,
        "description": description,
    })
    # Keep last 20 versions
    entry["versions"] = entry["versions"][-20:]
    set_entry(skill_name, entry)


def pin(skill_name: str) -> None:
    """Pin a skill — it will never be auto-pruned or auto-merged."""
    entry = get_entry(skill_name)
    entry["pinned"] = True
    set_entry(skill_name, entry)


def unpin(skill_name: str) -> None:
    entry = get_entry(skill_name)
    entry["pinned"] = False
    set_entry(skill_name, entry)


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def confidence_score(record: Dict[str, Any], now: Optional[datetime] = None) -> float:
    """Return a 0.0–1.0 confidence score for a skill usage record.

    Three components, equally weighted:
      - frequency:  sigmoid of use_count
      - recency:    exponential decay from last_used_at (half-life = 30 days)
      - age bonus:  mild reward for surviving > 14 days without being pruned
    """
    if now is None:
        now = datetime.now(timezone.utc)

    use_count: int = record.get("use_count", 0)
    last_used_at: Optional[str] = record.get("last_used_at")
    created_at: Optional[str] = record.get("created_at")

    # A skill we have literally never used has zero confidence, regardless
    # of how long it has existed. Without this guard, the age component
    # rescues abandoned skills — the exact ones we want to prune.
    if use_count <= 0 and not last_used_at:
        return 0.0

    # Frequency: sigmoid centred at 5 uses
    freq = 1.0 / (1.0 + math.exp(-(use_count - 5) / 3.0))

    # Recency: decay from last use; if never used, score 0
    last_used = _parse_iso(last_used_at)
    if last_used is None:
        recency = 0.0
    else:
        if last_used.tzinfo is None:
            last_used = last_used.replace(tzinfo=timezone.utc)
        days_since = max(0.0, (now - last_used).total_seconds() / 86400)
        recency = math.exp(-days_since / 30.0)

    # Age bonus: reward skills that have existed > 14 days
    created = _parse_iso(created_at)
    if created is None:
        age_bonus = 0.0
    else:
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - created).total_seconds() / 86400)
        age_bonus = min(1.0, age_days / 90.0)  # saturates at 90 days

    return round((freq + recency + age_bonus) / 3.0, 4)


# ---------------------------------------------------------------------------
# Prune candidates
# ---------------------------------------------------------------------------

def prune_candidates(now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Return skills that are below MIN_CONFIDENCE and old enough to prune.

    Does NOT delete anything — returns a list of candidates for human review.
    Each item: {name, confidence, last_used_at, use_count, pinned}.
    """
    try:
        from tools.skill_usage import load_usage, list_agent_created_skill_names
    except ImportError:
        logger.warning("skill_registry: tools.skill_usage not available")
        return []

    if now is None:
        now = datetime.now(timezone.utc)

    usage = load_usage()
    agent_names: Set[str] = set(list_agent_created_skill_names())
    registry = load_registry()
    candidates = []

    for name, record in usage.items():
        if name not in agent_names:
            continue
        entry = registry.get(name, {})
        if entry.get("pinned", False):
            continue

        score = confidence_score(record, now=now)
        if score >= MIN_CONFIDENCE:
            continue

        # Check age — only prune if skill has been around long enough
        created = _parse_iso(record.get("created_at"))
        if created is None:
            # No created_at means it pre-dates tracking; include
            age_days = PRUNE_AFTER_DAYS + 1
        else:
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_days = (now - created).total_seconds() / 86400

        if age_days >= PRUNE_AFTER_DAYS:
            candidates.append({
                "name": name,
                "confidence": score,
                "last_used_at": record.get("last_used_at"),
                "use_count": record.get("use_count", 0),
                "pinned": False,
            })

    candidates.sort(key=lambda x: x["confidence"])
    return candidates


# ---------------------------------------------------------------------------
# Similarity / merge detection
# ---------------------------------------------------------------------------

def _trigrams(text: str) -> Set[str]:
    text = re.sub(r"[^a-z0-9]", "", text.lower())
    if len(text) < 3:
        return {text} if text else set()
    return {text[i:i+3] for i in range(len(text) - 2)}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def merge_candidates(descriptions: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """Return pairs of agent-created skills that are likely duplicates.

    descriptions: optional {skill_name: description_text} to augment name similarity.
    Each item: {skill_a, skill_b, similarity, reason}.
    """
    try:
        from tools.skill_usage import list_agent_created_skill_names
    except ImportError:
        logger.warning("skill_registry: tools.skill_usage not available")
        return []

    names = list_agent_created_skill_names()
    descriptions = descriptions or {}
    registry = load_registry()

    pairs: List[Dict[str, Any]] = []
    for i, a in enumerate(names):
        if registry.get(a, {}).get("merge_target"):
            continue  # already scheduled
        for b in names[i+1:]:
            if registry.get(b, {}).get("merge_target"):
                continue

            name_sim = _jaccard(a, b)
            desc_a = descriptions.get(a, "")
            desc_b = descriptions.get(b, "")
            desc_sim = _jaccard(desc_a, desc_b) if desc_a and desc_b else 0.0

            similarity = max(name_sim, desc_sim)
            if similarity >= MERGE_THRESHOLD:
                reason = "name" if name_sim >= desc_sim else "description"
                pairs.append({
                    "skill_a": a,
                    "skill_b": b,
                    "similarity": round(similarity, 3),
                    "reason": reason,
                })

    pairs.sort(key=lambda x: -x["similarity"])
    return pairs


# ---------------------------------------------------------------------------
# Registry report (for curator integration)
# ---------------------------------------------------------------------------

def registry_report() -> Dict[str, Any]:
    """Produce a summary dict suitable for logging or display."""
    prune = prune_candidates()
    merge = merge_candidates()
    return {
        "prune_candidates": prune,
        "merge_candidates": merge,
        "prune_count": len(prune),
        "merge_count": len(merge),
    }
