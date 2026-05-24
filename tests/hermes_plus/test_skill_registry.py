"""Tests for hermes_plus.skill_registry — scoring, pruning, merge detection."""

from datetime import datetime, timedelta, timezone

import pytest

from hermes_plus import skill_registry as sr


NOW = datetime(2026, 5, 24, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


# ---------------------------------------------------------------------------
# confidence_score
# ---------------------------------------------------------------------------

def test_score_unused_skill_is_low():
    record = {"use_count": 0, "last_used_at": None, "created_at": _iso(NOW)}
    score = sr.confidence_score(record, now=NOW)
    assert score < sr.MIN_CONFIDENCE


def test_score_frequently_used_recent_is_high():
    record = {
        "use_count": 30,
        "last_used_at": _iso(NOW - timedelta(hours=1)),
        "created_at": _iso(NOW - timedelta(days=60)),
    }
    score = sr.confidence_score(record, now=NOW)
    assert score > 0.7


def test_score_recency_decays():
    base = {"use_count": 10, "created_at": _iso(NOW - timedelta(days=30))}
    recent = sr.confidence_score({**base, "last_used_at": _iso(NOW)}, now=NOW)
    stale = sr.confidence_score(
        {**base, "last_used_at": _iso(NOW - timedelta(days=90))}, now=NOW
    )
    assert recent > stale


# ---------------------------------------------------------------------------
# prune_candidates
# ---------------------------------------------------------------------------

def test_prune_candidates_flags_old_unused(monkeypatch):
    usage = {
        "stale-skill": {
            "use_count": 0,
            "last_used_at": None,
            "created_at": _iso(NOW - timedelta(days=120)),
        },
        "active-skill": {
            "use_count": 25,
            "last_used_at": _iso(NOW),
            "created_at": _iso(NOW - timedelta(days=120)),
        },
    }
    monkeypatch.setattr(sr, "load_registry", lambda: {})
    monkeypatch.setattr(
        "tools.skill_usage.load_usage", lambda: usage
    )
    monkeypatch.setattr(
        "tools.skill_usage.list_agent_created_skill_names",
        lambda: list(usage.keys()),
    )

    candidates = sr.prune_candidates(now=NOW)
    names = {c["name"] for c in candidates}
    assert "stale-skill" in names
    assert "active-skill" not in names


def test_prune_skips_pinned(monkeypatch):
    usage = {
        "stale-skill": {
            "use_count": 0,
            "last_used_at": None,
            "created_at": _iso(NOW - timedelta(days=120)),
        },
    }
    monkeypatch.setattr(sr, "load_registry", lambda: {"stale-skill": {"pinned": True}})
    monkeypatch.setattr("tools.skill_usage.load_usage", lambda: usage)
    monkeypatch.setattr(
        "tools.skill_usage.list_agent_created_skill_names",
        lambda: list(usage.keys()),
    )

    candidates = sr.prune_candidates(now=NOW)
    assert candidates == []


def test_prune_skips_recently_created(monkeypatch):
    usage = {
        "new-skill": {
            "use_count": 0,
            "last_used_at": None,
            "created_at": _iso(NOW - timedelta(days=2)),
        },
    }
    monkeypatch.setattr(sr, "load_registry", lambda: {})
    monkeypatch.setattr("tools.skill_usage.load_usage", lambda: usage)
    monkeypatch.setattr(
        "tools.skill_usage.list_agent_created_skill_names",
        lambda: list(usage.keys()),
    )

    assert sr.prune_candidates(now=NOW) == []


# ---------------------------------------------------------------------------
# merge detection
# ---------------------------------------------------------------------------

def test_jaccard_identical_is_one():
    assert sr._jaccard("deploy", "deploy") == 1.0


def test_jaccard_disjoint_is_low():
    assert sr._jaccard("apple", "zzzzz") < 0.1


def test_merge_candidates_flags_similar_names(monkeypatch):
    names = ["deploy-to-prod", "deploy-to-production", "make-coffee"]
    monkeypatch.setattr(
        "tools.skill_usage.list_agent_created_skill_names", lambda: names
    )
    monkeypatch.setattr(sr, "load_registry", lambda: {})

    pairs = sr.merge_candidates()
    flagged = {(p["skill_a"], p["skill_b"]) for p in pairs}
    assert ("deploy-to-prod", "deploy-to-production") in flagged
    assert all("make-coffee" not in pair for pair in flagged)


def test_merge_uses_description_when_names_differ(monkeypatch):
    names = ["alpha", "beta"]
    descriptions = {
        "alpha": "generate quarterly financial report from ledger data",
        "beta": "generate quarterly financial report from ledger data",
    }
    monkeypatch.setattr(
        "tools.skill_usage.list_agent_created_skill_names", lambda: names
    )
    monkeypatch.setattr(sr, "load_registry", lambda: {})

    pairs = sr.merge_candidates(descriptions=descriptions)
    assert any(p["reason"] == "description" for p in pairs)
