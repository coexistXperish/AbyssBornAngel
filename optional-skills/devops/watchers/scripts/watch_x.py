#!/usr/bin/env python3
"""Watch X (Twitter) accounts via nitter RSS; print new posts to stdout, silent on empty.

Uses nitter — an open-source Twitter frontend that exposes RSS for any public account.
No API key required. Falls back to X API v2 Bearer token if --x-api-bearer is set.

Usage (via cron with --no-agent):

    hermes cron create hermes-devwatch \\
      --schedule "0 */4 * * *" --no-agent \\
      --script "$HERMES_HOME/skills/devops/watchers/scripts/watch_x.py" \\
      --script-args "--name nous-devs --handles teknium nousresearch"

    # With custom nitter instance:
    --script-args "--name nous-devs --handles teknium --nitter https://nitter.privacydev.net"

    # With X API v2 (requires Bearer token):
    --script-args "--name nous-devs --handles teknium --x-api-bearer $XAI_BEARER_TOKEN"

First run records a baseline (emits nothing). Subsequent runs emit only
posts whose tweet ID is not in the watermark.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).parent))
from _watermark import Watermark, format_items_as_markdown  # type: ignore

# Public nitter instances — tried in order, first success wins.
DEFAULT_NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
]


# ---------------------------------------------------------------------------
# Nitter RSS path
# ---------------------------------------------------------------------------

def _nitter_rss_url(handle: str, base: str) -> str:
    handle = handle.lstrip("@")
    base = base.rstrip("/")
    return f"{base}/{handle}/rss"


def _fetch_url(url: str, timeout: float, bearer: str = "") -> bytes:
    headers: dict = {"User-Agent": "Hermes-Watcher/1.0"}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _parse_rss(xml_bytes: bytes, handle: str) -> list[dict]:
    """Parse nitter RSS and return [{id, title, url, summary}]."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        print(f"watch_x: XML parse error for @{handle}: {e}", file=sys.stderr)
        return []

    entries = []
    for item in root.iter():
        if _strip_ns(item.tag) != "item":
            continue
        children = {_strip_ns(c.tag): c for c in item}

        guid_el = children.get("guid")
        link_el = children.get("link")
        href = (link_el.text or "").strip() if link_el is not None else ""
        guid = (guid_el.text or "").strip() if guid_el is not None else href
        if not guid:
            continue

        # Extract tweet ID from URL for a stable watermark key
        # e.g. https://nitter.net/teknium/status/12345 → "12345"
        tweet_id = guid.rstrip("/").rsplit("/", 1)[-1].split("#")[0]
        tweet_id = tweet_id if tweet_id.isdigit() else guid

        title_el = children.get("title")
        title = (title_el.text or "").strip() if title_el is not None else ""

        desc_el = children.get("description")
        summary = (desc_el.text or "").strip() if desc_el is not None else ""
        # Strip HTML tags from nitter descriptions (basic)
        import re
        summary = re.sub(r"<[^>]+>", "", summary).strip()

        entries.append({
            "id": tweet_id,
            "title": f"@{handle}: {title}" if title else f"@{handle}",
            "url": href.replace(href.split("/")[2], "x.com", 1) if href else "",
            "summary": summary,
        })
    return entries


def _fetch_handle_nitter(
    handle: str,
    instances: list[str],
    timeout: float,
) -> list[dict]:
    """Try nitter instances in order until one succeeds."""
    for base in instances:
        url = _nitter_rss_url(handle, base)
        try:
            xml_bytes = _fetch_url(url, timeout)
            entries = _parse_rss(xml_bytes, handle)
            return entries
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            continue
    print(f"watch_x: all nitter instances failed for @{handle}", file=sys.stderr)
    return []


# ---------------------------------------------------------------------------
# X API v2 fallback
# ---------------------------------------------------------------------------

def _fetch_handle_api(handle: str, bearer: str, timeout: float) -> list[dict]:
    """Fetch recent tweets via X API v2. Requires Bearer token."""
    import json

    # Look up user ID first
    lookup_url = f"https://api.twitter.com/2/users/by/username/{handle}"
    try:
        data = json.loads(_fetch_url(lookup_url, timeout, bearer=bearer))
        user_id = data.get("data", {}).get("id")
        if not user_id:
            print(f"watch_x: user not found via API: @{handle}", file=sys.stderr)
            return []
    except Exception as e:
        print(f"watch_x: X API user lookup failed for @{handle}: {e}", file=sys.stderr)
        return []

    tweets_url = (
        f"https://api.twitter.com/2/users/{user_id}/tweets"
        "?max_results=10&tweet.fields=created_at,text"
    )
    try:
        data = json.loads(_fetch_url(tweets_url, timeout, bearer=bearer))
    except Exception as e:
        print(f"watch_x: X API tweets fetch failed for @{handle}: {e}", file=sys.stderr)
        return []

    entries = []
    for tweet in data.get("data", []):
        tid = tweet.get("id", "")
        text = tweet.get("text", "")
        entries.append({
            "id": tid,
            "title": f"@{handle}: {text[:80]}",
            "url": f"https://x.com/{handle}/status/{tid}",
            "summary": text,
        })
    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Watch X/Twitter accounts for new posts."
    )
    p.add_argument("--name", required=True,
                   help="Watcher name (used for state file, must be unique per watcher job)")
    p.add_argument("--handles", nargs="+", required=True,
                   help="X/Twitter handles to watch (without @), e.g. --handles teknium nousresearch")
    p.add_argument("--nitter", default=None,
                   help="Nitter instance base URL (default: try public instances)")
    p.add_argument("--x-api-bearer", default="",
                   help="X API v2 Bearer token (skips nitter and uses official API instead)")
    p.add_argument("--max", type=int, default=10,
                   help="Max new posts to emit per tick per handle (default: 10)")
    p.add_argument("--with-summary", action="store_true",
                   help="Include tweet text under the title")
    p.add_argument("--timeout", type=float, default=15.0,
                   help="HTTP timeout in seconds (default: 15)")
    args = p.parse_args()

    nitter_instances = [args.nitter] if args.nitter else DEFAULT_NITTER_INSTANCES

    all_new: list[dict] = []

    for handle in args.handles:
        handle = handle.lstrip("@")
        if args.x_api_bearer:
            entries = _fetch_handle_api(handle, args.x_api_bearer, args.timeout)
        else:
            entries = _fetch_handle_nitter(handle, nitter_instances, args.timeout)

        if not entries:
            continue

        # Per-handle watermark (name scoped by handle so handles don't collide)
        wm_name = f"{args.name}-{handle}"
        wm = Watermark.load(wm_name)
        new_items = wm.filter_new(entries, id_key="id")
        wm.save()

        if args.max > 0:
            new_items = new_items[: args.max]
        all_new.extend(new_items)

    body_key = "summary" if args.with_summary else None
    output = format_items_as_markdown(all_new, body_key=body_key)
    if output:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
