"""Ch27: Duplicate grouping for repair / merge."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Literal

from ..connectors.google_calendar import TOOL_MARKER


def event_start_local_date(ev: dict) -> date | None:
    """Calendar date of event start in local timezone (for range filtering)."""
    st = ev.get("start") or {}
    if st.get("date"):
        return date.fromisoformat(str(st["date"]))
    raw = st.get("dateTime")
    if not raw:
        return None
    s = str(raw).replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone().date()


def filter_events_within_start_end_dates(
    events: list[dict],
    range_start: datetime,
    range_end: datetime,
) -> list[dict]:
    """Keep events whose **start** (local calendar day) lies in range_start.date()…range_end.date() inclusive.

    API ``timeMin``/``timeMax`` can still return overlapping instances; this narrows to the user's date range.
    """
    d0 = range_start.date()
    d1 = range_end.date()
    out: list[dict] = []
    for ev in events:
        sd = event_start_local_date(ev)
        if sd is None:
            continue
        if d0 <= sd <= d1:
            out.append(ev)
    return out


@dataclass(frozen=True)
class DuplicateGroup:
    """One duplicate group with merge eligibility."""

    group_id: str
    items: list[dict]
    mergeable: bool
    blocked_reason: str | None  # e.g. "location_mismatch"


def find_duplicates(events: list) -> dict[str, list]:
    """Group tool-managed events by sync_key; return groups with 2+ items."""
    groups: dict[str, list] = defaultdict(list)
    for item in events:
        private = ((item.get("extendedProperties") or {}).get("private") or {})
        key = private.get("sync_key")
        if key:
            groups[key].append(item)
    return {k: v for k, v in groups.items() if len(v) > 1}


def _content_group_key(item: dict) -> tuple[str, str, str]:
    summ = (item.get("summary") or "").strip().lower()
    st = item.get("start") or {}
    en = item.get("end") or {}
    s_key = st.get("dateTime") or st.get("date") or ""
    e_key = en.get("dateTime") or en.get("date") or ""
    return (summ, str(s_key), str(e_key))


def find_duplicates_by_summary_time(events: list) -> dict[str, list]:
    """Group by normalized summary + start + end; return groups with 2+ items."""
    groups: dict[tuple[str, str, str], list] = defaultdict(list)
    for item in events:
        groups[_content_group_key(item)].append(item)
    out: dict[str, list] = {}
    for k, items in groups.items():
        if len(items) <= 1:
            continue
        h = hashlib.sha256("|".join(k).encode("utf-8")).hexdigest()[:24]
        out[f"content:{h}"] = items
    return out


def _norm_loc(item: dict) -> str:
    loc = item.get("location")
    if not loc:
        return ""
    return str(loc).strip()


def _start_sort_key(item: dict) -> str:
    st = item.get("start") or {}
    return str(st.get("dateTime") or st.get("date") or "")


def pick_winner_event_id(items: list[dict]) -> str:
    """Pick representative event id: earliest start, then first row."""
    sorted_items = sorted(items, key=_start_sort_key)
    for it in sorted_items:
        eid = str(it.get("id", "")).strip()
        if eid:
            return eid
    return ""


def location_merge_allowed(items: list[dict]) -> bool:
    """False if two or more distinct non-empty locations exist."""
    distinct: set[str] = set()
    for it in items:
        n = _norm_loc(it)
        if n:
            distinct.add(n)
        if len(distinct) > 1:
            return False
    return True


def groups_to_duplicate_groups(
    raw: dict[str, list],
) -> list[DuplicateGroup]:
    """Attach mergeable / blocked_reason to each group."""
    result: list[DuplicateGroup] = []
    for gid, items in raw.items():
        ok = location_merge_allowed(items)
        result.append(
            DuplicateGroup(
                group_id=gid,
                items=items,
                mergeable=ok,
                blocked_reason=None if ok else "location_mismatch",
            )
        )
    return result


def build_groups_for_mode(
    mode: Literal["sync_key", "content"],
    events: list[dict],
) -> list[DuplicateGroup]:
    if mode == "sync_key":
        raw = find_duplicates(events)
    else:
        raw = find_duplicates_by_summary_time(events)
    return groups_to_duplicate_groups(raw)


def pick_managed_source_for_private(items: list[dict]) -> dict | None:
    """First item with tool_marker + sync_key, or None."""
    for item in items:
        private = ((item.get("extendedProperties") or {}).get("private") or {})
        if private.get("tool_marker") == TOOL_MARKER and private.get("sync_key"):
            return item
    return None
