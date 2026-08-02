"""Ch27: Duplicate grouping for repair / merge.

Phase 2: raw-dict access replaced with :class:`GoogleEventView`. Public
function signatures still accept plain dicts for backward compatibility
with callers in ``gui/`` and ``sync/duplicate_merge.py``.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from ..constants import TOOL_MARKER
from ..models.google_event import GoogleEventView


def event_start_local_date(ev: dict) -> date | None:
    """Calendar date of event start in local timezone (for range filtering).

    Kept as a free function for backward compatibility with existing
    imports; delegates to :meth:`GoogleEventView.start_local_date`.
    """
    return GoogleEventView(ev).start_local_date()


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
        view = GoogleEventView(item)
        key = view.sync_key
        if key:
            groups[key].append(item)
    return {k: v for k, v in groups.items() if len(v) > 1}


def _content_group_key(item: dict) -> tuple[str, str, str]:
    view = GoogleEventView(item)
    summ = view.summary.strip().lower()
    return (summ, view.start_value, view.end_value)


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
    return GoogleEventView(item).start_value


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
        view = GoogleEventView(item)
        if view.is_managed and view.sync_key:
            return item
    return None
