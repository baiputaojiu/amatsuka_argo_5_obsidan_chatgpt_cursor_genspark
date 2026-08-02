"""Ch17: Conflict detection (fixed 3-condition algorithm).

Phase 2: the implementation now operates on :class:`GoogleEventView`
internally, but the public ``has_conflict(google_item: dict, ...)`` signature
is preserved so existing callers and tests work unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..constants import TOOL_MARKER
from ..models.google_event import GoogleEventView

__all__ = ["TOOL_MARKER", "COMPARE_FIELDS", "has_conflict"]

COMPARE_FIELDS = ("summary", "description", "location", "visibility", "colorId", "start", "end")


def has_conflict(google_item: Mapping[str, Any], candidate_body: Mapping[str, Any]) -> bool:
    """Return True iff the 3-condition algorithm detects a conflict.

    Condition 1: ``tool_marker`` matches our tool.
    Condition 2: Google ``updated`` is strictly newer than
    ``last_tool_write_utc``.
    Condition 3: At least one sync-target field differs between source
    and Google.
    """
    view = GoogleEventView(google_item)

    # Condition 1
    if not view.is_managed:
        return False

    # Condition 2
    if not view.updated or not view.last_tool_write_utc:
        return False
    if view.updated <= view.last_tool_write_utc:
        return False

    # Condition 3
    for key in COMPARE_FIELDS:
        g_val = _norm(google_item.get(key))
        c_val = _norm(candidate_body.get(key))
        if g_val != c_val:
            return True
    return False


def _norm(val: Any) -> str:
    """Normalize a field value for string comparison."""
    if val is None:
        return ""
    if isinstance(val, dict):
        return str(val.get("dateTime") or val.get("date") or "")
    return str(val)
