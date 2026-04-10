"""Ch18: Merge logic for existing events."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.profile import MergeSettings


def merge_fields(
    source_body: dict,
    google_item: dict,
    settings: MergeSettings | None = None,
) -> dict:
    """Merge source fields into Google event body respecting priority rules (Ch18.4)."""
    out = dict(google_item)

    # Always use source for datetime and summary (Ch18.4)
    for key in ("summary", "start", "end"):
        if source_body.get(key):
            out[key] = source_body[key]

    if settings:
        # description priority
        if settings.description_priority == "source" and source_body.get("description"):
            out["description"] = source_body["description"]
        # location: non-empty wins, then setting
        src_loc = source_body.get("location", "")
        goo_loc = google_item.get("location", "")
        if src_loc and goo_loc:
            out["location"] = src_loc if settings.location_priority == "source" else goo_loc
        elif src_loc:
            out["location"] = src_loc
    else:
        # default: source truthy values override
        for key, val in source_body.items():
            if val:
                out[key] = val

    # Preserve extendedProperties from source
    if "extendedProperties" in source_body:
        out["extendedProperties"] = source_body["extendedProperties"]

    return out


def is_merge_candidate(
    source_summary: str,
    source_start_iso: str,
    source_end_iso: str,
    google_item: dict,
    tolerance_minutes: int = 2,
) -> bool:
    """Ch18.1: Check if a Google event is a merge candidate."""
    from datetime import datetime, timedelta

    private = ((google_item.get("extendedProperties") or {}).get("private") or {})
    if private.get("tool_marker"):
        return False  # Already managed, not a merge candidate

    g_summary = (google_item.get("summary") or "").strip().lower()
    if g_summary != source_summary.strip().lower():
        return False

    g_start = _extract_dt(google_item.get("start", {}))
    g_end = _extract_dt(google_item.get("end", {}))
    if not g_start or not g_end:
        return False

    try:
        s_start = datetime.fromisoformat(source_start_iso)
        s_end = datetime.fromisoformat(source_end_iso)
    except (ValueError, TypeError):
        return False

    tol = timedelta(minutes=tolerance_minutes)
    return abs(g_start - s_start) <= tol and abs(g_end - s_end) <= tol


def _extract_dt(val: dict):
    from datetime import datetime
    raw = val.get("dateTime") or val.get("date")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except (ValueError, TypeError):
        return None
