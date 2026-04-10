"""Ch17: Conflict detection (fixed 3-condition algorithm)."""


TOOL_MARKER = "outlook_google_sync_v1"

COMPARE_FIELDS = ("summary", "description", "location", "visibility", "colorId", "start", "end")


def has_conflict(google_item: dict, candidate_body: dict) -> bool:
    """Return True if the 3-condition algorithm detects a conflict.

    Condition 1: tool_marker matches our tool.
    Condition 2: Google ``updated`` is strictly newer than ``last_tool_write_utc``.
    Condition 3: At least one sync-target field differs between source and Google.
    """
    private = ((google_item.get("extendedProperties") or {}).get("private") or {})

    # Condition 1
    if private.get("tool_marker") != TOOL_MARKER:
        return False

    # Condition 2
    updated = google_item.get("updated")
    last_write = private.get("last_tool_write_utc")
    if not updated or not last_write:
        return False
    if updated <= last_write:
        return False

    # Condition 3
    for key in COMPARE_FIELDS:
        g_val = _norm(google_item.get(key))
        c_val = _norm(candidate_body.get(key))
        if g_val != c_val:
            return True
    return False


def _norm(val) -> str:
    """Normalize field value for comparison."""
    if val is None:
        return ""
    if isinstance(val, dict):
        return str(val.get("dateTime") or val.get("date") or "")
    return str(val)
