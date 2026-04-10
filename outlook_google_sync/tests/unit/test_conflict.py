"""Unit tests: conflict detection (Ch17.1 three-condition) (Ch34.2 mandatory)."""

from outlook_google_sync.sync.conflict import has_conflict

_MARKER = "outlook_google_sync_v1"


def _google_item(summary="Meeting", updated="2026-01-15T12:00:00Z", last_write="2026-01-15T11:00:00Z"):
    return {
        "summary": summary,
        "start": {"dateTime": "2026-01-15T10:00:00"},
        "end": {"dateTime": "2026-01-15T11:00:00"},
        "description": "",
        "location": "",
        "visibility": "default",
        "updated": updated,
        "extendedProperties": {"private": {
            "tool_marker": _MARKER,
            "last_tool_write_utc": last_write,
        }},
    }


def test_conflict_all_three_conditions():
    """Conditions 1+2+3: tool_marker match, updated > last_write, field diff."""
    google = _google_item(summary="Modified Meeting")
    candidate = {"summary": "Original Meeting", "start": {"dateTime": "2026-01-15T10:00:00"}, "end": {"dateTime": "2026-01-15T11:00:00"}, "description": "", "location": "", "visibility": "default"}
    assert has_conflict(google, candidate) is True


def test_no_conflict_tool_marker_mismatch():
    """Condition 1 fails: wrong tool_marker."""
    google = _google_item()
    google["extendedProperties"]["private"]["tool_marker"] = "other_tool"
    candidate = {"summary": "Different"}
    assert has_conflict(google, candidate) is False


def test_no_conflict_updated_not_newer():
    """Condition 2 fails: updated <= last_tool_write_utc."""
    google = _google_item(updated="2026-01-15T10:00:00Z", last_write="2026-01-15T11:00:00Z")
    candidate = {"summary": "Different"}
    assert has_conflict(google, candidate) is False


def test_no_conflict_no_field_diff():
    """Condition 3 fails: no field differences."""
    google = _google_item(summary="Meeting")
    candidate = {
        "summary": "Meeting",
        "start": {"dateTime": "2026-01-15T10:00:00"},
        "end": {"dateTime": "2026-01-15T11:00:00"},
        "description": "",
        "location": "",
        "visibility": "default",
    }
    assert has_conflict(google, candidate) is False
