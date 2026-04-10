"""Unit tests: delete candidate extraction (DEL-01~05) (Ch34.2 mandatory)."""

from datetime import datetime
from outlook_google_sync.sync.delete_candidates import select_delete_candidates

_MARKER = "outlook_google_sync_v1"


def _make_item(sync_key, start_iso="2026-01-15T10:00:00", end_iso="2026-01-15T11:00:00"):
    return {
        "id": f"evt_{sync_key}",
        "start": {"dateTime": start_iso},
        "end": {"dateTime": end_iso},
        "summary": f"Event {sync_key}",
        "extendedProperties": {"private": {
            "tool_marker": _MARKER,
            "sync_key": sync_key,
        }},
    }


def test_del01_only_tool_marker_items():
    """DEL-01: Only tool_marker matching items are candidates."""
    existing = {
        "k1": _make_item("k1"),
        "k2": {
            "id": "evt_k2",
            "start": {"dateTime": "2026-01-15T10:00:00"},
            "end": {"dateTime": "2026-01-15T11:00:00"},
            "extendedProperties": {"private": {"tool_marker": "other_tool", "sync_key": "k2"}},
        },
    }
    result = select_delete_candidates(existing, [])
    assert len(result) == 1
    assert result[0]["id"] == "evt_k1"


def test_del02_range_overlap():
    """DEL-02: Only events overlapping R are candidates."""
    existing = {
        "k1": _make_item("k1", "2026-01-15T10:00:00", "2026-01-15T11:00:00"),
        "k2": _make_item("k2", "2026-02-01T10:00:00", "2026-02-01T11:00:00"),
    }
    r_start = datetime(2026, 1, 1)
    r_end = datetime(2026, 1, 31, 23, 59, 59)
    result = select_delete_candidates(existing, [], r_start, r_end)
    assert len(result) == 1
    assert result[0]["id"] == "evt_k1"


def test_del03_missing_keys_only():
    """DEL-03: Only items whose sync_key is not in current_keys."""
    existing = {
        "k1": _make_item("k1"),
        "k2": _make_item("k2"),
    }
    result = select_delete_candidates(existing, ["k1"])
    assert len(result) == 1
    assert result[0]["id"] == "evt_k2"


def test_del05_moved_outside_range():
    """DEL-05: Events moved outside R are not candidates."""
    existing = {
        "k1": _make_item("k1", "2026-03-01T10:00:00", "2026-03-01T11:00:00"),
    }
    r_start = datetime(2026, 1, 1)
    r_end = datetime(2026, 1, 31, 23, 59, 59)
    result = select_delete_candidates(existing, [], r_start, r_end)
    assert len(result) == 0
