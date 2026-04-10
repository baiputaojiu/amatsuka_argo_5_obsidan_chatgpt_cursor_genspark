"""Unit tests: differential sync Google-side edit detection (DIFF-GC-01~03) (Ch34.2 mandatory)."""

from datetime import datetime
from outlook_google_sync.models.event import EventModel
from outlook_google_sync.sync.diff_sync import filter_diff_targets


def _event(key="k1", fp="fp1"):
    return EventModel(
        sync_key=key, summary="Test",
        start=datetime(2026, 1, 15, 10), end=datetime(2026, 1, 15, 11),
        fingerprint=fp, reader_engine="ics", input_method="ics_manual",
    )


def test_diff_includes_google_manually_edited_event():
    """DIFF-GC-01: Even with same fingerprint, if Google updated > last_tool_write_utc, must not skip."""
    e = _event("k1", "fp1")
    prev = {"k1": "fp1"}  # same fingerprint → would normally skip
    existing = {"k1": {
        "id": "g1",
        "updated": "2026-01-15T14:00:00Z",
        "extendedProperties": {"private": {
            "tool_marker": "outlook_google_sync_v1",
            "last_tool_write_utc": "2026-01-15T12:00:00Z",
        }},
    }}
    result = filter_diff_targets([e], prev, existing)
    assert len(result) == 1, "Google-edited event must not be skipped"


def test_diff_skips_unchanged_event():
    """Normal case: same fingerprint, Google not edited → skip."""
    e = _event("k1", "fp1")
    prev = {"k1": "fp1"}
    existing = {"k1": {
        "id": "g1",
        "updated": "2026-01-15T10:00:00Z",
        "extendedProperties": {"private": {
            "tool_marker": "outlook_google_sync_v1",
            "last_tool_write_utc": "2026-01-15T11:00:00Z",
        }},
    }}
    result = filter_diff_targets([e], prev, existing)
    assert len(result) == 0, "Unchanged event should be skipped"


def test_diff_includes_source_changed_event():
    """Source fingerprint changed → must be included."""
    e = _event("k1", "new_fp")
    prev = {"k1": "old_fp"}
    existing = {"k1": {"id": "g1"}}
    result = filter_diff_targets([e], prev, existing)
    assert len(result) == 1
