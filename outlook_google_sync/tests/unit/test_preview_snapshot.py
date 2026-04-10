"""Unit tests: preview snapshot (Ch34.2 mandatory)."""

from datetime import UTC, datetime, timedelta
from outlook_google_sync.models.event import EventModel
from outlook_google_sync.sync.preview import PreviewSnapshot, build_preview, STALE_THRESHOLD_MINUTES


def _event(key="k1", summary="Test"):
    return EventModel(
        sync_key=key, summary=summary,
        start=datetime(2026, 1, 15, 10), end=datetime(2026, 1, 15, 11),
        fingerprint="fp1", reader_engine="ics", input_method="ics_manual",
    )


def test_preview_classifies_new_event():
    """New event not in existing → create."""
    snap = build_preview([_event()], {}, {})
    assert len(snap.created) == 1
    assert snap.created[0].action == "create"


def test_preview_classifies_update():
    """Event in existing with different fingerprint → update."""
    existing = {"k1": {
        "id": "g1",
        "summary": "Test",
        "start": {"dateTime": "2026-01-15T10:00:00"},
        "end": {"dateTime": "2026-01-15T11:00:00"},
        "extendedProperties": {"private": {"tool_marker": "outlook_google_sync_v1"}},
    }}
    snap = build_preview([_event()], existing, {"k1": "old_fp"})
    assert len(snap.updated) == 1


def test_preview_classifies_skip():
    """Event with same fingerprint and no Google edit → skip."""
    existing = {"k1": {
        "id": "g1",
        "summary": "Test",
        "start": {"dateTime": "2026-01-15T10:00:00"},
        "end": {"dateTime": "2026-01-15T11:00:00"},
        "updated": "2026-01-15T09:00:00Z",
        "extendedProperties": {"private": {
            "tool_marker": "outlook_google_sync_v1",
            "last_tool_write_utc": "2026-01-15T10:00:00Z",
        }},
    }}
    snap = build_preview([_event()], existing, {"k1": "fp1"})
    assert len(snap.skipped) == 1


def test_snapshot_staleness():
    """PRE-SNAP-03: snapshot is stale after threshold."""
    snap = PreviewSnapshot()
    snap.generated_at = datetime.now(UTC) - timedelta(minutes=STALE_THRESHOLD_MINUTES + 1)
    assert snap.is_stale() is True

    fresh = PreviewSnapshot()
    assert fresh.is_stale() is False
