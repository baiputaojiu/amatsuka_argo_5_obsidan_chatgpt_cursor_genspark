"""Unit tests: title_only mode preserves description/location (Ch20, Ch34.2)."""

from datetime import datetime
from outlook_google_sync.models.event import EventModel


def test_title_only_create_excludes_description_location():
    """Ch20.1: title_only does not send description/location on create."""
    e = EventModel(
        sync_key="k1", summary="Meeting",
        start=datetime(2026, 1, 15, 10), end=datetime(2026, 1, 15, 11),
        description="Important notes", location="Room 101",
        reader_engine="ics", input_method="ics_manual",
    )
    body = e.to_google_body("title_only", time_zone="Asia/Tokyo")
    assert "description" not in body
    assert "location" not in body
    assert body["summary"] == "Meeting"
    assert body["start"]["timeZone"] == "Asia/Tokyo"
    assert body["end"]["timeZone"] == "Asia/Tokyo"


def test_full_mode_includes_description_location():
    """Ch20: full mode includes description/location."""
    e = EventModel(
        sync_key="k1", summary="Meeting",
        start=datetime(2026, 1, 15, 10), end=datetime(2026, 1, 15, 11),
        description="Important notes", location="Room 101",
        reader_engine="ics", input_method="ics_manual",
    )
    body = e.to_google_body("full", time_zone="Asia/Tokyo")
    assert body["description"] == "Important notes"
    assert body["location"] == "Room 101"
    assert body["start"]["timeZone"] == "Asia/Tokyo"
    assert body["end"]["timeZone"] == "Asia/Tokyo"


def test_title_only_preserves_existing_google_fields():
    """Ch20.2: title_only update does not include description/location in PATCH body,
    so existing Google values are preserved (not overwritten with empty)."""
    e = EventModel(
        sync_key="k1", summary="Updated Meeting",
        start=datetime(2026, 1, 15, 10), end=datetime(2026, 1, 15, 11),
        description="Should not be sent", location="Should not be sent",
        reader_engine="ics", input_method="ics_manual",
    )
    body = e.to_google_body("title_only", time_zone="Asia/Tokyo")
    assert "description" not in body
    assert "location" not in body
    assert body["start"]["timeZone"] == "Asia/Tokyo"


def test_all_day_event_uses_date_field():
    """Ch10.4: All-day events use date field instead of dateTime."""
    e = EventModel(
        sync_key="k1", summary="Holiday",
        start=datetime(2026, 1, 15), end=datetime(2026, 1, 16),
        is_all_day=True,
        reader_engine="ics", input_method="ics_manual",
    )
    body = e.to_google_body("full", time_zone="Asia/Tokyo")
    assert "date" in body["start"]
    assert "date" in body["end"]
    assert "dateTime" not in body["start"]
    assert "timeZone" not in body["start"]
    assert "timeZone" not in body["end"]


def test_timed_event_without_time_zone_omits_key():
    """後方互換: time_zone 未指定時は Google に timeZone を付けない。"""
    e = EventModel(
        sync_key="k1", summary="Meeting",
        start=datetime(2026, 1, 15, 10), end=datetime(2026, 1, 15, 11),
        reader_engine="ics", input_method="ics_manual",
    )
    body = e.to_google_body("full")
    assert "timeZone" not in body["start"]
    assert "timeZone" not in body["end"]
