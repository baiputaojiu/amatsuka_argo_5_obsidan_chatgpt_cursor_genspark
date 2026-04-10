"""Unit tests for duplicate grouping and merge body building."""

from datetime import datetime

from outlook_google_sync.sync.duplicate_merge import build_merged_body
from outlook_google_sync.sync.duplicate_repair import (
    event_start_local_date,
    filter_events_within_start_end_dates,
    find_duplicates,
    find_duplicates_by_summary_time,
    location_merge_allowed,
)
from outlook_google_sync.connectors.google_calendar import TOOL_MARKER


def _managed(sync_key: str = "abc") -> dict:
    return {
        "id": "e1",
        "summary": "Meet",
        "start": {"dateTime": "2025-01-01T10:00:00+09:00"},
        "end": {"dateTime": "2025-01-01T11:00:00+09:00"},
        "extendedProperties": {
            "private": {"tool_marker": TOOL_MARKER, "sync_key": sync_key},
        },
    }


def test_find_duplicates_same_key():
    a = _managed("k1")
    b = {**_managed("k1"), "id": "e2"}
    b["extendedProperties"]["private"]["sync_key"] = "k1"
    out = find_duplicates([a, b])
    assert len(out) == 1
    assert len(out["k1"]) == 2


def test_find_duplicates_by_summary_time():
    a = {
        "id": "1",
        "summary": "X",
        "start": {"date": "2025-01-02"},
        "end": {"date": "2025-01-03"},
    }
    b = {**a, "id": "2"}
    out = find_duplicates_by_summary_time([a, b])
    assert len(out) == 1
    assert len(next(iter(out.values()))) == 2


def test_location_merge_allowed():
    assert location_merge_allowed([{"location": " A "}, {"location": "A"}]) is True
    assert location_merge_allowed([{"location": "Tokyo"}, {"location": "Osaka"}]) is False
    assert location_merge_allowed([{"location": ""}, {"location": "Osaka"}]) is True
    assert location_merge_allowed([{}, {}]) is True


def test_build_merged_body_longer():
    w = _managed()
    w["description"] = "short"
    o = {**_managed(), "id": "x", "description": "much longer text here"}
    body = build_merged_body(
        w,
        description_mode="longer",
        group_items=[w, o],
    )
    assert body["description"] == "much longer text here"


def test_build_merged_body_concat():
    w = _managed()
    w["description"] = "a"
    o = {**_managed(), "id": "x", "description": "b"}
    body = build_merged_body(
        w,
        description_mode="concat",
        group_items=[w, o],
    )
    assert "a" in body["description"] and "b" in body["description"]
    assert "---" in body["description"]


def test_build_merged_body_copies_managed_private():
    w = _managed("sk1")
    u = {
        "id": "u1",
        "summary": "Meet",
        "start": {"dateTime": "2025-01-01T10:00:00+09:00"},
        "end": {"dateTime": "2025-01-01T11:00:00+09:00"},
    }
    body = build_merged_body(
        w,
        description_mode="longer",
        group_items=[w, u],
    )
    priv = (body.get("extendedProperties") or {}).get("private") or {}
    assert priv.get("sync_key") == "sk1"


def test_event_start_local_date_all_day():
    ev = {"start": {"date": "2025-06-01"}, "end": {"date": "2025-06-02"}}
    assert event_start_local_date(ev) is not None
    assert event_start_local_date(ev).isoformat() == "2025-06-01"


def test_filter_events_within_start_end_dates():
    rs = datetime(2025, 1, 10, 0, 0, 0)
    re = datetime(2025, 1, 20, 23, 59, 59)
    ev_in = {"id": "a", "start": {"dateTime": "2025-01-15T10:00:00+09:00"}}
    ev_before = {"id": "b", "start": {"dateTime": "2025-01-05T10:00:00+09:00"}}
    out = filter_events_within_start_end_dates([ev_in, ev_before], rs, re)
    assert [x["id"] for x in out] == ["a"]
