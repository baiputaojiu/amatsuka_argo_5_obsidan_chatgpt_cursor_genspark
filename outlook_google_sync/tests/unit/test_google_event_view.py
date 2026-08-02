"""Unit tests for the :class:`GoogleEventView` value object (Phase 2)."""

from __future__ import annotations

from datetime import datetime

import pytest

from outlook_google_sync.constants import TOOL_MARKER
from outlook_google_sync.models.google_event import (
    GoogleEventView,
    parse_event_datetime,
    to_local_naive,
)


# ────────────────────────────────────────────────────────────────────
# Basic accessors
# ────────────────────────────────────────────────────────────────────

def test_empty_event_returns_empty_strings():
    view = GoogleEventView({})
    assert view.id == ""
    assert view.summary == ""
    assert view.description == ""
    assert view.location == ""
    assert view.visibility == ""
    assert view.color_id == ""
    assert view.updated == ""
    assert view.attendees == []


def test_accessors_pull_raw_fields():
    item = {
        "id": "abc123",
        "summary": "Morning Meeting",
        "description": "weekly sync",
        "location": "Room 5",
        "visibility": "private",
        "colorId": "7",
        "updated": "2026-04-20T09:00:00.000Z",
    }
    view = GoogleEventView(item)
    assert view.id == "abc123"
    assert view.summary == "Morning Meeting"
    assert view.description == "weekly sync"
    assert view.location == "Room 5"
    assert view.visibility == "private"
    assert view.color_id == "7"
    assert view.updated == "2026-04-20T09:00:00.000Z"


def test_raw_round_trips_and_is_not_copied():
    item = {"id": "x"}
    view = GoogleEventView(item)
    assert view.raw is item  # zero-copy contract


# ────────────────────────────────────────────────────────────────────
# extendedProperties.private
# ────────────────────────────────────────────────────────────────────

def test_private_props_absent():
    assert dict(GoogleEventView({}).private_props) == {}
    assert dict(GoogleEventView({"extendedProperties": None}).private_props) == {}
    assert dict(GoogleEventView({"extendedProperties": {"private": None}}).private_props) == {}


def test_private_props_accessors():
    item = {
        "extendedProperties": {
            "private": {
                "tool_marker": TOOL_MARKER,
                "sync_key": "abc",
                "sync_key_kind": "id",
                "reader_engine": "com",
                "input_method": "com",
                "last_tool_write_utc": "2026-04-20T08:00:00Z",
            }
        }
    }
    view = GoogleEventView(item)
    assert view.tool_marker == TOOL_MARKER
    assert view.sync_key == "abc"
    assert view.sync_key_kind == "id"
    assert view.reader_engine == "com"
    assert view.input_method == "com"
    assert view.last_tool_write_utc == "2026-04-20T08:00:00Z"
    assert view.is_managed is True


def test_is_managed_false_when_marker_missing_or_wrong():
    assert GoogleEventView({}).is_managed is False
    assert GoogleEventView(
        {"extendedProperties": {"private": {"tool_marker": "other_tool"}}}
    ).is_managed is False


# ────────────────────────────────────────────────────────────────────
# start / end
# ────────────────────────────────────────────────────────────────────

def test_start_end_value_timed():
    item = {
        "start": {"dateTime": "2026-04-20T10:00:00+09:00"},
        "end": {"dateTime": "2026-04-20T11:00:00+09:00"},
    }
    view = GoogleEventView(item)
    assert view.start_value == "2026-04-20T10:00:00+09:00"
    assert view.end_value == "2026-04-20T11:00:00+09:00"
    assert view.is_all_day is False
    assert view.start_dt is not None
    assert view.end_dt is not None
    assert view.start_dt.hour == 10


def test_start_end_value_all_day():
    item = {"start": {"date": "2026-04-20"}, "end": {"date": "2026-04-21"}}
    view = GoogleEventView(item)
    assert view.start_value == "2026-04-20"
    assert view.is_all_day is True
    sd = view.start_local_date()
    assert sd is not None
    assert sd.isoformat() == "2026-04-20"


def test_start_value_when_missing():
    assert GoogleEventView({}).start_value == ""
    assert GoogleEventView({"start": {}}).start_value == ""


def test_parse_event_datetime_handles_z_suffix():
    dt = parse_event_datetime({"dateTime": "2026-04-20T10:00:00Z"})
    assert dt is not None
    assert dt.tzinfo is not None


def test_parse_event_datetime_returns_none_for_bad_input():
    assert parse_event_datetime(None) is None
    assert parse_event_datetime({}) is None
    assert parse_event_datetime({"dateTime": ""}) is None
    assert parse_event_datetime({"dateTime": "not-a-date"}) is None


# ────────────────────────────────────────────────────────────────────
# overlaps
# ────────────────────────────────────────────────────────────────────

def test_overlaps_inside_range():
    item = {
        "start": {"dateTime": "2026-04-20T10:00:00"},
        "end": {"dateTime": "2026-04-20T11:00:00"},
    }
    view = GoogleEventView(item)
    assert view.overlaps(datetime(2026, 4, 20, 0, 0), datetime(2026, 4, 20, 23, 59)) is True


def test_overlaps_outside_range():
    item = {
        "start": {"dateTime": "2026-04-20T10:00:00"},
        "end": {"dateTime": "2026-04-20T11:00:00"},
    }
    view = GoogleEventView(item)
    assert view.overlaps(datetime(2026, 4, 21, 0, 0), datetime(2026, 4, 21, 23, 59)) is False


def test_overlaps_conservative_when_unparseable():
    view = GoogleEventView({"start": {}, "end": {}})
    # Unknown → include (conservative)
    assert view.overlaps(datetime(2026, 4, 20, 0, 0), datetime(2026, 4, 21, 0, 0)) is True


def test_overlaps_with_aware_event_and_naive_range():
    item = {
        "start": {"dateTime": "2026-04-20T10:00:00+09:00"},
        "end": {"dateTime": "2026-04-20T11:00:00+09:00"},
    }
    view = GoogleEventView(item)
    # aware datetime should be converted to local naive for comparison
    assert view.overlaps(datetime(2026, 4, 20, 0, 0), datetime(2026, 4, 20, 23, 59)) is True


# ────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────

def test_to_local_naive_passthrough_for_naive():
    dt = datetime(2026, 4, 20, 10, 0)
    assert to_local_naive(dt) is dt


def test_to_local_naive_strips_tz():
    dt = datetime.fromisoformat("2026-04-20T10:00:00+09:00")
    out = to_local_naive(dt)
    assert out.tzinfo is None


def test_wrap_many():
    items = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    views = GoogleEventView.wrap_many(items)
    assert len(views) == 3
    assert [v.id for v in views] == ["a", "b", "c"]


def test_repr_includes_key_fields():
    view = GoogleEventView(
        {
            "id": "xyz",
            "summary": "Hi",
            "extendedProperties": {
                "private": {"tool_marker": TOOL_MARKER, "sync_key": "k1"}
            },
        }
    )
    r = repr(view)
    assert "xyz" in r
    assert "Hi" in r
    assert "k1" in r
    assert "managed=True" in r


def test_get_passthrough():
    view = GoogleEventView({"foo": "bar"})
    assert view.get("foo") == "bar"
    assert view.get("missing", "fallback") == "fallback"
