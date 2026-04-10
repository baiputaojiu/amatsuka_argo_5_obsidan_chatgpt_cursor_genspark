"""Unit tests: event filtering (Ch8)."""

from datetime import datetime
from outlook_google_sync.models.event import EventModel
from outlook_google_sync.models.profile import FilterConfig
from outlook_google_sync.sync.filters import apply_filters


def _event(summary="Test", is_all_day=False, busy=2, is_private=False, categories=None, location=""):
    return EventModel(
        sync_key="k1", summary=summary,
        start=datetime(2026, 1, 15, 10), end=datetime(2026, 1, 15, 11),
        is_all_day=is_all_day, busy_status=busy, is_private=is_private,
        categories=categories or [], location=location,
        reader_engine="ics", input_method="ics_manual",
    )


def test_exclude_all_day():
    fc = FilterConfig(exclude_all_day=True)
    events = [_event(is_all_day=True), _event(is_all_day=False)]
    result = apply_filters(events, fc, True)
    assert len(result) == 1
    assert not result[0].is_all_day


def test_exclude_free():
    fc = FilterConfig(exclude_free=True)
    events = [_event(busy=0), _event(busy=2)]
    result = apply_filters(events, fc, True)
    assert len(result) == 1
    assert result[0].busy_status == 2


def test_exclude_tentative():
    fc = FilterConfig(exclude_tentative=True)
    events = [_event(busy=1), _event(busy=2)]
    result = apply_filters(events, fc, True)
    assert len(result) == 1


def test_exclude_private_takes_priority():
    """Ch8.3-4: Private exclusion takes priority over include_private."""
    fc = FilterConfig(exclude_private=True)
    events = [_event(is_private=True), _event(is_private=False)]
    result = apply_filters(events, fc, include_private=True)
    assert len(result) == 1
    assert not result[0].is_private


def test_subject_keyword_exclusion():
    fc = FilterConfig(subject_keywords=["cancel"])
    events = [_event(summary="Cancelled Meeting"), _event(summary="Normal Meeting")]
    result = apply_filters(events, fc, True)
    assert len(result) == 1
    assert result[0].summary == "Normal Meeting"


def test_location_keyword_exclusion():
    fc = FilterConfig(location_keywords=["room 999"])
    events = [_event(location="Room 999"), _event(location="Room 101")]
    result = apply_filters(events, fc, True)
    assert len(result) == 1
    assert result[0].location == "Room 101"


def test_category_include_only():
    fc = FilterConfig(category_mode="include_only", category_list=["Work"])
    events = [_event(categories=["Work"]), _event(categories=["Personal"])]
    result = apply_filters(events, fc, True)
    assert len(result) == 1
    assert "Work" in result[0].categories


def test_category_exclude():
    fc = FilterConfig(category_mode="exclude", category_list=["Personal"])
    events = [_event(categories=["Work"]), _event(categories=["Personal"])]
    result = apply_filters(events, fc, True)
    assert len(result) == 1
    assert "Work" in result[0].categories
