"""Unit tests: datetime_utils.default_sync_range."""

from datetime import date

from outlook_google_sync.utils.datetime_utils import default_sync_range


def test_default_sync_range_fixed_day():
    start, end = default_sync_range(date(2026, 7, 15))
    assert start == date(2026, 6, 15)
    assert end == date(2026, 9, 15)


def test_default_sync_range_month_end():
    start, end = default_sync_range(date(2026, 3, 31))
    assert start == date(2026, 2, 28)
    assert end == date(2026, 5, 31)
