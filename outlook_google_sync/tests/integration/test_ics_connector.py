"""Integration test: ICS connector reads fixture file."""

from datetime import datetime
from pathlib import Path

from outlook_google_sync.connectors.outlook_ics import read_events_from_ics


def test_ics_connector_reads_fixture():
    fixture = Path(__file__).parents[1] / "fixtures" / "sample.ics"
    events = read_events_from_ics(str(fixture), input_method="ics_manual")
    assert len(events) >= 1
    e = events[0]
    assert e.summary == "Sample Event"
    assert e.reader_engine == "ics"
    assert e.input_method == "ics_manual"
    assert e.sync_key_kind == "primary"  # has UID


def test_ics_connector_with_date_range():
    fixture = Path(__file__).parents[1] / "fixtures" / "sample.ics"
    events = read_events_from_ics(
        str(fixture), input_method="ics_manual",
        date_start=datetime(2026, 3, 1),
        date_end=datetime(2026, 4, 1),
    )
    assert len(events) >= 1


def test_ics_connector_outside_range():
    fixture = Path(__file__).parents[1] / "fixtures" / "sample.ics"
    events = read_events_from_ics(
        str(fixture), input_method="ics_manual",
        date_start=datetime(2025, 1, 1),
        date_end=datetime(2025, 2, 1),
    )
    assert len(events) == 0
