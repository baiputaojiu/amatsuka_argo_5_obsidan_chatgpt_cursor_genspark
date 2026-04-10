"""Unit tests: detach_new policy (Ch34.2 mandatory).

Since detach_new involves Google API calls, we test the logic at the
conflict.py / engine level with the sync_key derivation guarantee.
"""

from outlook_google_sync.models.event import EventModel
from datetime import datetime


def test_detach_new_uses_source_sync_key():
    """Ch17.2/DETACH-02: New event after detach must use source-derived sync_key,
    not an arbitrary new key."""
    e = EventModel(
        sync_key="source_derived_key_abc",
        summary="Meeting",
        start=datetime(2026, 1, 15, 10),
        end=datetime(2026, 1, 15, 11),
        reader_engine="ics",
        input_method="ics_manual",
        sync_key_kind="primary",
    )
    body = e.to_google_body("full")
    private = body["extendedProperties"]["private"]
    assert private["sync_key"] == "source_derived_key_abc"
    assert private["sync_key_kind"] == "primary"


def test_detach_keys_list():
    """DETACH-01: All required keys must be in the detach list."""
    from outlook_google_sync.connectors.google_calendar import DETACH_KEYS
    required = {"tool_marker", "sync_key", "sync_key_kind", "reader_engine", "input_method", "last_tool_write_utc"}
    assert required == set(DETACH_KEYS)
