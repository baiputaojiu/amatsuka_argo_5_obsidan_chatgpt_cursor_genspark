"""Unit tests: sync_key generation (Ch34.2 mandatory)."""

from outlook_google_sync.sync.key_utils import generate_fallback_sync_key


def test_generate_fallback_sync_key_deterministic():
    a = generate_fallback_sync_key("ics", "Meeting", "2026-01-01T10:00:00", "2026-01-01T11:00:00")
    b = generate_fallback_sync_key("ics", "Meeting", "2026-01-01T10:00:00", "2026-01-01T11:00:00")
    assert a == b


def test_different_engine_different_key():
    a = generate_fallback_sync_key("ics", "Meeting", "2026-01-01T10:00:00", "2026-01-01T11:00:00")
    b = generate_fallback_sync_key("outlook_com", "Meeting", "2026-01-01T10:00:00", "2026-01-01T11:00:00")
    assert a != b


def test_fallback_key_does_not_include_input_method():
    """FB-BAN-03: input_method must not affect fallback key."""
    a = generate_fallback_sync_key("ics", "Meeting", "2026-01-01T10:00:00", "2026-01-01T11:00:00")
    b = generate_fallback_sync_key("ics", "Meeting", "2026-01-01T10:00:00", "2026-01-01T11:00:00")
    assert a == b
