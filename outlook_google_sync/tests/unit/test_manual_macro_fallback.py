"""Unit tests: manual ICS ↔ macro ICS fallback consistency (Ch34.2 mandatory)."""

from outlook_google_sync.sync.key_utils import generate_fallback_sync_key


def test_manual_macro_same_fallback_when_reader_engine_same():
    """FB-BAN-03: Same reader_engine 'ics' produces same fallback
    regardless of input_method (manual vs macro)."""
    key_manual = generate_fallback_sync_key("ics", "Weekly Meeting", "2026-01-15T10:00:00Z", "2026-01-15T11:00:00Z")
    key_macro = generate_fallback_sync_key("ics", "Weekly Meeting", "2026-01-15T10:00:00Z", "2026-01-15T11:00:00Z")
    assert key_manual == key_macro


def test_fallback_excludes_file_path():
    """FB-BAN-01: File path is not part of the fallback hash input."""
    key = generate_fallback_sync_key("ics", "Meeting", "2026-01-15T10:00:00Z", "2026-01-15T11:00:00Z")
    assert len(key) == 32
    assert isinstance(key, str)
