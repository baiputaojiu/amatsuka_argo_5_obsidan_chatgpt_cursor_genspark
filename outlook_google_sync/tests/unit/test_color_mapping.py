"""Unit tests: category color mapping (Ch19)."""

from outlook_google_sync.utils.color_mapping import outlook_color_to_google_color_id


def test_known_colors():
    assert outlook_color_to_google_color_id(1) == "11"   # Red
    assert outlook_color_to_google_color_id(4) == "5"    # Yellow
    assert outlook_color_to_google_color_id(8) == "9"    # Blue
    assert outlook_color_to_google_color_id(9) == "3"    # Purple
    assert outlook_color_to_google_color_id(13) == "8"   # Gray


def test_none_returns_none():
    assert outlook_color_to_google_color_id(None) is None


def test_unknown_returns_none():
    assert outlook_color_to_google_color_id(99) is None
