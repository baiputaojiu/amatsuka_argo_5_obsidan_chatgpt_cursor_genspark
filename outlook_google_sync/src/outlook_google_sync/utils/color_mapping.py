"""Ch19: OlCategoryColor → Google colorId fixed mapping (color-based, not name-based)."""

OL_COLOR_TO_GOOGLE: dict[int, str] = {
    1: "11",   # Red
    2: "6",    # Orange
    3: "6",    # Peach
    4: "5",    # Yellow
    5: "10",   # Green
    6: "2",    # Teal
    7: "10",   # Olive
    8: "9",    # Blue
    9: "3",    # Purple
    10: "11",  # Maroon
    11: "7",   # Steel
    12: "7",   # DarkSteel
    13: "8",   # Gray
    14: "8",   # DarkGray
    15: "8",   # Black
    16: "11",  # DarkRed
    17: "6",   # DarkOrange
    18: "6",   # DarkPeach
    19: "5",   # DarkYellow
    20: "10",  # DarkGreen
    21: "2",   # DarkTeal
    22: "10",  # DarkOlive
    23: "9",   # DarkBlue
    24: "3",   # DarkPurple
    25: "11",  # DarkMaroon
}


def outlook_color_to_google_color_id(ol_category_color: int | None) -> str | None:
    """Return Google colorId for the given OlCategoryColor enum value, or None."""
    if ol_category_color is None:
        return None
    return OL_COLOR_TO_GOOGLE.get(ol_category_color)
