"""Ch26: Delete candidate selection with DEL-01~05 rules."""

from datetime import datetime

TOOL_MARKER = "outlook_google_sync_v1"


def select_delete_candidates(
    existing_by_key: dict[str, dict],
    current_keys,
    range_start: datetime | None = None,
    range_end: datetime | None = None,
) -> list[dict]:
    """Return Google events that are deletion candidates (Ch26.2 DEL-01~05).

    DEL-01: tool_marker must match.
    DEL-02: event must overlap with R (range_start, range_end). If range not given, skip check.
    DEL-03: sync_key not in current_keys.
    DEL-04: R overlap checked before key diff.
    DEL-05: events moved outside R are not candidates.
    """
    current = set(current_keys)
    out: list[dict] = []
    for key, item in existing_by_key.items():
        private = ((item.get("extendedProperties") or {}).get("private") or {})

        # DEL-01
        if private.get("tool_marker") != TOOL_MARKER:
            continue

        # DEL-02: must overlap with R
        if range_start and range_end:
            if not _overlaps_range(item, range_start, range_end):
                continue

        # DEL-03
        if key not in current:
            out.append(item)
    return out


def _overlaps_range(item: dict, range_start: datetime, range_end: datetime) -> bool:
    """Check if event overlaps with [range_start, range_end]."""
    ev_start = _parse_event_dt(item.get("start", {}))
    ev_end = _parse_event_dt(item.get("end", {}))
    if ev_start is None or ev_end is None:
        return True  # conservative: include if we can't determine
    return ev_end > range_start and ev_start < range_end


def _parse_event_dt(val: dict) -> datetime | None:
    raw = val.get("dateTime") or val.get("date")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except (ValueError, TypeError):
        return None
