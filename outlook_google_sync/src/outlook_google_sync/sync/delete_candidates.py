"""Ch26: Delete candidate selection with DEL-01~05 rules.

Phase 2: the implementation now uses :class:`GoogleEventView` internally
for ``extendedProperties.private`` access and time-range overlap, keeping
the public ``select_delete_candidates`` signature unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from ..constants import TOOL_MARKER
from ..models.google_event import GoogleEventView

__all__ = ["TOOL_MARKER", "select_delete_candidates"]


def select_delete_candidates(
    existing_by_key: dict[str, dict],
    current_keys: Iterable[str],
    range_start: datetime | None = None,
    range_end: datetime | None = None,
) -> list[dict]:
    """Return Google events that are deletion candidates (Ch26.2 DEL-01~05).

    DEL-01: ``tool_marker`` must match.
    DEL-02: event must overlap with R (``range_start``, ``range_end``).
            If range not given, the overlap check is skipped.
    DEL-03: ``sync_key`` not in ``current_keys``.
    DEL-04: R overlap is checked before key diff.
    DEL-05: events moved outside R are not candidates.
    """
    current = set(current_keys)
    out: list[dict] = []
    for key, item in existing_by_key.items():
        view = GoogleEventView(item)

        # DEL-01
        if not view.is_managed:
            continue

        # DEL-02 / DEL-05: must overlap with R if provided
        if range_start is not None and range_end is not None:
            if not view.overlaps(range_start, range_end):
                continue

        # DEL-03
        if key not in current:
            out.append(item)
    return out
