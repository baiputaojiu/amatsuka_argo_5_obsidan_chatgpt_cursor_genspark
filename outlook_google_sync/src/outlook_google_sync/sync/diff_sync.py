"""Ch13.5: Differential sync target selection.

Picks events that actually need to be written — either because the source
fingerprint changed, the Google-side event disappeared, or Google was
hand-edited after our last write (DIFF-GC-01~03).

Phase 2: ``extendedProperties.private`` access routed through
:class:`GoogleEventView`.
"""

from __future__ import annotations

from ..models.google_event import GoogleEventView

__all__ = ["filter_diff_targets"]


def filter_diff_targets(events, previous_fingerprints: dict, google_existing: dict):
    targets = []
    for e in events:
        prev = previous_fingerprints.get(e.sync_key)
        if prev != e.fingerprint:
            targets.append(e)
            continue
        item = google_existing.get(e.sync_key)
        if not item:
            targets.append(e)
            continue
        view = GoogleEventView(item)
        last_write = view.last_tool_write_utc
        updated = view.updated
        if not last_write or not updated or updated > last_write:
            targets.append(e)
    return targets
