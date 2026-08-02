"""Central constants (single source of truth).

Phase 1 refactor: these values used to be duplicated or inlined across
multiple modules. Keep this module import-side-effect-free so it can be
imported from any layer (gui / sync / connectors / models / config).

Historical locations (all consolidated here):
- ``TOOL_MARKER``: previously defined in ``connectors/google_calendar.py``,
  ``sync/conflict.py``, ``sync/delete_candidates.py`` and hardcoded in
  ``models/event.py``.
- ``DETACH_KEYS``: previously in ``connectors/google_calendar.py``.
- ``SCHEMA_VERSION``: previously in ``models/config_schema.py`` (still
  re-exported there for backward compatibility).
- ``STALE_THRESHOLD_MINUTES``: previously inline in ``sync/preview.py``
  (still re-exported there for backward compatibility).
- ``GOOGLE_API_MAX_RESULTS_PER_PAGE``: previously the literal ``2500``
  in three list() calls in ``connectors/google_calendar.py``.
- ``GOOGLE_OAUTH_SCOPES``: previously ``SCOPES`` in
  ``connectors/google_calendar.py``.
- ``GOOGLE_CALENDAR_TZ_FALLBACK``: previously inline in
  ``connectors/google_calendar.py``.
"""

from __future__ import annotations


TOOL_MARKER: str = "outlook_google_sync_v1"
"""Identifier stamped onto every Google event we create.

Used by conflict detection, delete-candidate selection, and duplicate repair
to tell "our" events apart from Google-native events.
"""


DETACH_KEYS: tuple[str, ...] = (
    "tool_marker",
    "sync_key",
    "sync_key_kind",
    "reader_engine",
    "input_method",
    "last_tool_write_utc",
)
"""Keys removed from ``extendedProperties.private`` when detaching an event
from tool management (conflict policy ``detach_new``)."""


SCHEMA_VERSION: int = 1
"""Current config.json schema version. Bump when introducing breaking keys."""


STALE_THRESHOLD_MINUTES: int = 5
"""PRE-SNAP-03 (fixed): preview snapshots older than this are considered
stale and the user is asked to re-preview."""


GOOGLE_API_MAX_RESULTS_PER_PAGE: int = 2500
"""``events.list`` / ``calendarList.list`` page size. The Google API cap is
2500 per page; requests beyond that must follow ``nextPageToken`` (handled
by ``connectors/google_calendar._paginate_events_list`` since Phase 3)."""


GOOGLE_OAUTH_SCOPES: tuple[str, ...] = (
    "https://www.googleapis.com/auth/calendar",
)


GOOGLE_CALENDAR_TZ_FALLBACK: str = "UTC"
"""Fallback IANA time zone used when ``calendars.get`` fails to return one.
Intentionally environment-independent."""


# ── Phase 3: retry / backoff ────────────────────────────────────────

GOOGLE_API_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
"""HTTP statuses that warrant a retry for Google Calendar API calls.

- 429 Too Many Requests (quota / rate limit)
- 500 Internal Server Error, 502 Bad Gateway, 503 Service Unavailable,
  504 Gateway Timeout — transient Google-side failures
"""


GOOGLE_API_MAX_RETRY_ATTEMPTS: int = 5
"""Maximum total attempts (including the first) for a retryable call.

With the default base/max delay this yields roughly 1 + 2 + 4 + 8 + 16 ≈ 31
seconds of waiting in the worst case before giving up.
"""


GOOGLE_API_RETRY_BASE_DELAY_SECONDS: float = 1.0
"""Initial delay between retries. Doubles each attempt (capped at max)."""


GOOGLE_API_RETRY_MAX_DELAY_SECONDS: float = 30.0
"""Ceiling for retry delay, regardless of how many attempts have passed."""
