"""Read-only value object wrapping a Google Calendar API event dict.

Phase 2 refactor goal: replace the repetitive
``((item.get("extendedProperties") or {}).get("private") or {})`` pattern
(which appeared ~10 times across ``sync/`` and ``gui/``) with a single
ergonomic accessor.

Design constraints
------------------
- **Zero-copy, non-owning.** The view stores a reference to the underlying
  ``dict`` (actually any ``Mapping``). Construction is O(1).
- **Read-only.** All attributes are properties; there are no setters.
  Mutation still happens on the raw dict (via the API layer or
  ``upsert_event``). This is intentional: Google API requests want bodies
  in the native ``dict`` shape.
- **Backward compatible with dict-shaped callers.** Public functions in
  ``sync/`` still accept plain ``dict`` arguments; they call
  ``GoogleEventView(item)`` internally. This keeps existing tests,
  connector outputs, and GUI code working unchanged.
- **Null-safe.** Every accessor degrades gracefully to an empty value.

See :class:`GoogleEventView` for the available accessors.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime
from typing import Any

from ..constants import TOOL_MARKER

__all__ = ["GoogleEventView", "parse_event_datetime", "to_local_naive"]


def parse_event_datetime(dt_field: Mapping[str, Any] | None) -> datetime | None:
    """Parse a Google event ``start`` / ``end`` object into a ``datetime``.

    Accepts both ``{"dateTime": "..."}`` (timed events, possibly with
    offset or trailing ``Z``) and ``{"date": "YYYY-MM-DD"}`` (all-day).
    Returns ``None`` when the field is missing or unparseable.

    Note: the ``Z`` suffix is normalised to ``+00:00`` for
    ``datetime.fromisoformat`` compatibility across Python 3.10 (which
    cannot parse ``Z`` natively) and 3.11+.
    """
    if not dt_field:
        return None
    raw = dt_field.get("dateTime") or dt_field.get("date")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def to_local_naive(dt: datetime) -> datetime:
    """Convert an aware ``datetime`` to a naive local-timezone one.

    Naive inputs are returned unchanged. Used at GUI boundaries where
    ``tkcalendar`` and user-entered ranges are always naive-local.
    """
    if dt.tzinfo is None:
        return dt
    return dt.astimezone().replace(tzinfo=None)


class GoogleEventView:
    """Ergonomic read-only view over a Google Calendar API event.

    Example::

        view = GoogleEventView(google_item)
        if view.is_managed and view.sync_key == target_key:
            ...
    """

    __slots__ = ("_raw",)

    def __init__(self, raw: Mapping[str, Any]) -> None:
        self._raw = raw

    # ── Core raw access ───────────────────────────────────────────────

    @property
    def raw(self) -> Mapping[str, Any]:
        """The underlying dict. Use when calling Google API (which wants
        native dicts) or when mutation is required."""
        return self._raw

    def get(self, key: str, default: Any = None) -> Any:
        """Pass-through ``dict.get`` for convenience during migration."""
        return self._raw.get(key, default)

    # ── Identity / top-level fields ───────────────────────────────────

    @property
    def id(self) -> str:
        return str(self._raw.get("id") or "")

    @property
    def summary(self) -> str:
        return str(self._raw.get("summary") or "")

    @property
    def description(self) -> str:
        return str(self._raw.get("description") or "")

    @property
    def location(self) -> str:
        return str(self._raw.get("location") or "")

    @property
    def visibility(self) -> str:
        return str(self._raw.get("visibility") or "")

    @property
    def color_id(self) -> str:
        return str(self._raw.get("colorId") or "")

    @property
    def updated(self) -> str:
        """Google-side last-update timestamp (RFC3339 string, or empty)."""
        return str(self._raw.get("updated") or "")

    @property
    def attendees(self) -> list[dict]:
        """Attendee list as raw dicts (empty if absent)."""
        at = self._raw.get("attendees")
        return list(at) if at else []

    # ── extendedProperties.private accessors ──────────────────────────

    @property
    def private_props(self) -> Mapping[str, Any]:
        """``extendedProperties.private`` map; empty dict if absent.

        This replaces the noisy
        ``((item.get("extendedProperties") or {}).get("private") or {})``
        pattern.
        """
        ext = self._raw.get("extendedProperties") or {}
        return ext.get("private") or {}

    @property
    def tool_marker(self) -> str:
        return str(self.private_props.get("tool_marker") or "")

    @property
    def sync_key(self) -> str:
        return str(self.private_props.get("sync_key") or "")

    @property
    def sync_key_kind(self) -> str:
        return str(self.private_props.get("sync_key_kind") or "")

    @property
    def reader_engine(self) -> str:
        return str(self.private_props.get("reader_engine") or "")

    @property
    def input_method(self) -> str:
        return str(self.private_props.get("input_method") or "")

    @property
    def last_tool_write_utc(self) -> str:
        return str(self.private_props.get("last_tool_write_utc") or "")

    @property
    def is_managed(self) -> bool:
        """True iff this event carries our tool's marker (Ch17 condition 1,
        Ch26 DEL-01)."""
        return self.tool_marker == TOOL_MARKER

    # ── Time fields ───────────────────────────────────────────────────

    @property
    def start_raw(self) -> Mapping[str, Any]:
        """The ``start`` sub-dict as-is (possibly empty)."""
        return self._raw.get("start") or {}

    @property
    def end_raw(self) -> Mapping[str, Any]:
        """The ``end`` sub-dict as-is (possibly empty)."""
        return self._raw.get("end") or {}

    @property
    def start_value(self) -> str:
        """String form of ``start`` — ``dateTime`` if present, else ``date``,
        else empty. Convenient for sort keys and hashing."""
        st = self.start_raw
        return str(st.get("dateTime") or st.get("date") or "")

    @property
    def end_value(self) -> str:
        en = self.end_raw
        return str(en.get("dateTime") or en.get("date") or "")

    @property
    def is_all_day(self) -> bool:
        return bool(self.start_raw.get("date")) and not self.start_raw.get("dateTime")

    @property
    def start_dt(self) -> datetime | None:
        """Parsed ``start`` datetime, or ``None`` if absent/invalid."""
        return parse_event_datetime(self.start_raw)

    @property
    def end_dt(self) -> datetime | None:
        """Parsed ``end`` datetime, or ``None`` if absent/invalid."""
        return parse_event_datetime(self.end_raw)

    def start_local_date(self) -> date | None:
        """Calendar date of ``start`` in the local timezone.

        Used by duplicate-repair range filtering (Ch27) — we care about
        which user-visible calendar day the event lives on, regardless of
        the stored representation (date / naive dateTime / aware dateTime).
        """
        st = self.start_raw
        if st.get("date"):
            try:
                return date.fromisoformat(str(st["date"]))
            except (ValueError, TypeError):
                return None
        dt = self.start_dt
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.date()
        return dt.astimezone().date()

    def overlaps(self, range_start: datetime, range_end: datetime) -> bool:
        """True if the event intersects ``[range_start, range_end]``.

        ``range_start`` / ``range_end`` are treated as naive-local (matching
        GUI inputs). Aware datetimes on the event side are converted via
        :func:`to_local_naive`. If start/end cannot be determined, returns
        ``True`` (conservative — included rather than silently dropped).
        """
        s, e = self.start_dt, self.end_dt
        if s is None or e is None:
            return True
        s_cmp = to_local_naive(s)
        e_cmp = to_local_naive(e)
        return e_cmp > range_start and s_cmp < range_end

    # ── Representation ────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"GoogleEventView(id={self.id!r}, summary={self.summary!r}, "
            f"sync_key={self.sync_key!r}, managed={self.is_managed})"
        )

    # ── Class-level helpers ───────────────────────────────────────────

    @classmethod
    def wrap_many(cls, items: Iterable[Mapping[str, Any]]) -> list[GoogleEventView]:
        """Wrap an iterable of raw event dicts. Convenience for GUI/test
        code that lists results."""
        return [cls(i) for i in items]
