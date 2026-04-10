"""Outlook COM connector — reads events via pywin32 (Classic Outlook only)."""

from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from ..models.event import EventModel
from ..utils.hash_utils import sha256_text
from ..utils.color_mapping import outlook_color_to_google_color_id

logger = logging.getLogger("outlook_google_sync")

# Outlook BusyStatus constants
_OL_FREE = 0
_OL_TENTATIVE = 1
_OL_BUSY = 2


def read_events_from_outlook(
    start_dt: datetime,
    end_dt: datetime,
    include_private: bool = True,
) -> list[EventModel]:
    try:
        import win32com.client  # type: ignore
    except Exception:
        logger.warning("pywin32 not available; COM mode disabled")
        return []

    app = win32com.client.Dispatch("Outlook.Application")
    ns = app.GetNamespace("MAPI")
    folder = ns.GetDefaultFolder(9)  # olFolderCalendar
    items = folder.Items
    items.Sort("[Start]")
    items.IncludeRecurrences = True

    restrict_fmt = "%m/%d/%Y %H:%M %p"
    restriction = (
        f'[Start] < "{end_dt.strftime(restrict_fmt)}" '
        f'AND [End] > "{start_dt.strftime(restrict_fmt)}"'
    )
    try:
        restricted = items.Restrict(restriction)
    except Exception:
        logger.warning("Restrict failed; falling back to full iteration")
        restricted = items

    events: list[EventModel] = []
    for item in restricted:
        try:
            _process_item(item, start_dt, end_dt, include_private, events)
        except Exception as exc:
            logger.debug("Skip item due to: %s", exc)
    return events


def _process_item(
    item, start_dt: datetime, end_dt: datetime,
    include_private: bool, events: list[EventModel],
) -> None:
    start, end = _item_start_end_local_naive(item)
    if start is None or end is None:
        return

    # Period overlap check (Ch10.3) — GUI 範囲は naive ローカル想定。COM が返す aware 日時との
    # 比較で TypeError になり全件スキップしていたケースを防ぐ。
    if end <= start_dt or start >= end_dt:
        return

    sensitivity = getattr(item, "Sensitivity", 0)
    is_private = sensitivity == 2
    if is_private and not include_private:
        return

    summary = getattr(item, "Subject", "") or "(no subject)"
    entry_id = getattr(item, "EntryID", "") or ""
    busy_status = getattr(item, "BusyStatus", _OL_BUSY)

    is_all_day = bool(getattr(item, "AllDayEvent", False))

    # Category color (Ch19 — color-based, not name-based)
    categories_str = getattr(item, "Categories", "") or ""
    categories = [c.strip() for c in categories_str.split(",") if c.strip()]
    ol_color = _get_first_category_color(item)
    color_id = outlook_color_to_google_color_id(ol_color)

    # Ch11: sync_key — primary uses EntryID
    is_recurring = bool(getattr(item, "IsRecurring", False))
    if entry_id and not is_recurring:
        sync_key = sha256_text(entry_id)[:32]
        sync_key_kind = "primary"
    else:
        # Ch11.3 COM-REC-03: recurring occurrences default to fallback
        start_iso = start.isoformat()
        end_iso = end.isoformat()
        sync_key = sha256_text(f"outlook_com|{summary}|{start_iso}|{end_iso}")[:32]
        sync_key_kind = "fallback"

    last_mod = str(getattr(item, "LastModificationTime", ""))
    fingerprint = sha256_text(f"{summary}|{start}|{end}|{last_mod}")

    events.append(EventModel(
        sync_key=sync_key,
        summary=summary,
        start=start,
        end=end,
        description=getattr(item, "Body", "") or "",
        location=getattr(item, "Location", "") or "",
        is_private=is_private,
        is_all_day=is_all_day,
        source_id=entry_id,
        reader_engine="outlook_com",
        input_method="com",
        fingerprint=fingerprint,
        sync_key_kind=sync_key_kind,
        color_id=color_id,
        ol_category_color=ol_color,
        busy_status=busy_status,
        categories=categories,
    ))


def _to_datetime(val) -> Optional[datetime]:
    if isinstance(val, datetime):
        return val
    try:
        return datetime(val.year, val.month, val.day, val.hour, val.minute, val.second)
    except Exception:
        return None


def _utc_prop_to_local_naive(val: Any) -> Optional[datetime]:
    """StartUTC / EndUTC は UTC。ローカル naive に揃える（Teams/Exchange の時刻と画面を揃える）。"""
    dt = _to_datetime(val)
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.astimezone().replace(tzinfo=None)


def _item_start_end_local_naive(item: Any) -> tuple[Optional[datetime], Optional[datetime]]:
    """終日は Start/End。時刻付きは StartUTC/EndUTC を優先（Teams/Exchange で画面時刻と一致させる）。"""
    is_all_day = bool(getattr(item, "AllDayEvent", False))
    if is_all_day:
        sv = getattr(item, "Start", None)
        ev = getattr(item, "End", None)
        if not sv or not ev:
            return None, None
        return (
            _normalize_outlook_datetime(sv),
            _normalize_outlook_datetime(ev),
        )

    s_utc = getattr(item, "StartUTC", None)
    e_utc = getattr(item, "EndUTC", None)
    if s_utc is not None and e_utc is not None:
        s = _utc_prop_to_local_naive(s_utc)
        e = _utc_prop_to_local_naive(e_utc)
        if s is not None and e is not None:
            return s, e

    sv = getattr(item, "Start", None)
    ev = getattr(item, "End", None)
    if not sv or not ev:
        return None, None
    return (
        _normalize_outlook_datetime(sv),
        _normalize_outlook_datetime(ev),
    )


def _normalize_outlook_datetime(val: Any) -> Optional[datetime]:
    """COM の日時を naive ローカルに揃え、期間比較・EventModel 用に使う。"""
    dt = _to_datetime(val)
    if dt is None:
        return None
    if dt.tzinfo is not None:
        try:
            return dt.astimezone().replace(tzinfo=None)
        except Exception:
            return dt.replace(tzinfo=None)
    return dt


def _get_first_category_color(item) -> Optional[int]:
    """Get OlCategoryColor for the first category."""
    try:
        cats = getattr(item, "Categories", "") or ""
        if not cats:
            return None
        ns = item.Application.GetNamespace("MAPI")
        for cat in ns.Categories:
            if cat.Name == cats.split(",")[0].strip():
                return int(cat.Color)
    except Exception:
        pass
    return None
