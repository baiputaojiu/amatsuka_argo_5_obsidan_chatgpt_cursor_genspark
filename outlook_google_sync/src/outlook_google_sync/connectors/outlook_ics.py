"""ICS connector — reads events from .ics files (manual or macro)."""

from __future__ import annotations
import logging
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Optional

from icalendar import Calendar

from ..models.event import EventModel
from ..utils.hash_utils import sha256_text

logger = logging.getLogger("outlook_google_sync")


def read_events_from_ics(
    ics_path: str,
    input_method: str = "ics_manual",
    date_start: Optional[datetime] = None,
    date_end: Optional[datetime] = None,
) -> list[EventModel]:
    path = Path(ics_path)
    if not path.exists():
        return []

    cal = Calendar.from_ical(path.read_bytes())
    events: list[EventModel] = []

    for comp in cal.walk("VEVENT"):
        try:
            _process_vevent(comp, input_method, date_start, date_end, events)
        except Exception as exc:
            logger.debug("Skip VEVENT: %s", exc)

    # Ch12: expand RRULE occurrences within date range
    _expand_rrule_events(cal, input_method, date_start, date_end, events)

    return events


def _process_vevent(
    comp, input_method: str,
    date_start: Optional[datetime], date_end: Optional[datetime],
    events: list[EventModel],
) -> None:
    ds = comp.get("dtstart")
    de = comp.get("dtend")
    if not ds:
        return

    start_raw = ds.dt
    is_all_day = isinstance(start_raw, date) and not isinstance(start_raw, datetime)

    if is_all_day:
        start = datetime.combine(start_raw, time.min)
        if de:
            end = datetime.combine(de.dt, time.min)
        else:
            dur = comp.get("duration")
            end = start + dur.dt if dur else start + timedelta(days=1)
    else:
        start = start_raw if isinstance(start_raw, datetime) else datetime.combine(start_raw, time.min)
        if de:
            end_raw = de.dt
            end = end_raw if isinstance(end_raw, datetime) else datetime.combine(end_raw, time.min)
        else:
            dur = comp.get("duration")
            end = start + dur.dt if dur else start + timedelta(hours=1)

    # Period filter (normalize tz-awareness for comparison)
    if date_start and date_end:
        s_cmp = start.replace(tzinfo=None) if start.tzinfo else start
        e_cmp = end.replace(tzinfo=None) if end.tzinfo else end
        ds_cmp = date_start.replace(tzinfo=None) if date_start.tzinfo else date_start
        de_cmp = date_end.replace(tzinfo=None) if date_end.tzinfo else date_end
        if e_cmp <= ds_cmp or s_cmp >= de_cmp:
            return

    # Skip if this VEVENT has RRULE (will be handled by _expand_rrule_events)
    if comp.get("rrule"):
        return

    uid = str(comp.get("uid", ""))
    rec_id = str(comp.get("recurrence-id", ""))
    source_id = uid + (f"::{rec_id}" if rec_id else "")

    summary = str(comp.get("summary", "(no subject)"))
    description = str(comp.get("description", ""))
    location = str(comp.get("location", ""))
    is_private = str(comp.get("class", "")).upper() == "PRIVATE"

    categories_prop = comp.get("categories")
    categories: list[str] = []
    if categories_prop:
        if hasattr(categories_prop, "cats"):
            categories = [str(c) for c in categories_prop.cats]
        elif isinstance(categories_prop, list):
            for cp in categories_prop:
                if hasattr(cp, "cats"):
                    categories.extend(str(c) for c in cp.cats)

    busy_status = _parse_transp(comp)

    if source_id and uid:
        sync_key = sha256_text(source_id)[:32]
        sync_key_kind = "primary"
    else:
        start_iso = start.isoformat()
        end_iso = end.isoformat()
        sync_key = sha256_text(f"ics|{summary}|{start_iso}|{end_iso}")[:32]
        sync_key_kind = "fallback"

    fingerprint = sha256_text(f"{summary}|{start}|{end}|{uid}|{rec_id}")

    events.append(EventModel(
        sync_key=sync_key,
        summary=summary,
        start=start,
        end=end,
        description=description,
        location=location,
        is_private=is_private,
        is_all_day=is_all_day,
        source_id=source_id,
        reader_engine="ics",
        input_method=input_method,
        fingerprint=fingerprint,
        sync_key_kind=sync_key_kind,
        busy_status=busy_status,
        categories=categories,
    ))


def _expand_rrule_events(
    cal: Calendar, input_method: str,
    date_start: Optional[datetime], date_end: Optional[datetime],
    events: list[EventModel],
) -> None:
    """Ch12: Expand recurring events (RRULE) into individual occurrences."""
    try:
        from dateutil.rrule import rrulestr  # type: ignore
    except ImportError:
        logger.warning("python-dateutil not available; RRULE expansion skipped")
        return

    for comp in cal.walk("VEVENT"):
        rrule = comp.get("rrule")
        if not rrule:
            continue
        ds = comp.get("dtstart")
        if not ds:
            continue

        start_raw = ds.dt
        is_all_day = isinstance(start_raw, date) and not isinstance(start_raw, datetime)
        if is_all_day:
            base_start = datetime.combine(start_raw, time.min)
        else:
            base_start = start_raw if isinstance(start_raw, datetime) else datetime.combine(start_raw, time.min)

        de = comp.get("dtend")
        if de:
            end_raw = de.dt
            if is_all_day:
                base_end = datetime.combine(end_raw, time.min)
            else:
                base_end = end_raw if isinstance(end_raw, datetime) else datetime.combine(end_raw, time.min)
        else:
            dur = comp.get("duration")
            base_end = base_start + (dur.dt if dur else timedelta(hours=1))

        duration = base_end - base_start
        uid = str(comp.get("uid", ""))
        summary = str(comp.get("summary", "(no subject)"))
        description = str(comp.get("description", ""))
        location = str(comp.get("location", ""))
        is_private = str(comp.get("class", "")).upper() == "PRIVATE"
        busy_status = _parse_transp(comp)

        rrule_str = rrule.to_ical().decode("utf-8")
        try:
            rule = rrulestr(f"RRULE:{rrule_str}", dtstart=base_start, ignoretz=True)
        except Exception as exc:
            logger.debug("RRULE parse error: %s", exc)
            continue

        limit = date_end or (base_start + timedelta(days=365))
        occurrences = list(rule.between(
            date_start or base_start,
            limit,
            inc=True,
        ))

        for occ_start in occurrences[:500]:
            occ_end = occ_start + duration
            rec_id_str = occ_start.isoformat()
            source_id = f"{uid}::{rec_id_str}" if uid else ""
            if source_id:
                sync_key = sha256_text(source_id)[:32]
                sync_key_kind = "primary"
            else:
                sync_key = sha256_text(f"ics|{summary}|{occ_start.isoformat()}|{occ_end.isoformat()}")[:32]
                sync_key_kind = "fallback"

            fingerprint = sha256_text(f"{summary}|{occ_start}|{occ_end}|{uid}|{rec_id_str}")
            events.append(EventModel(
                sync_key=sync_key,
                summary=summary,
                start=occ_start,
                end=occ_end,
                description=description,
                location=location,
                is_private=is_private,
                is_all_day=is_all_day,
                source_id=source_id,
                reader_engine="ics",
                input_method=input_method,
                fingerprint=fingerprint,
                sync_key_kind=sync_key_kind,
                busy_status=busy_status,
            ))


def _parse_transp(comp) -> int:
    """Parse TRANSP property to busy_status."""
    transp = str(comp.get("transp", "")).upper()
    if transp == "TRANSPARENT":
        return 0  # Free
    return 2  # Busy
