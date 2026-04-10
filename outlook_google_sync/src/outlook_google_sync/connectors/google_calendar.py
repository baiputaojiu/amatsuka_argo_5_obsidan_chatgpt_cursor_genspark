"""Google Calendar API connector."""

from __future__ import annotations
import logging
from datetime import UTC, datetime

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from ..config.paths import credentials_path, token_path

logger = logging.getLogger("outlook_google_sync")

# API 失敗時のフォールバック（環境に依存しない IANA）
_CALENDAR_TZ_FALLBACK = "UTC"

SCOPES = ["https://www.googleapis.com/auth/calendar"]
TOOL_MARKER = "outlook_google_sync_v1"

DETACH_KEYS = [
    "tool_marker", "sync_key", "sync_key_kind",
    "reader_engine", "input_method", "last_tool_write_utc",
]


def get_service():
    creds = None
    tp = token_path()
    if tp.exists():
        creds = Credentials.from_authorized_user_file(str(tp), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path()), SCOPES)
            creds = flow.run_local_server(port=0)
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text(creds.to_json(), encoding="utf-8")
    return build("calendar", "v3", credentials=creds)


def list_calendars() -> list[dict]:
    return get_service().calendarList().list(maxResults=250).execute().get("items", [])


def get_calendar_time_zone(calendar_id: str) -> str:
    """同期先カレンダーの IANA タイムゾーン（events の dateTime + timeZone に使う）。"""
    try:
        service = get_service()
        cal = service.calendars().get(calendarId=calendar_id).execute()
        tz = cal.get("timeZone")
        if tz:
            return str(tz)
    except Exception as exc:
        logger.warning(
            "カレンダー timeZone 取得失敗 (%s): %s — %s を使用します",
            calendar_id,
            exc,
            _CALENDAR_TZ_FALLBACK,
        )
    return _CALENDAR_TZ_FALLBACK


def list_managed_events(calendar_id: str, time_min: datetime, time_max: datetime) -> dict[str, dict]:
    """Return {sync_key: google_event} for tool-managed events in the range."""
    service = get_service()
    items = (
        service.events()
        .list(
            calendarId=calendar_id,
            timeMin=time_min.isoformat() + "Z",
            timeMax=time_max.isoformat() + "Z",
            singleEvents=True,
            maxResults=2500,
        )
        .execute()
        .get("items", [])
    )
    by_key: dict[str, dict] = {}
    for item in items:
        private = ((item.get("extendedProperties") or {}).get("private") or {})
        if private.get("tool_marker") != TOOL_MARKER:
            continue
        key = private.get("sync_key")
        if key:
            by_key[key] = item
    return by_key


def list_all_events_in_range(calendar_id: str, time_min: datetime, time_max: datetime) -> list[dict]:
    """Return all events (including non-managed) in the range — used for merge candidate search."""
    service = get_service()
    return (
        service.events()
        .list(
            calendarId=calendar_id,
            timeMin=time_min.isoformat() + "Z",
            timeMax=time_max.isoformat() + "Z",
            singleEvents=True,
            maxResults=2500,
        )
        .execute()
        .get("items", [])
    )


def upsert_event(
    calendar_id: str,
    event_id: str | None,
    body: dict,
) -> tuple[str, str]:
    """Insert or patch a single event. Returns (action, event_id).

    Sets last_tool_write_utc after successful write (Ch11.2.1, Ch14.1 step 6).
    """
    service = get_service()
    private = body.setdefault("extendedProperties", {}).setdefault("private", {})
    private["tool_marker"] = TOOL_MARKER

    if event_id:
        result = service.events().patch(
            calendarId=calendar_id, eventId=event_id, body=body,
        ).execute()
        _stamp_last_write(calendar_id, result["id"], service)
        return "updated", result["id"]
    else:
        result = service.events().insert(
            calendarId=calendar_id, body=body,
        ).execute()
        _stamp_last_write(calendar_id, result["id"], service)
        return "created", result["id"]


def delete_event(calendar_id: str, event_id: str) -> None:
    """Delete a single event (Ch26.4)."""
    service = get_service()
    service.events().delete(calendarId=calendar_id, eventId=event_id).execute()


def detach_event(calendar_id: str, event_id: str) -> None:
    """Remove tool markers from event (Ch17.3 DETACH-01)."""
    service = get_service()
    existing = service.events().get(calendarId=calendar_id, eventId=event_id).execute()
    private = (existing.get("extendedProperties") or {}).get("private", {})
    for key in DETACH_KEYS:
        private.pop(key, None)
    body = {"extendedProperties": {"private": private if private else {}}}
    service.events().patch(calendarId=calendar_id, eventId=event_id, body=body).execute()


def _stamp_last_write(calendar_id: str, event_id: str, service) -> None:
    """Ch11.2.1: Write last_tool_write_utc immediately after successful write."""
    now_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    body = {"extendedProperties": {"private": {"last_tool_write_utc": now_utc}}}
    try:
        service.events().patch(calendarId=calendar_id, eventId=event_id, body=body).execute()
    except Exception as exc:
        logger.warning("Failed to stamp last_tool_write_utc: %s", exc)


# Legacy compat wrapper
def upsert_events(calendar_id: str, events, detail_level: str = "full"):
    """Batch upsert for backward compatibility."""
    if not events:
        return 0, 0, []
    existing = list_managed_events(
        calendar_id,
        min(e.start for e in events),
        max(e.end for e in events),
    )
    created = updated = 0
    errors: list[str] = []
    tz = get_calendar_time_zone(calendar_id)
    for e in events:
        body = e.to_google_body(detail_level, time_zone=tz)
        try:
            eid = existing.get(e.sync_key, {}).get("id")
            action, _ = upsert_event(calendar_id, eid, body)
            if action == "created":
                created += 1
            else:
                updated += 1
        except Exception as exc:
            errors.append(str(exc))
    return created, updated, errors
