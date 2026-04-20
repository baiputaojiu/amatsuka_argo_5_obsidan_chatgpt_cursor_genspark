"""Google Calendar API connector."""

from __future__ import annotations
import logging
import threading
from datetime import UTC, datetime

from google.auth.exceptions import RefreshError
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

_SERVICE_LOCAL = threading.local()


def _run_oauth_flow() -> Credentials:
    """ブラウザでの新規認可フローを実行して Credentials を返す。"""
    flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path()), SCOPES)
    return flow.run_local_server(port=0)


def get_service():
    cached = getattr(_SERVICE_LOCAL, "service", None)
    if cached is not None:
        return cached

    creds = None
    tp = token_path()
    if tp.exists():
        creds = Credentials.from_authorized_user_file(str(tp), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError as exc:
                # refresh_token が失効/取消（OAuth同意画面がTesting時の7日ルール、
                # ユーザーによる権限取消、長期未使用、パスワード変更等）。
                # 壊れたトークンを破棄し、新規認可フローに切り替える。
                logger.warning(
                    "Google OAuth refresh_token が無効です（%s）。token.json を破棄して再認可を行います。",
                    exc,
                )
                try:
                    tp.unlink(missing_ok=True)
                except OSError as unlink_exc:
                    logger.warning("token.json の削除に失敗: %s", unlink_exc)
                creds = _run_oauth_flow()
        else:
            creds = _run_oauth_flow()
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text(creds.to_json(), encoding="utf-8")
    service = build("calendar", "v3", credentials=creds)
    _SERVICE_LOCAL.service = service
    return service


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


def list_managed_event_items(calendar_id: str, time_min: datetime, time_max: datetime) -> list[dict]:
    """Return all tool-managed events in the range (multiple rows may share one sync_key)."""
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
    out: list[dict] = []
    for item in items:
        private = ((item.get("extendedProperties") or {}).get("private") or {})
        if private.get("tool_marker") != TOOL_MARKER:
            continue
        if private.get("sync_key"):
            out.append(item)
    return out


def get_event(calendar_id: str, event_id: str) -> dict:
    """Full event resource (for preview: description, attendees, etc.)."""
    service = get_service()
    return service.events().get(calendarId=calendar_id, eventId=event_id).execute()


def patch_event_merge(calendar_id: str, event_id: str, body: dict) -> dict:
    """Patch event without injecting TOOL_MARKER (duplicate merge for unmanaged-only groups)."""
    service = get_service()
    return service.events().patch(calendarId=calendar_id, eventId=event_id, body=body).execute()


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


# update() フォールバック時、本体から除外する読み取り専用/派生フィールド。
# これらを送信するとサーバー側で弾かれたり意図しない動作になり得るため削除する。
_UPDATE_STRIP_KEYS = frozenset({
    "etag",
    "kind",
    "id",
    "iCalUID",
    "created",
    "updated",
    "htmlLink",
    "creator",
    "organizer",
    "hangoutLink",
    "conferenceData",
})


def _start_end_kind(obj: object) -> str | None:
    """start/end オブジェクトが date 型か dateTime 型かを返す（判定不能は None）。"""
    if not isinstance(obj, dict):
        return None
    if obj.get("dateTime"):
        return "dateTime"
    if obj.get("date"):
        return "date"
    return None


def _needs_update_fallback(existing_event: dict, body: dict) -> bool:
    """既存イベントと送信ボディで start/end の型（date↔dateTime）が食い違うなら True。

    Google Calendar API の patch 意味論では、start/end のネストオブジェクトを
    送っても反対側のフィールド（dateTime と date）が残り続けてしまい、
    "Invalid start time." 400 エラーが発生する。型が食い違うときだけ
    events().update() にフォールバックして完全置換する。
    """
    for key in ("start", "end"):
        e_kind = _start_end_kind(existing_event.get(key))
        b_kind = _start_end_kind(body.get(key))
        if e_kind and b_kind and e_kind != b_kind:
            return True
    return False


def _build_update_body(existing_event: dict, body: dict) -> dict:
    """update() 用に、既存リソースへ送信ボディを重ねたフルボディを生成する。"""
    merged: dict = {
        k: v for k, v in existing_event.items() if k not in _UPDATE_STRIP_KEYS
    }
    for key, val in body.items():
        if key in ("start", "end"):
            # 反対側のフィールドが残存しないよう完全置換する
            merged[key] = dict(val) if isinstance(val, dict) else val
        elif key == "extendedProperties" and isinstance(val, dict):
            existing_ext = merged.get("extendedProperties") or {}
            existing_priv = dict(existing_ext.get("private") or {})
            existing_priv.update(val.get("private") or {})
            new_ext: dict = {"private": existing_priv}
            shared = existing_ext.get("shared")
            if shared:
                new_ext["shared"] = dict(shared)
            merged["extendedProperties"] = new_ext
        else:
            merged[key] = val
    return merged


def upsert_event(
    calendar_id: str,
    event_id: str | None,
    body: dict,
    existing_event: dict | None = None,
) -> tuple[str, str]:
    """Insert or patch a single event. Returns (action, event_id).

    Stamps last_tool_write_utc in the primary write body.

    existing_event を渡すと、start/end の型（date/dateTime）が食い違う場合に
    patch ではなく update で完全置換してフォールバックする。
    """
    service = get_service()
    private = body.setdefault("extendedProperties", {}).setdefault("private", {})
    private["tool_marker"] = TOOL_MARKER
    private["last_tool_write_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    if event_id:
        if existing_event and _needs_update_fallback(existing_event, body):
            update_body = _build_update_body(existing_event, body)
            logger.info(
                "start/end 型不一致のため update フォールバック: eventId=%s",
                event_id,
            )
            result = service.events().update(
                calendarId=calendar_id, eventId=event_id, body=update_body,
            ).execute()
        else:
            result = service.events().patch(
                calendarId=calendar_id, eventId=event_id, body=body,
            ).execute()
        return "updated", result["id"]
    else:
        result = service.events().insert(
            calendarId=calendar_id, body=body,
        ).execute()
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
            google_item = existing.get(e.sync_key) or {}
            eid = google_item.get("id")
            action, _ = upsert_event(calendar_id, eid, body, existing_event=google_item or None)
            if action == "created":
                created += 1
            else:
                updated += 1
        except Exception as exc:
            errors.append(str(exc))
    return created, updated, errors
