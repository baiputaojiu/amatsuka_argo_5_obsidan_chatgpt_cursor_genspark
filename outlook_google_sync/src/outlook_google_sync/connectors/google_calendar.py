"""Google Calendar API connector.

Phase 3 hardening
-----------------
- **Pagination**: ``events().list()`` calls now follow ``nextPageToken``
  until the server stops returning one. Prior behaviour silently dropped
  events beyond the first 2500 in a range.
- **Retry with exponential backoff + jitter**: 429 / 5xx responses from
  Google are retried up to :data:`GOOGLE_API_MAX_RETRY_ATTEMPTS` times with
  delays doubling from :data:`GOOGLE_API_RETRY_BASE_DELAY_SECONDS` and
  capped at :data:`GOOGLE_API_RETRY_MAX_DELAY_SECONDS`. ``Retry-After``
  headers on 429 responses are honoured when present.
- **Init lock for ``get_service``**: the token-file load/refresh/save path
  is serialised via a module-level lock so multiple cold-start threads
  cannot race OAuth flows or corrupt ``token.json``.

Write-path contract (kept unchanged for test compatibility)
-----------------------------------------------------------
``upsert_event`` still stamps ``tool_marker`` and ``last_tool_write_utc``
directly into the incoming ``body`` dict. Read accessors use
:class:`GoogleEventView`.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from ..config.paths import credentials_path, token_path
from ..constants import (
    DETACH_KEYS,
    GOOGLE_API_MAX_RESULTS_PER_PAGE,
    GOOGLE_API_MAX_RETRY_ATTEMPTS,
    GOOGLE_API_RETRY_BASE_DELAY_SECONDS,
    GOOGLE_API_RETRY_MAX_DELAY_SECONDS,
    GOOGLE_API_RETRYABLE_STATUSES,
    GOOGLE_CALENDAR_TZ_FALLBACK,
    GOOGLE_OAUTH_SCOPES,
    TOOL_MARKER,
)
from ..models.google_event import GoogleEventView

logger = logging.getLogger("outlook_google_sync")

# Backward-compatible aliases — existing call sites and tests import these
# names from this module. Do not remove without auditing downstream usage.
_CALENDAR_TZ_FALLBACK = GOOGLE_CALENDAR_TZ_FALLBACK
SCOPES = list(GOOGLE_OAUTH_SCOPES)

__all__ = [
    "TOOL_MARKER",
    "DETACH_KEYS",
    "SCOPES",
    "get_service",
    "list_calendars",
    "get_calendar_time_zone",
    "list_managed_events",
    "list_managed_event_items",
    "list_all_events_in_range",
    "get_event",
    "patch_event_merge",
    "upsert_event",
    "upsert_events",
    "delete_event",
    "detach_event",
]

# ── Service cache / init lock ────────────────────────────────────────

_SERVICE_LOCAL = threading.local()
# Serialises the OAuth / token-file path in ``get_service``. The built
# service itself is kept per-thread (``_SERVICE_LOCAL``) because
# ``googleapiclient`` client objects are documented as non-thread-safe.
_CREDS_LOCK = threading.Lock()


def _run_oauth_flow() -> Credentials:
    """ブラウザでの新規認可フローを実行して Credentials を返す。"""
    flow = InstalledAppFlow.from_client_secrets_file(
        str(credentials_path()), list(GOOGLE_OAUTH_SCOPES),
    )
    return flow.run_local_server(port=0)


def _load_or_acquire_credentials() -> Credentials:
    """Load ``token.json`` if present, refresh if expired, or run OAuth.

    Must be called under :data:`_CREDS_LOCK` because it reads/writes
    ``token.json`` and may open a local web server.
    """
    creds: Credentials | None = None
    tp = token_path()
    if tp.exists():
        creds = Credentials.from_authorized_user_file(str(tp), list(GOOGLE_OAUTH_SCOPES))
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError as exc:
                # refresh_token が失効/取消（OAuth同意画面がTesting時の7日ルール、
                # ユーザーによる権限取消、長期未使用、パスワード変更等）。
                # 壊れたトークンを破棄し、新規認可フローに切り替える。
                logger.warning(
                    "Google OAuth refresh_token が無効です（%s）。"
                    "token.json を破棄して再認可を行います。",
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
    return creds


def get_service():
    """Return a thread-local ``calendar`` service, building it on first use.

    Credential acquisition (file I/O + OAuth) is serialised with a module
    lock; the built service is cached per thread since the googleapiclient
    client is not safe for concurrent use across threads.
    """
    cached = getattr(_SERVICE_LOCAL, "service", None)
    if cached is not None:
        return cached

    with _CREDS_LOCK:
        # Double-check: another thread may have cached in the meantime
        # (not possible under threading.local but cheap to verify).
        cached = getattr(_SERVICE_LOCAL, "service", None)
        if cached is not None:
            return cached
        creds = _load_or_acquire_credentials()
        service = build("calendar", "v3", credentials=creds)
        _SERVICE_LOCAL.service = service
        return service


# ── Retry helper ─────────────────────────────────────────────────────


def _http_status(exc: HttpError) -> int | None:
    """Extract the numeric HTTP status from ``HttpError``, if available."""
    resp = getattr(exc, "resp", None)
    status = getattr(resp, "status", None)
    if status is None:
        return None
    try:
        return int(status)
    except (TypeError, ValueError):
        return None


def _retry_after_seconds(exc: HttpError) -> float | None:
    """Parse a ``Retry-After`` header value (seconds only, RFC-date ignored).

    Google quota errors usually send an integer seconds value; we keep the
    parsing minimal and fall back to the computed backoff when anything is
    missing or malformed.
    """
    resp = getattr(exc, "resp", None)
    if resp is None:
        return None
    try:
        header_val = resp.get("retry-after") if hasattr(resp, "get") else None
    except Exception:
        return None
    if header_val is None:
        return None
    try:
        secs = float(str(header_val).strip())
    except (TypeError, ValueError):
        return None
    if secs < 0:
        return None
    return secs


def _compute_backoff(attempt: int) -> float:
    """Exponential backoff with full-jitter (AWS-style).

    ``attempt`` is zero-indexed among *failed* attempts. Jitter avoids the
    thundering-herd effect when many threads retry simultaneously.
    """
    cap = GOOGLE_API_RETRY_MAX_DELAY_SECONDS
    base = GOOGLE_API_RETRY_BASE_DELAY_SECONDS * (2 ** attempt)
    return random.uniform(0.0, min(cap, base))


def _sleep(seconds: float) -> None:
    """Indirection so tests can monkeypatch the wait."""
    time.sleep(seconds)


def _execute_with_retry(request: Any, *, label: str) -> Any:
    """Run ``request.execute()`` retrying on transient Google errors.

    Retryable: :data:`GOOGLE_API_RETRYABLE_STATUSES` (429 / 5xx).
    Other ``HttpError`` statuses and non-HTTP exceptions propagate
    immediately. ``Retry-After`` is honoured for 429.

    ``label`` is a short identifier used only for logging (e.g.
    ``"events.list"``). The function is intentionally tolerant of mocks —
    anything with an ``.execute()`` method is accepted.
    """
    attempt = 0
    last_exc: HttpError | None = None
    while True:
        try:
            return request.execute()
        except HttpError as exc:
            status = _http_status(exc)
            if status is None or status not in GOOGLE_API_RETRYABLE_STATUSES:
                raise
            last_exc = exc
            attempt += 1
            if attempt >= GOOGLE_API_MAX_RETRY_ATTEMPTS:
                logger.warning(
                    "%s: HTTP %s after %s attempts — giving up",
                    label, status, attempt,
                )
                raise
            delay = (
                _retry_after_seconds(exc)
                if status == 429
                else None
            )
            if delay is None:
                delay = _compute_backoff(attempt - 1)
            logger.warning(
                "%s: HTTP %s (attempt %s/%s), retrying in %.2fs",
                label, status, attempt, GOOGLE_API_MAX_RETRY_ATTEMPTS, delay,
            )
            _sleep(delay)
    # Unreachable — loop always returns or raises. Kept for mypy clarity.
    assert last_exc is not None
    raise last_exc


# ── Pagination helper ────────────────────────────────────────────────

_LIST_PAGE_SIZE = GOOGLE_API_MAX_RESULTS_PER_PAGE


def _paginate_events_list(
    calendar_id: str,
    time_min: datetime,
    time_max: datetime,
    *,
    label: str,
) -> Iterable[dict]:
    """Yield every event in the range, following ``nextPageToken``.

    Previously the three list-* helpers dropped silently at the 2500-item
    cap. With this generator pagination is fully drained. Each page is
    wrapped in :func:`_execute_with_retry` so transient failures don't
    abort the enumeration.
    """
    service = get_service()
    page_token: str | None = None
    while True:
        kwargs: dict[str, Any] = {
            "calendarId": calendar_id,
            "timeMin": time_min.isoformat() + "Z",
            "timeMax": time_max.isoformat() + "Z",
            "singleEvents": True,
            "maxResults": _LIST_PAGE_SIZE,
        }
        if page_token:
            kwargs["pageToken"] = page_token
        request = service.events().list(**kwargs)
        resp = _execute_with_retry(request, label=label) or {}
        for item in resp.get("items", []) or []:
            yield item
        page_token = resp.get("nextPageToken")
        if not page_token:
            return


# ── Public read API ──────────────────────────────────────────────────


def list_calendars() -> list[dict]:
    """List every writable calendar visible to the OAuth user.

    Follows ``nextPageToken`` in case the user has > 250 calendars.
    """
    service = get_service()
    out: list[dict] = []
    page_token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"maxResults": 250}
        if page_token:
            kwargs["pageToken"] = page_token
        resp = _execute_with_retry(
            service.calendarList().list(**kwargs),
            label="calendarList.list",
        ) or {}
        out.extend(resp.get("items", []) or [])
        page_token = resp.get("nextPageToken")
        if not page_token:
            return out


def get_calendar_time_zone(calendar_id: str) -> str:
    """同期先カレンダーの IANA タイムゾーン（events の dateTime + timeZone に使う）。"""
    try:
        service = get_service()
        cal = _execute_with_retry(
            service.calendars().get(calendarId=calendar_id),
            label="calendars.get",
        )
        tz = cal.get("timeZone") if isinstance(cal, dict) else None
        if tz:
            return str(tz)
    except Exception as exc:
        logger.warning(
            "カレンダー timeZone 取得失敗 (%s): %s — %s を使用します",
            calendar_id,
            exc,
            GOOGLE_CALENDAR_TZ_FALLBACK,
        )
    return GOOGLE_CALENDAR_TZ_FALLBACK


def list_managed_events(calendar_id: str, time_min: datetime, time_max: datetime) -> dict[str, dict]:
    """Return ``{sync_key: google_event}`` for tool-managed events in the range.

    Paginates the underlying ``events.list`` call so ranges exceeding 2500
    events are enumerated completely (Ch13 correctness fix).
    """
    by_key: dict[str, dict] = {}
    for item in _paginate_events_list(
        calendar_id, time_min, time_max, label="events.list(managed)"
    ):
        view = GoogleEventView(item)
        if not view.is_managed:
            continue
        if view.sync_key:
            by_key[view.sync_key] = item
    return by_key


def list_managed_event_items(calendar_id: str, time_min: datetime, time_max: datetime) -> list[dict]:
    """Return every tool-managed event in the range (same sync_key may repeat)."""
    out: list[dict] = []
    for item in _paginate_events_list(
        calendar_id, time_min, time_max, label="events.list(managed-items)"
    ):
        view = GoogleEventView(item)
        if not view.is_managed:
            continue
        if view.sync_key:
            out.append(item)
    return out


def list_all_events_in_range(calendar_id: str, time_min: datetime, time_max: datetime) -> list[dict]:
    """Return every event (managed or not) in the range — merge candidate search."""
    return list(
        _paginate_events_list(
            calendar_id, time_min, time_max, label="events.list(all)"
        )
    )


def get_event(calendar_id: str, event_id: str) -> dict:
    """Full event resource (for preview: description, attendees, etc.)."""
    service = get_service()
    return _execute_with_retry(
        service.events().get(calendarId=calendar_id, eventId=event_id),
        label="events.get",
    )


def patch_event_merge(calendar_id: str, event_id: str, body: dict) -> dict:
    """Patch event without injecting TOOL_MARKER (duplicate merge for unmanaged-only groups)."""
    service = get_service()
    return _execute_with_retry(
        service.events().patch(calendarId=calendar_id, eventId=event_id, body=body),
        label="events.patch(merge)",
    )


# ── Write / update helpers (internal) ────────────────────────────────

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


# ── Public write API ─────────────────────────────────────────────────


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
            result = _execute_with_retry(
                service.events().update(
                    calendarId=calendar_id, eventId=event_id, body=update_body,
                ),
                label="events.update",
            )
        else:
            result = _execute_with_retry(
                service.events().patch(
                    calendarId=calendar_id, eventId=event_id, body=body,
                ),
                label="events.patch",
            )
        return "updated", result["id"]
    else:
        result = _execute_with_retry(
            service.events().insert(calendarId=calendar_id, body=body),
            label="events.insert",
        )
        return "created", result["id"]


def delete_event(calendar_id: str, event_id: str) -> None:
    """Delete a single event (Ch26.4)."""
    service = get_service()
    _execute_with_retry(
        service.events().delete(calendarId=calendar_id, eventId=event_id),
        label="events.delete",
    )


def detach_event(calendar_id: str, event_id: str) -> None:
    """Remove tool markers from event (Ch17.3 DETACH-01)."""
    service = get_service()
    existing = _execute_with_retry(
        service.events().get(calendarId=calendar_id, eventId=event_id),
        label="events.get(detach)",
    )
    private = (existing.get("extendedProperties") or {}).get("private", {})
    for key in DETACH_KEYS:
        private.pop(key, None)
    body = {"extendedProperties": {"private": private if private else {}}}
    _execute_with_retry(
        service.events().patch(calendarId=calendar_id, eventId=event_id, body=body),
        label="events.patch(detach)",
    )


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
