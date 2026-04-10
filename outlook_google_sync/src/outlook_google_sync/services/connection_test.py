"""Ch24-25: Connection test services."""

from __future__ import annotations
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger("outlook_google_sync")

# Outlook OlObjectClass (代表値) — 既定カレンダーでよく出るもの
_OL_CLASS_NAMES: dict[int, str] = {
    26: "AppointmentItem（通常の予定）",
    53: "MeetingItem（会議）",
    48: "DistListItem",
}


def _fmt_com_dt(val: Any) -> str:
    if val is None:
        return "（不明）"
    try:
        if hasattr(val, "strftime"):
            return val.strftime("%Y-%m-%d %H:%M")
        return str(val)
    except Exception:
        return str(val)


def _first_item_detail_lines(first: Any) -> list[str]:
    subject = getattr(first, "Subject", "") or "（件名なし）"
    start_s = _fmt_com_dt(getattr(first, "Start", None))
    end_s = _fmt_com_dt(getattr(first, "End", None))
    all_day = bool(getattr(first, "AllDayEvent", False))
    recurring = bool(getattr(first, "IsRecurring", False))
    ol_class = getattr(first, "Class", None)
    class_label = _OL_CLASS_NAMES.get(int(ol_class), f"Class={ol_class!r}") if ol_class is not None else "Class=不明"
    msg_class = getattr(first, "MessageClass", "") or ""
    loc = getattr(first, "Location", "") or ""
    loc_line = f"  場所: {loc}" if loc.strip() else "  場所: （なし）"
    return [
        "【先頭1件（[Start] 昇順・繰り返し展開なしのフォルダ既定ビューに近い並び）】",
        f"  件名: {subject}",
        f"  開始: {start_s}  終了: {end_s}",
        f"  終日: {'はい' if all_day else 'いいえ'}  繰り返し: {'はい' if recurring else 'いいえ'}",
        f"  種別: {class_label}",
        f"  MessageClass: {msg_class or '（なし）'}",
        loc_line,
    ]


# ── Outlook COM test (Ch24.2) ──

def test_outlook_com(start_dt: datetime | None = None, end_dt: datetime | None = None) -> tuple[bool, str]:
    """Ch24.2: COM connection test."""
    try:
        import win32com.client  # type: ignore
    except ImportError:
        return False, "pywin32 未導入。手動ICSまたはマクロ連携をお使いください。"

    try:
        app = win32com.client.Dispatch("Outlook.Application")
        ns = app.GetNamespace("MAPI")
        folder = ns.GetDefaultFolder(9)
        items = folder.Items
        items.Sort("[Start]")
        count = int(items.Count) if hasattr(items, "Count") else 0

        if count == 0:
            msg = "接続成功。予定が0件のため読取確認スキップ（警告）。"
            logger.warning(msg)
            return True, msg

        first = items.Item(1)
        _ = getattr(first, "Subject", "")
        detail_lines = _first_item_detail_lines(first)

        restrict_lines: list[str] = []
        if start_dt is not None and end_dt is not None:
            restrict_fmt = "%m/%d/%Y %H:%M %p"
            restriction = (
                f'[Start] < "{end_dt.strftime(restrict_fmt)}" '
                f'AND [End] > "{start_dt.strftime(restrict_fmt)}"'
            )
            restrict_count: int | None = None
            restrict_exc: str | None = None
            try:
                restricted = items.Restrict(restriction)
                restrict_count = int(restricted.Count) if hasattr(restricted, "Count") else None
            except Exception as ex:
                restrict_exc = type(ex).__name__
            restrict_lines.append("")
            restrict_lines.append("【同期と同じ Restrict（期間オーバーラップ）※読取ロジック検証用】")
            restrict_lines.append(f"  期間: {start_dt.isoformat()} ～ {end_dt.isoformat()}")
            if restrict_exc:
                restrict_lines.append(f"  Restrict 失敗（同期側は全件走査にフォールバック）: {restrict_exc}")
            elif restrict_count is not None:
                restrict_lines.append(f"  該当件数（Restrict 直後の件数）: {restrict_count} 件")
            restrict_lines.append(
                "  ※件数は同期のフィルタ（非公開・重複展開等）前の目安です。"
            )

        msg = (
            "Outlook COM 接続テスト成功（疎通と最小読取確認済み）。同期の全操作の成功は保証しません。\n"
            + "\n".join(detail_lines)
            + ("\n" + "\n".join(restrict_lines) if restrict_lines else "")
        )
        logger.info(msg)
        return True, msg
    except Exception as exc:
        msg = f"Outlook COM 接続テスト失敗: {exc}"
        logger.error(msg)
        return False, msg


# ── ICS test (Ch25 detailed checks 1-11) ──

def test_ics_file(
    path: str,
    range_start: datetime | None = None,
    range_end: datetime | None = None,
) -> tuple[bool, str]:
    """Ch25: Detailed ICS validation."""
    p = Path(path)
    results: list[str] = []

    # 1. File exists
    if not p.exists():
        return False, "ICS ファイルが存在しません"
    results.append("✓ ファイル存在")

    # 2. Readable
    try:
        raw = p.read_bytes()
    except Exception as exc:
        return False, f"ICS ファイル読み取り不可: {exc}"
    results.append("✓ 読み取り可能")

    # 6. Encoding check
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("utf-8", errors="replace")
            results.append("⚠ 文字コードに問題あり（置換文字使用）")
        except Exception:
            return False, "ICS ファイルの文字コードが異常です"
    else:
        results.append("✓ UTF-8 デコード成功")

    # 3. BEGIN:VCALENDAR
    if "BEGIN:VCALENDAR" not in text:
        return False, "BEGIN:VCALENDAR が見つかりません"
    results.append("✓ BEGIN:VCALENDAR あり")

    # 4. BEGIN:VEVENT
    if "BEGIN:VEVENT" not in text:
        return False, "BEGIN:VEVENT が見つかりません"
    results.append("✓ BEGIN:VEVENT あり")

    # 5. Parseable
    try:
        from icalendar import Calendar
        cal = Calendar.from_ical(raw)
    except Exception as exc:
        return False, f"ICS パース失敗: {exc}"
    results.append("✓ パース成功")

    # 7. DTSTART/DTEND readable
    vevents = list(cal.walk("VEVENT"))
    dt_ok = False
    for comp in vevents[:5]:
        ds = comp.get("dtstart")
        if ds:
            try:
                _ = ds.dt
                dt_ok = True
                break
            except Exception:
                pass
    if dt_ok:
        results.append("✓ DTSTART 読取成功")
    else:
        results.append("⚠ DTSTART が読取不可")

    # 8. UID
    has_uid = any(comp.get("uid") for comp in vevents[:10])
    if has_uid:
        results.append("✓ UID あり")
    else:
        results.append("⚠ UID なし（fallbackキー依存）")

    # 9. Events in range
    if range_start and range_end:
        in_range = 0
        for comp in vevents:
            ds = comp.get("dtstart")
            de = comp.get("dtend")
            if ds and de:
                try:
                    s = ds.dt
                    e = de.dt
                    if isinstance(s, datetime) and isinstance(e, datetime):
                        if e > range_start and s < range_end:
                            in_range += 1
                except Exception:
                    pass
        if in_range == 0:
            results.append("⚠ 指定期間内のイベントが0件（警告・失敗扱いではない）")
        else:
            results.append(f"✓ 期間内イベント {in_range} 件")

    # 10. TZID / VTIMEZONE
    has_tz = any(cal.walk("VTIMEZONE")) or "TZID" in text
    if has_tz:
        results.append("✓ タイムゾーン情報あり")
    else:
        results.append("⚠ タイムゾーン情報なし")

    # 11. File mtime (log only)
    mtime = datetime.fromtimestamp(p.stat().st_mtime)
    results.append(f"📎 ファイル更新日時: {mtime.isoformat()}")

    summary = "\n".join(results)
    logger.info("ICS connection test:\n%s", summary)
    return True, f"ICS 接続テスト成功。\n{summary}"


# ── Google test (Ch24.4 two-stage) ──

def test_google(target_calendar_id: str | None) -> tuple[bool, str]:
    """Ch24.4: Two-stage Google connection test."""
    from ..connectors.google_calendar import list_calendars

    results: list[str] = []

    # Stage 1: Basic connectivity (Ch24.4.1)
    try:
        items = list_calendars()
        results.append("✓ Google 基本接続成功（トークン有効・calendarList取得済み）")
    except Exception as exc:
        msg = f"Google 基本接続テスト失敗: {exc}"
        logger.error(msg)
        return False, msg

    # Stage 2: Calendar confirmation (Ch24.4.2)
    if target_calendar_id:
        # calendarList の主カレンダーは id がメールアドレスで、API の calendarId としての
        # "primary" エイリアスと一致しないことが多い。
        found = any(i.get("id") == target_calendar_id for i in items)
        if not found and target_calendar_id == "primary":
            found = any(i.get("primary") for i in items)
        if found:
            results.append(f"✓ 対象カレンダー確認済み: {target_calendar_id}")
        else:
            msg = f"対象カレンダーが見つかりません: {target_calendar_id}"
            results.append(f"✗ {msg}")
            logger.warning(msg)
            return False, "\n".join(results)
    else:
        # CT-CAL-01/02: unselected
        results.append("⚠ 対象カレンダー未選択のため、カレンダー存在確認は未実施です。")

    results.append("疎通と最小読取は問題ありません。同期の全操作の成功は保証しません。")
    summary = "\n".join(results)
    logger.info("Google connection test:\n%s", summary)
    return True, summary
