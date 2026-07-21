from __future__ import annotations

import calendar
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import TextIO

from analyst_forecast.application.runs import CreateRunRequest, CreateRunResult, create_run
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.domain.models import Medium


class WizardCancelled(Exception):
    """対話wizardがキャンセルされた。"""


def interactive_start(
    settings: AppSettings,
    *,
    input_func: Callable[[], str] = input,
    output: TextIO | None = None,
    now: datetime | None = None,
) -> CreateRunResult:
    timestamp = now or datetime.now(UTC)
    today = timestamp.date()
    _emit(output, "対話式で案件を作成します。空行でキャンセルできます。")
    _emit(output, "入力をやり直す場合は redo と入力してください。")

    while True:
        name = _ask(input_func, output, "分析対象者名（例: 匿名アナリストA）")
        default_start = _add_months(today, -settings.default_period_months)
        period_start = _ask_date(
            input_func,
            output,
            f"調査開始日 YYYY-MM-DD（既定: {default_start.isoformat()}）",
            default=default_start,
        )
        period_end = _ask_date(
            input_func,
            output,
            f"調査終了日 YYYY-MM-DD（既定: {today.isoformat()}）",
            default=today,
        )
        evaluation_as_of = _ask_date(
            input_func,
            output,
            f"評価基準日 YYYY-MM-DD（既定: {today.isoformat()}）",
            default=today,
        )
        media = _ask_media(input_func, output)
        focus_raw = _ask(
            input_func,
            output,
            "重点対象（カンマ区切り、空欄可）",
            allow_empty=True,
        )
        focus_targets = [item.strip() for item in focus_raw.split(",") if item.strip()]
        if settings.cursor_model is None or settings.chatgpt_model is None:
            _emit(
                output,
                "警告: 高性能モデル名が未設定です。意味判断の前に設定してください。",
            )
        _emit(output, "確認:")
        _emit(output, f"- 対象者: {name}")
        _emit(output, f"- 期間: {period_start} ～ {period_end}")
        _emit(output, f"- 基準日: {evaluation_as_of}")
        _emit(output, f"- 媒体: {', '.join(m.value for m in media)}")
        _emit(output, f"- 重点対象: {', '.join(focus_targets) or 'なし'}")
        confirm = _ask(
            input_func,
            output,
            "この内容で作成しますか？ yes / redo / cancel",
            allow_empty=False,
        ).casefold()
        if confirm in {"cancel", "c", "no", "n"}:
            raise WizardCancelled("案件作成をキャンセルしました。")
        if confirm in {"redo", "r"}:
            continue
        if confirm not in {"yes", "y"}:
            _emit(output, "yes / redo / cancel のいずれかを入力してください。")
            continue
        return create_run(
            settings,
            CreateRunRequest(
                canonical_name=name,
                period_start=period_start,
                period_end=period_end,
                evaluation_as_of=evaluation_as_of,
                selected_media=media,
                focus_targets=focus_targets,
            ),
            now=timestamp,
        )


def _ask_media(input_func: Callable[[], str], output: TextIO | None) -> list[Medium]:
    _emit(
        output,
        "媒体を複数選択（カンマ区切り）: youtube, blog, x, web（既定: youtube）",
    )
    while True:
        raw = _ask(input_func, output, "媒体", allow_empty=True)
        if not raw:
            return [Medium.YOUTUBE]
        try:
            values = [Medium(item.strip().lower()) for item in raw.split(",") if item.strip()]
            if not values:
                raise ValueError("媒体が空です")
            return values
        except ValueError:
            _emit(output, "youtube / blog / x / web から選んでください。")


def _ask_date(
    input_func: Callable[[], str],
    output: TextIO | None,
    prompt: str,
    *,
    default: date,
) -> date:
    while True:
        raw = _ask(input_func, output, prompt, allow_empty=True)
        if not raw:
            return default
        try:
            return date.fromisoformat(raw)
        except ValueError:
            _emit(output, "日付は YYYY-MM-DD 形式で入力してください。")


def _ask(
    input_func: Callable[[], str],
    output: TextIO | None,
    prompt: str,
    *,
    allow_empty: bool = False,
) -> str:
    while True:
        _emit(output, prompt)
        value = str(input_func()).strip()
        if value.casefold() == "cancel":
            raise WizardCancelled("案件作成をキャンセルしました。")
        if value or allow_empty:
            return value
        _emit(output, "入力が必要です。cancel で中止できます。")


def _emit(output: TextIO | None, message: str) -> None:
    if output is not None:
        output.write(message + "\n")
        output.flush()
    else:
        print(message)


def _add_months(value: date, months: int) -> date:
    year = value.year + (value.month - 1 + months) // 12
    month = (value.month - 1 + months) % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)
