from __future__ import annotations

from datetime import date, datetime, time

from dateutil.relativedelta import relativedelta


def day_range(start_date, end_date):
    return datetime.combine(start_date, time.min), datetime.combine(end_date, time.max)


def default_sync_range(today: date | None = None) -> tuple[date, date]:
    """標準同期期間: 今日 − 1ヶ月 〜 今日 ＋ 2ヶ月。"""
    base = today if today is not None else date.today()
    start = base + relativedelta(months=-1)
    end = base + relativedelta(months=+2)
    return start, end
