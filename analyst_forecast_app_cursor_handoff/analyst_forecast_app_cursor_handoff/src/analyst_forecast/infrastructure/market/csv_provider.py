from __future__ import annotations

import csv
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import ClassVar

from analyst_forecast.domain.market import (
    MarketBar,
    MarketDataRequest,
    MarketDataUnavailable,
    MarketSeries,
)


class CsvMarketDataProvider:
    name = "csv"
    _required_columns: ClassVar[set[str]] = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "adjusted_open",
        "adjusted_close",
    }

    def __init__(self, *, csv_path: Path) -> None:
        self.csv_path = csv_path

    def fetch(self, request: MarketDataRequest) -> MarketSeries:
        if not self.csv_path.is_file():
            raise MarketDataUnavailable(f"市場CSVがありません: {self.csv_path}")
        try:
            with self.csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                columns = set(reader.fieldnames or [])
                missing = self._required_columns - columns
                if missing:
                    raise MarketDataUnavailable(
                        "市場CSVの必須列が不足しています: " + ", ".join(sorted(missing))
                    )
                bars = tuple(
                    self._parse_row(row, line_number)
                    for line_number, row in enumerate(reader, start=2)
                    if self._in_range(row.get("date"), request)
                )
        except UnicodeDecodeError as error:
            raise MarketDataUnavailable("市場CSVをUTF-8として読めません") from error

        return MarketSeries(
            provider=self.name,
            symbol=request.symbol,
            currency=request.currency,
            adjustment_type="split_adjusted_ohlc",
            frequency="1d",
            retrieved_at=datetime.now(UTC),
            bars=bars,
        )

    def _in_range(self, raw_date: str | None, request: MarketDataRequest) -> bool:
        if raw_date is None:
            raise MarketDataUnavailable("市場CSVの日付が空です")
        try:
            day = date.fromisoformat(raw_date)
        except ValueError as error:
            raise MarketDataUnavailable(f"市場CSVの日付形式が不正です: {raw_date}") from error
        return request.start <= day <= request.end

    def _parse_row(self, row: dict[str, str | None], line_number: int) -> MarketBar:
        try:
            day = date.fromisoformat(_required(row, "date"))
            return MarketBar(
                date=day,
                open=Decimal(_required(row, "open")),
                high=Decimal(_required(row, "high")),
                low=Decimal(_required(row, "low")),
                close=Decimal(_required(row, "close")),
                adjusted_open=Decimal(_required(row, "adjusted_open")),
                adjusted_close=Decimal(_required(row, "adjusted_close")),
            )
        except (ValueError, InvalidOperation) as error:
            raise MarketDataUnavailable(
                f"市場CSVの{line_number}行目に不正な値があります: {error}"
            ) from error


def _required(row: dict[str, str | None], key: str) -> str:
    value = row.get(key)
    if value is None or not value.strip():
        raise ValueError(f"{key}が空です")
    return value.strip()
