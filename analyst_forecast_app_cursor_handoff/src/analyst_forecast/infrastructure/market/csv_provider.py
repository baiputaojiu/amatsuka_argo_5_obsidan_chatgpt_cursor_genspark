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
                has_symbol = "symbol" in columns
                has_currency = "currency" in columns
                bars = self._read_bars(
                    reader,
                    request=request,
                    has_symbol=has_symbol,
                    has_currency=has_currency,
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

    def _read_bars(
        self,
        reader: csv.DictReader[str],
        *,
        request: MarketDataRequest,
        has_symbol: bool,
        has_currency: bool,
    ) -> tuple[MarketBar, ...]:
        symbol_matched = False
        seen_keys: set[tuple[str, date] | date] = set()
        bars: list[MarketBar] = []
        for line_number, row in enumerate(reader, start=2):
            if has_symbol:
                row_symbol = (row.get("symbol") or "").strip()
                if not row_symbol:
                    raise MarketDataUnavailable(f"市場CSVの{line_number}行目のsymbolが空です")
                if row_symbol != request.symbol:
                    continue
                symbol_matched = True
                if has_currency:
                    row_currency = (row.get("currency") or "").strip()
                    if not row_currency:
                        raise MarketDataUnavailable(f"市場CSVの{line_number}行目のcurrencyが空です")
                    if row_currency != request.currency:
                        raise MarketDataUnavailable(
                            f"市場CSVのcurrencyがrequestと不一致です: "
                            f"{row_currency} != {request.currency}"
                        )

            raw_date = row.get("date")
            if not self._in_range(raw_date, request):
                continue

            day = date.fromisoformat(_required(row, "date"))
            dup_key: tuple[str, date] | date
            if has_symbol:
                dup_key = (request.symbol, day)
                dup_message = (
                    f"市場CSVに同一symbol+dateの重複があります: {request.symbol} {day.isoformat()}"
                )
            else:
                dup_key = day
                dup_message = f"市場CSVに同一dateの重複があります: {day.isoformat()}"
            if dup_key in seen_keys:
                raise MarketDataUnavailable(dup_message)
            seen_keys.add(dup_key)
            bars.append(self._parse_row(row, line_number))

        if has_symbol and not symbol_matched:
            raise MarketDataUnavailable(f"市場CSVにsymbolがありません: {request.symbol}")
        return tuple(bars)

    def _in_range(self, raw_date: str | None, request: MarketDataRequest) -> bool:
        if raw_date is None:
            raise MarketDataUnavailable("市場CSVの日付が空です")
        try:
            day = date.fromisoformat(raw_date.strip())
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
