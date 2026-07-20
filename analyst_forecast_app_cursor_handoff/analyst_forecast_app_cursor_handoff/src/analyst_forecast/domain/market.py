from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol


class MarketDataUnavailable(RuntimeError):
    """市場データを安全に取得・使用できない場合。"""


def _decimal(value: Decimal | str | int | float) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


@dataclass(frozen=True, slots=True)
class MarketBar:
    date: dt.date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    adjusted_open: Decimal
    adjusted_close: Decimal

    def __post_init__(self) -> None:
        for field_name in (
            "open",
            "high",
            "low",
            "close",
            "adjusted_open",
            "adjusted_close",
        ):
            object.__setattr__(self, field_name, _decimal(getattr(self, field_name)))
        if self.high < self.low:
            raise ValueError("高値は安値以上である必要があります")

    @classmethod
    def from_prices(
        cls,
        day: dt.date,
        open_price: Decimal | str | int | float,
        close_price: Decimal | str | int | float,
        *,
        high: Decimal | str | int | float | None = None,
        low: Decimal | str | int | float | None = None,
    ) -> MarketBar:
        opening = _decimal(open_price)
        closing = _decimal(close_price)
        high_value = _decimal(high) if high is not None else max(opening, closing)
        low_value = _decimal(low) if low is not None else min(opening, closing)
        return cls(
            date=day,
            open=opening,
            high=high_value,
            low=low_value,
            close=closing,
            adjusted_open=opening,
            adjusted_close=closing,
        )


@dataclass(frozen=True, slots=True)
class MarketDataRequest:
    symbol: str
    currency: str
    start: dt.date
    end: dt.date

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbolは空にできません")
        if self.end < self.start:
            raise ValueError("市場データの終了日は開始日以後にしてください")


@dataclass(frozen=True, slots=True)
class MarketSeries:
    provider: str
    symbol: str
    currency: str
    adjustment_type: str
    frequency: str
    retrieved_at: dt.datetime
    bars: tuple[MarketBar, ...]

    def __post_init__(self) -> None:
        if not self.bars:
            raise MarketDataUnavailable("市場データが0件です")
        dates = [bar.date for bar in self.bars]
        if dates != sorted(dates):
            raise MarketDataUnavailable("市場データの日付が昇順ではありません")
        if len(dates) != len(set(dates)):
            raise MarketDataUnavailable("市場データに重複日があります")


class MarketDataProvider(Protocol):
    name: str

    def fetch(self, request: MarketDataRequest) -> MarketSeries:
        """指定範囲の市場系列を返す。取得不能値を補完してはならない。"""
        ...
