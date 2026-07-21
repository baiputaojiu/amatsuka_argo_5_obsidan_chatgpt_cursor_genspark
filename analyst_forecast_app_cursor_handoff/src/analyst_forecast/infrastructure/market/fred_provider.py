from __future__ import annotations

import os
from datetime import UTC, datetime
from decimal import Decimal

from analyst_forecast.domain.market import (
    MarketBar,
    MarketDataRequest,
    MarketDataUnavailable,
    MarketSeries,
)


class FredMarketDataProvider:
    name = "fred"

    def __init__(self, *, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("FRED_API_KEY")

    def fetch(self, request: MarketDataRequest) -> MarketSeries:
        if not self._api_key:
            raise MarketDataUnavailable(
                "FRED_API_KEYが未設定です。値は推測せず、キー設定またはCSVを使用してください。"
            )
        try:
            from fredapi import Fred

            values = Fred(api_key=self._api_key).get_series(
                request.symbol,
                observation_start=request.start,
                observation_end=request.end,
            )
        except Exception as error:
            raise MarketDataUnavailable(
                f"FREDから{request.symbol}を取得できません: {error}"
            ) from error
        if values is None or values.empty:
            raise MarketDataUnavailable(f"FREDに{request.symbol}の指定期間データがありません")

        bars: list[MarketBar] = []
        for index, value in values.items():
            if value is None or value != value:
                raise MarketDataUnavailable("FRED応答に欠損値があります")
            numeric = Decimal(str(float(value)))
            bars.append(
                MarketBar(
                    date=index.date(),
                    open=numeric,
                    high=numeric,
                    low=numeric,
                    close=numeric,
                    adjusted_open=numeric,
                    adjusted_close=numeric,
                )
            )
        return MarketSeries(
            provider=self.name,
            symbol=request.symbol,
            currency=request.currency,
            adjustment_type="none",
            frequency="1d",
            retrieved_at=datetime.now(UTC),
            bars=tuple(bars),
        )
