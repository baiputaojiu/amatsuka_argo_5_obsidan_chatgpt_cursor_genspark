from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from analyst_forecast.domain.market import (
    MarketBar,
    MarketDataRequest,
    MarketDataUnavailable,
    MarketSeries,
)


class YFinanceMarketDataProvider:
    name = "yfinance"

    def fetch(self, request: MarketDataRequest) -> MarketSeries:
        try:
            import yfinance as yf

            frame = yf.download(
                request.symbol,
                start=request.start.isoformat(),
                end=(request.end + timedelta(days=1)).isoformat(),
                auto_adjust=False,
                actions=True,
                progress=False,
                threads=False,
                multi_level_index=False,
            )
        except Exception as error:
            raise MarketDataUnavailable(
                f"yfinanceから{request.symbol}を取得できません: {error}"
            ) from error
        if frame is None or frame.empty:
            raise MarketDataUnavailable(f"yfinanceに{request.symbol}の指定期間データがありません")
        if getattr(frame.columns, "nlevels", 1) > 1:
            try:
                frame = frame.xs(request.symbol, axis=1, level=-1)
            except Exception as error:
                raise MarketDataUnavailable("yfinance応答の列構造を解釈できません") from error

        bars: list[MarketBar] = []
        for index, row in frame.iterrows():
            values = {name: _number(row, name) for name in ("Open", "High", "Low", "Close")}
            day = index.date()
            if request.start <= day <= request.end:
                bars.append(
                    MarketBar(
                        date=day,
                        open=values["Open"],
                        high=values["High"],
                        low=values["Low"],
                        close=values["Close"],
                        adjusted_open=values["Open"],
                        adjusted_close=values["Close"],
                    )
                )

        return MarketSeries(
            provider=self.name,
            symbol=request.symbol,
            currency=request.currency,
            adjustment_type="split_adjusted_ohlc",
            frequency="1d",
            retrieved_at=datetime.now(UTC),
            bars=tuple(bars),
        )


def _number(row: Any, column: str) -> Decimal:
    value = row[column]
    if value is None or value != value:
        raise MarketDataUnavailable(f"yfinance応答の{column}に欠損があります")
    return Decimal(str(float(value)))
