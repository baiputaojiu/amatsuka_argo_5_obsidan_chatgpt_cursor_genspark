from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from analyst_forecast.domain.market import (
    MarketBar,
    MarketDataRequest,
    MarketSeries,
    ProviderError,
)

MAX_ATTEMPTS = 3
BASE_BACKOFF_SECONDS = 1.0


class YFinanceMarketDataProvider:
    name = "yfinance"

    def __init__(
        self,
        *,
        sleeper: Callable[[float], None] | None = None,
        downloader: Callable[..., Any] | None = None,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        self._sleeper = sleeper or time.sleep
        self._downloader = downloader
        self._max_attempts = max_attempts

    def fetch(self, request: MarketDataRequest) -> MarketSeries:
        last_error: ProviderError | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                frame = self._download(request)
                if frame is None or getattr(frame, "empty", True):
                    raise ProviderError(
                        code="no_data",
                        message=f"yfinanceに{request.symbol}の指定期間データがありません",
                        retryable=False,
                        attempt_count=attempt,
                        guidance=(
                            "真にデータが無い可能性があります。"
                            "CSV providerへ切り替えて再評価してください。"
                        ),
                    )
                return self._to_series(request, frame)
            except ProviderError as error:
                last_error = ProviderError(
                    code=error.code,
                    message=error.message,
                    retryable=error.retryable,
                    attempt_count=attempt,
                    guidance=error.guidance,
                )
                if not error.retryable or attempt >= self._max_attempts:
                    if error.retryable:
                        raise ProviderError(
                            code=error.code,
                            message=(f"{error.message}（再試行{self._max_attempts}回で上限超過）"),
                            retryable=False,
                            attempt_count=attempt,
                            guidance=(
                                "rate limitまたは一時障害の可能性があります。"
                                "しばらく待つか、--provider csv --csv-path で再実行してください。"
                            ),
                        ) from error
                    raise last_error from error
                self._sleeper(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)))
        assert last_error is not None
        raise last_error

    def _download(self, request: MarketDataRequest) -> Any:
        try:
            if self._downloader is not None:
                return self._downloader(
                    request.symbol,
                    start=request.start.isoformat(),
                    end=(request.end + timedelta(days=1)).isoformat(),
                )
            import yfinance as yf

            return yf.download(
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
            code, retryable, guidance = _classify_exception(error, request.symbol)
            raise ProviderError(
                code=code,
                message=f"yfinanceから{request.symbol}を取得できません: {error}",
                retryable=retryable,
                guidance=guidance,
            ) from error

    def _to_series(self, request: MarketDataRequest, frame: Any) -> MarketSeries:
        if getattr(frame.columns, "nlevels", 1) > 1:
            try:
                frame = frame.xs(request.symbol, axis=1, level=-1)
            except Exception as error:
                raise ProviderError(
                    code="invalid_response",
                    message="yfinance応答の列構造を解釈できません",
                    retryable=False,
                ) from error

        bars: list[MarketBar] = []
        for index, row in frame.iterrows():
            values = {name: _number(row, name) for name in ("Open", "High", "Low", "Close")}
            day = index.date() if hasattr(index, "date") else index
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
        if not bars:
            raise ProviderError(
                code="no_data",
                message=f"yfinanceに{request.symbol}の指定期間データがありません",
                retryable=False,
                guidance="CSV providerへ切り替えて再評価してください。",
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


def _classify_exception(error: Exception, symbol: str) -> tuple[str, bool, str]:
    text = f"{type(error).__name__}: {error}".lower()
    if "ratelimit" in text or "too many" in text or "429" in text:
        return (
            "rate_limit",
            True,
            "rate limitです。待機後に再試行するか、CSVへ切り替えてください。",
        )
    if any(token in text for token in ("timeout", "connection", "network", "temporarily")):
        return (
            "network_error",
            True,
            "ネットワーク障害の可能性があります。再試行またはCSV切替を検討してください。",
        )
    if any(token in text for token in ("delisted", "invalid", "not found", "symbol")):
        return (
            "invalid_symbol",
            False,
            f"symbol `{symbol}` が無効な可能性があります。mappingを見直してください。",
        )
    return (
        "provider_error",
        False,
        "取得不能です。推測せずCSV等の代替データで再評価してください。",
    )


def _number(row: Any, column: str) -> Decimal:
    value = row[column]
    if value is None or value != value:
        raise ProviderError(
            code="invalid_response",
            message=f"yfinance応答の{column}に欠損があります",
            retryable=False,
        )
    return Decimal(str(float(value)))
