import os
from datetime import date

import pytest

from analyst_forecast.domain.market import MarketDataRequest
from analyst_forecast.infrastructure.market.fred_provider import FredMarketDataProvider
from analyst_forecast.infrastructure.market.yfinance_provider import YFinanceMarketDataProvider

pytestmark = pytest.mark.integration


def network_tests_enabled() -> bool:
    return os.environ.get("RUN_NETWORK_TESTS") == "1"


@pytest.mark.skipif(not network_tests_enabled(), reason="RUN_NETWORK_TESTS=1 のときだけ実行")
def test_yfinance_known_symbol_can_be_fetched() -> None:
    series = YFinanceMarketDataProvider().fetch(
        MarketDataRequest(
            symbol="AAPL",
            currency="USD",
            start=date(2024, 1, 2),
            end=date(2024, 1, 10),
        )
    )

    assert series.provider == "yfinance"
    assert series.bars
    assert series.adjustment_type == "split_adjusted_ohlc"


@pytest.mark.skipif(not network_tests_enabled(), reason="RUN_NETWORK_TESTS=1 のときだけ実行")
def test_yfinance_split_is_normalized_without_dividend_adjustment() -> None:
    series = YFinanceMarketDataProvider().fetch(
        MarketDataRequest(
            symbol="AAPL",
            currency="USD",
            start=date(2020, 8, 27),
            end=date(2020, 9, 2),
        )
    )

    prices = [bar.adjusted_close for bar in series.bars]
    assert max(prices) / min(prices) < 2


@pytest.mark.skipif(
    not network_tests_enabled() or not os.environ.get("FRED_API_KEY"),
    reason="RUN_NETWORK_TESTS=1 と FRED_API_KEY があるときだけ実行",
)
def test_fred_known_series_can_be_fetched() -> None:
    series = FredMarketDataProvider(api_key=os.environ["FRED_API_KEY"]).fetch(
        MarketDataRequest(
            symbol="DGS10",
            currency="PERCENT",
            start=date(2024, 1, 2),
            end=date(2024, 1, 10),
        )
    )

    assert series.provider == "fred"
    assert series.bars
