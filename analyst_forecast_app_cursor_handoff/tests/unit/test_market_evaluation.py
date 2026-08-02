from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import func, select

from analyst_forecast.application.evaluation import evaluate_component
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.domain.market import (
    MarketBar,
    MarketDataRequest,
    MarketDataUnavailable,
    MarketSeries,
)
from analyst_forecast.infrastructure.db.models import (
    EvaluationRecord,
    EvaluationSnapshotRecord,
    MarketSeriesRecord,
)
from analyst_forecast.infrastructure.db.session import create_session_factory
from analyst_forecast.infrastructure.market.csv_provider import CsvMarketDataProvider
from analyst_forecast.infrastructure.market.fred_provider import FredMarketDataProvider
from helpers_pipeline_v2 import import_locked_component


class FixedProvider:
    name = "fixture"

    def __init__(self, bars: tuple[MarketBar, ...]) -> None:
        self.bars = bars

    def fetch(self, request: MarketDataRequest) -> MarketSeries:
        return MarketSeries(
            provider=self.name,
            symbol=request.symbol,
            currency=request.currency,
            adjustment_type="split_adjusted_ohlc",
            frequency="1d",
            retrieved_at=datetime(2026, 7, 20, 13, tzinfo=UTC),
            bars=self.bars,
        )


class UnavailableProvider:
    name = "unavailable-fixture"

    def fetch(self, request: MarketDataRequest) -> MarketSeries:
        raise MarketDataUnavailable("fixtureでは値を取得できません")


def imported_component_id(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
    *,
    direction: str = "up",
) -> str:
    return import_locked_component(
        settings,
        run_result,
        source_result,
        tmp_path,
        direction=direction,
    )


def test_upward_direction_and_return_are_evaluated_without_network(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
) -> None:
    component_id = imported_component_id(settings, run_result, source_result, tmp_path)
    provider = FixedProvider(
        (
            MarketBar(
                date=date(2026, 1, 13),
                open=Decimal("100"),
                high=Decimal("104"),
                low=Decimal("98"),
                close=Decimal("102"),
                adjusted_open=Decimal("100"),
                adjusted_close=Decimal("102"),
            ),
            MarketBar(
                date=date(2026, 4, 13),
                open=Decimal("108"),
                high=Decimal("112"),
                low=Decimal("107"),
                close=Decimal("110"),
                adjusted_open=Decimal("108"),
                adjusted_close=Decimal("110"),
            ),
        )
    )

    result = evaluate_component(
        settings,
        component_id=component_id,
        provider=provider,
        as_of=date(2026, 4, 13),
    )

    assert result.direction_result == "hit"
    assert result.evaluation_status == "expired_hit"
    assert result.start_price == Decimal("100")
    assert result.end_price == Decimal("110")
    assert result.actual_return == Decimal("0.1")
    assert result.max_favorable_excursion == Decimal("0.12")
    assert result.max_adverse_excursion == Decimal("-0.02")
    assert result.method_version == "direction-v2.0.0"


def test_downward_direction_mfe_mae_are_direction_aware(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
) -> None:
    component_id = imported_component_id(
        settings, run_result, source_result, tmp_path, direction="down"
    )
    provider = FixedProvider(
        (
            MarketBar(
                date=date(2026, 1, 13),
                open=Decimal("100"),
                high=Decimal("104"),
                low=Decimal("98"),
                close=Decimal("102"),
                adjusted_open=Decimal("100"),
                adjusted_close=Decimal("102"),
            ),
            MarketBar(
                date=date(2026, 4, 13),
                open=Decimal("94"),
                high=Decimal("104"),
                low=Decimal("88"),
                close=Decimal("92"),
                adjusted_open=Decimal("94"),
                adjusted_close=Decimal("92"),
            ),
        )
    )
    result = evaluate_component(
        settings,
        component_id=component_id,
        provider=provider,
        as_of=date(2026, 4, 13),
    )
    assert result.direction_result == "hit"
    assert result.max_favorable_excursion == Decimal("0.12")
    assert result.max_adverse_excursion == Decimal("0.04")


def test_unavailable_data_is_unevaluable_and_never_guessed(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
) -> None:
    component_id = imported_component_id(settings, run_result, source_result, tmp_path)

    result = evaluate_component(
        settings,
        component_id=component_id,
        provider=UnavailableProvider(),
        as_of=date(2026, 4, 13),
    )

    assert result.evaluation_status == "unevaluable"
    assert result.start_price is None
    assert result.end_price is None
    assert "取得できません" in (result.unevaluable_reason or "")


def test_same_as_of_is_idempotent_and_another_as_of_keeps_history(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
) -> None:
    component_id = imported_component_id(settings, run_result, source_result, tmp_path)
    provider = FixedProvider(
        (
            MarketBar.from_prices(date(2026, 1, 13), "100", "102", high="103", low="99"),
            MarketBar.from_prices(date(2026, 3, 13), "104", "106", high="107", low="103"),
            MarketBar.from_prices(date(2026, 4, 13), "108", "110", high="111", low="107"),
        )
    )

    first = evaluate_component(
        settings,
        component_id=component_id,
        provider=provider,
        as_of=date(2026, 3, 13),
    )
    repeated = evaluate_component(
        settings,
        component_id=component_id,
        provider=provider,
        as_of=date(2026, 3, 13),
    )
    final = evaluate_component(
        settings,
        component_id=component_id,
        provider=provider,
        as_of=date(2026, 4, 13),
    )

    assert repeated.evaluation_id == first.evaluation_id
    assert final.evaluation_id != first.evaluation_id
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(EvaluationRecord)) == 2
        assert session.scalar(select(func.count()).select_from(EvaluationSnapshotRecord)) == 2
        assert session.scalar(select(func.count()).select_from(MarketSeriesRecord)) >= 1


def test_csv_provider_rejects_duplicate_dates(tmp_path: Path) -> None:
    csv_path = tmp_path / "market.csv"
    csv_path.write_text(
        "date,open,high,low,close,adjusted_open,adjusted_close\n"
        "2026-01-13,100,103,99,102,100,102\n"
        "2026-01-13,100,104,98,103,100,103\n",
        encoding="utf-8",
    )
    provider = CsvMarketDataProvider(csv_path=csv_path)

    with pytest.raises(MarketDataUnavailable, match="重複"):
        provider.fetch(
            MarketDataRequest(
                symbol="^N225",
                currency="JPY",
                start=date(2026, 1, 13),
                end=date(2026, 4, 13),
            )
        )


def test_csv_provider_reads_valid_local_fixture(tmp_path: Path) -> None:
    csv_path = tmp_path / "market.csv"
    csv_path.write_text(
        "date,open,high,low,close,adjusted_open,adjusted_close\n"
        "2026-01-13,100,103,99,102,100,102\n"
        "2026-04-13,108,111,107,110,108,110\n",
        encoding="utf-8",
    )

    series = CsvMarketDataProvider(csv_path=csv_path).fetch(
        MarketDataRequest(
            symbol="^N225",
            currency="JPY",
            start=date(2026, 1, 13),
            end=date(2026, 4, 13),
        )
    )

    assert len(series.bars) == 2
    assert series.bars[0].adjusted_open == Decimal("100")
    assert series.bars[-1].adjusted_close == Decimal("110")


def test_fred_without_api_key_is_explicitly_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    with pytest.raises(MarketDataUnavailable, match="FRED_API_KEY"):
        FredMarketDataProvider().fetch(
            MarketDataRequest(
                symbol="DGS10",
                currency="PERCENT",
                start=date(2026, 1, 13),
                end=date(2026, 1, 20),
            )
        )
