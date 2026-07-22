"""Round7 coverage_audit contract for invalid, duplicate, and missing market data."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import select
from test_round5_functional import SymbolProvider, _bar, _series
from tests.unit.test_round4_acceptance import import_locked_component

from analyst_forecast.application.evaluation import (
    INSTRUMENT_AUDIT_KEYS,
    build_instrument_coverage_audit,
    evaluate_component,
)
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.domain.market import MarketBar, MarketDataUnavailable
from analyst_forecast.infrastructure.db.models import EvaluationRecord
from analyst_forecast.infrastructure.db.session import create_session_factory


def _invalid_bar(day: date) -> MarketBar:
    return MarketBar(
        date=day,
        open=Decimal("0"),
        high=Decimal("1"),
        low=Decimal("0"),
        close=Decimal("1"),
        adjusted_open=Decimal("0"),
        adjusted_close=Decimal("1"),
    )


def _load_audit(settings: AppSettings, component_id: str) -> dict[str, Any]:
    sf = create_session_factory(settings.database_file)
    with sf() as session:
        row = session.scalar(
            select(EvaluationRecord).where(EvaluationRecord.forecast_component_id == component_id)
        )
        assert row is not None
        assert row.coverage_audit is not None
        return dict(row.coverage_audit)


class TestR7CoverageAuditBadData:
    def test_r7_invalid_rows_full_audit_and_reason(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ) -> None:
        component_id = import_locked_component(
            settings, run_result, source_result, tmp_path, label="r7inv"
        )
        sf = create_session_factory(settings.database_file)
        with sf() as session:
            from analyst_forecast.infrastructure.db.models import (
                ForecastComponentRecord,
                TargetRecord,
            )

            comp = session.get(ForecastComponentRecord, component_id)
            target = session.get(TargetRecord, comp.target_id) if comp else None
            assert comp is not None and target is not None
            symbol = target.ticker or "N225"

        provider = SymbolProvider(
            {
                symbol: _series(
                    symbol,
                    (
                        _bar(date(2026, 1, 13), "100", "100"),
                        _invalid_bar(date(2026, 1, 14)),
                    ),
                )
            }
        )
        result = evaluate_component(
            settings,
            component_id=component_id,
            provider=provider,
            as_of=date(2026, 4, 13),
            run_id=run_result.run_id,
        )
        assert result.evaluation_status == "unevaluable"
        audit = _load_audit(settings, component_id)
        inst = audit["instruments"][0]
        assert set(INSTRUMENT_AUDIT_KEYS) <= set(inst)
        assert inst["symbol"] == symbol
        assert inst["invalid_row_count"] >= 1
        assert inst["adjustment_type"] == "split_dividend"
        assert audit["reason_code"] == "invalid_market_rows"
        assert audit["coverage_status"] == "invalid"
        assert audit["reason_code"] != "market_data_unavailable"

    def test_r7_duplicate_dates_audit_counts(self) -> None:
        day = date(2026, 1, 13)
        stats = build_instrument_coverage_audit(
            symbol="N225",
            currency="JPY",
            weight=1.0,
            requested_start=day,
            requested_end=date(2026, 1, 14),
            input_bars=[
                _bar(day, "100", "100"),
                _bar(day, "101", "101"),
                _bar(date(2026, 1, 14), "100", "110"),
            ],
            adjustment_type="split_dividend",
        )
        assert stats["duplicate_date_count"] == 1
        assert stats["input_row_count"] == 3
        assert stats["series_hash"]

    def test_r7_missing_in_range_full_audit(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ) -> None:
        component_id = import_locked_component(
            settings, run_result, source_result, tmp_path, label="r7miss"
        )
        sf = create_session_factory(settings.database_file)
        with sf() as session:
            from analyst_forecast.infrastructure.db.models import (
                ForecastComponentRecord,
                TargetRecord,
            )

            comp = session.get(ForecastComponentRecord, component_id)
            target = session.get(TargetRecord, comp.target_id) if comp else None
            assert comp is not None and target is not None
            symbol = target.ticker or "N225"

        class OutOfRangeProvider:
            name = "out-of-range-fixture"

            def fetch(self, request):  # type: ignore[no-untyped-def]
                return _series(
                    request.symbol,
                    (_bar(date(2020, 1, 1), "100", "100"),),
                )

        result = evaluate_component(
            settings,
            component_id=component_id,
            provider=OutOfRangeProvider(),
            as_of=date(2026, 4, 13),
            run_id=run_result.run_id,
        )
        assert result.evaluation_status == "unevaluable"
        audit = _load_audit(settings, component_id)
        inst = audit["instruments"][0]
        assert inst["symbol"] == symbol
        assert inst["input_row_count"] == 1
        assert inst["in_range_row_count"] == 0
        assert inst["unique_valid_date_count"] == 0
        assert inst["series_hash"] is None
        assert audit["reason_code"] == "missing_market_data_in_range"
        assert audit["reason_code"] != "market_data_unavailable"

    def test_r7_provider_missing_symbol_builds_audit(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ) -> None:
        component_id = import_locked_component(
            settings, run_result, source_result, tmp_path, label="r7prov"
        )

        class MissingSymbolProvider:
            name = "missing-symbol"

            def fetch(self, request):  # type: ignore[no-untyped-def]
                raise MarketDataUnavailable(f"missing {request.symbol}")

        result = evaluate_component(
            settings,
            component_id=component_id,
            provider=MissingSymbolProvider(),
            as_of=date(2026, 4, 13),
            run_id=run_result.run_id,
        )
        assert result.evaluation_status == "unevaluable"
        audit = _load_audit(settings, component_id)
        assert audit["instruments"]
        assert audit["reason_code"] == "missing_market_data_in_range"
