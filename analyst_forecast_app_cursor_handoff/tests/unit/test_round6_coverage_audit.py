"""Round6 coverage_audit DB contract tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy import select
from test_round5_functional import SymbolProvider, _bar, _series
from tests.unit.test_round4_acceptance import import_locked_component

from analyst_forecast.application.evaluation import (
    INSTRUMENT_AUDIT_KEYS,
    evaluate_component,
    series_hash_for_audit,
)
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.infrastructure.db.models import (
    EvaluationRecord,
    ForecastComponentRecord,
    TargetMappingRecord,
    TargetRecord,
)
from analyst_forecast.infrastructure.db.session import create_session_factory

REQUIRED_TOP = {
    "schema_version",
    "coverage_status",
    "reason_code",
    "requested_start_date",
    "requested_end_date",
    "effective_start_date",
    "effective_end_date",
    "evaluation_as_of",
    "method_version",
    "series_kind",
    "selected_start_date",
    "selected_end_date",
    "common_date_count",
    "intersection_rule",
    "mapping_hash",
    "instruments",
}


def _prepare_locked(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
    *,
    label: str,
    instruments: list[dict[str, object]] | None = None,
) -> tuple[str, str]:
    component_id = import_locked_component(
        settings, run_result, source_result, tmp_path, label=label
    )
    sf = create_session_factory(settings.database_file)
    with sf.begin() as session:
        comp = session.get(ForecastComponentRecord, component_id)
        assert comp is not None and comp.target_mapping_id
        mapping = session.get(TargetMappingRecord, comp.target_mapping_id)
        target = session.get(TargetRecord, comp.target_id)
        assert mapping is not None and target is not None
        symbol = target.ticker or "N225"
        currency = target.currency or "JPY"
        if instruments is None:
            instruments = [{"symbol": symbol, "currency": currency, "weight": 1.0}]
        mapping.evaluation_instruments = instruments
        mapping.weights = [float(item["weight"]) for item in instruments]
        return component_id, symbol


def _load_audit(settings: AppSettings, component_id: str) -> dict[str, Any]:
    sf = create_session_factory(settings.database_file)
    with sf() as session:
        row = session.scalar(
            select(EvaluationRecord).where(EvaluationRecord.forecast_component_id == component_id)
        )
        assert row is not None
        assert row.coverage_audit is not None
        return dict(row.coverage_audit)


def _assert_full_audit(audit: dict[str, Any]) -> None:
    assert set(audit) >= REQUIRED_TOP
    assert audit["schema_version"] == "1.0.0"
    assert isinstance(audit["instruments"], list) and audit["instruments"]
    for inst in audit["instruments"]:
        assert set(INSTRUMENT_AUDIT_KEYS) <= set(inst)
        assert inst["symbol"]
        assert inst["currency"]


class TestR6CoverageAudit:
    def test_r6_024_single_one_day(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        component_id, symbol = _prepare_locked(
            settings, run_result, source_result, tmp_path, label="r6cov1"
        )
        provider = SymbolProvider(
            {symbol: _series(symbol, (_bar(date(2026, 1, 13), "100", "102"),))}
        )
        result = evaluate_component(
            settings,
            component_id=component_id,
            provider=provider,
            as_of=date(2026, 4, 13),
            run_id=run_result.run_id,
        )
        assert result.evaluation_status == "unevaluable"
        assert result.direction_result is None
        assert result.actual_return is None
        audit = _load_audit(settings, component_id)
        _assert_full_audit(audit)
        assert audit["coverage_status"] == "insufficient"
        assert audit["reason_code"] == "insufficient_trading_dates"
        assert audit["series_kind"] == "single"
        assert audit["instruments"][0]["unique_valid_date_count"] == 1
        assert audit["instruments"][0]["series_hash"]

    def test_r6_025_single_two_days(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        component_id, symbol = _prepare_locked(
            settings, run_result, source_result, tmp_path, label="r6cov2"
        )
        provider = SymbolProvider(
            {
                symbol: _series(
                    symbol,
                    (
                        _bar(date(2026, 1, 13), "100", "100"),
                        _bar(date(2026, 1, 14), "100", "110"),
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
        assert result.direction_result == "hit"
        audit = _load_audit(settings, component_id)
        _assert_full_audit(audit)
        assert audit["coverage_status"] == "sufficient"
        assert audit["selected_start_date"] < audit["selected_end_date"]
        assert audit["instruments"][0]["symbol"] == symbol
        assert audit["instruments"][0]["currency"]

    def test_r6_026_basket_common_one(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        instruments = [
            {"symbol": "AAA", "currency": "JPY", "weight": 0.5},
            {"symbol": "BBB", "currency": "JPY", "weight": 0.5},
        ]
        component_id, _ = _prepare_locked(
            settings,
            run_result,
            source_result,
            tmp_path,
            label="r6covb",
            instruments=instruments,
        )
        provider = SymbolProvider(
            {
                "AAA": _series(
                    "AAA",
                    (
                        _bar(date(2026, 1, 13), "10", "11"),
                        _bar(date(2026, 1, 14), "11", "12"),
                    ),
                ),
                "BBB": _series(
                    "BBB",
                    (_bar(date(2026, 1, 14), "20", "21"),),
                ),
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
        _assert_full_audit(audit)
        assert audit["reason_code"] == "insufficient_common_dates"
        assert audit["common_date_count"] == 1
        assert len(audit["instruments"]) == 2

    def test_r6_021_out_of_range_counts(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        component_id, symbol = _prepare_locked(
            settings, run_result, source_result, tmp_path, label="r6covr"
        )
        sf = create_session_factory(settings.database_file)
        with sf.begin() as session:
            comp = session.get(ForecastComponentRecord, component_id)
            assert comp is not None
            # Narrow window so outer bars are out of range
            comp.normalized_start = date(2026, 1, 14)
            comp.normalized_end = date(2026, 1, 15)
        provider = SymbolProvider(
            {
                symbol: _series(
                    symbol,
                    (
                        _bar(date(2026, 1, 13), "90", "91"),
                        _bar(date(2026, 1, 14), "100", "100"),
                        _bar(date(2026, 1, 15), "100", "110"),
                        _bar(date(2026, 1, 16), "110", "111"),
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
        assert result.direction_result == "hit"
        audit = _load_audit(settings, component_id)
        inst = audit["instruments"][0]
        assert inst["input_first_date"] == "2026-01-13"
        assert inst["input_last_date"] == "2026-01-16"
        assert inst["in_range_row_count"] == 2
        assert inst["dropped_out_of_range_count"] == 2

    def test_r6_023_hash_order_invariant_and_value_sensitive(self):
        bars_a = [
            _bar(date(2026, 1, 14), "100", "101"),
            _bar(date(2026, 1, 13), "100", "100"),
        ]
        bars_b = list(reversed(bars_a))
        h1 = series_hash_for_audit(symbol="N225", currency="JPY", bars=bars_a)
        h2 = series_hash_for_audit(symbol="N225", currency="JPY", bars=bars_b)
        assert h1 == h2
        bars_c = [
            _bar(date(2026, 1, 13), "100", "100"),
            _bar(date(2026, 1, 14), "100", "102"),
        ]
        h3 = series_hash_for_audit(symbol="N225", currency="JPY", bars=bars_c)
        assert h3 != h1
