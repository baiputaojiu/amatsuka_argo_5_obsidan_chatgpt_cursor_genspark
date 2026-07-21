from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from analyst_forecast.application.settings import AppSettings
from analyst_forecast.domain.market import (
    MarketBar,
    MarketDataProvider,
    MarketDataRequest,
    MarketDataUnavailable,
    MarketSeries,
    ProviderError,
)
from analyst_forecast.infrastructure.db.ids import next_id
from analyst_forecast.infrastructure.db.models import (
    AiArtifactRecord,
    AiImportRecord,
    EvaluationRecord,
    EvaluationSnapshotRecord,
    ForecastComponentRecord,
    ForecastIssuanceRecord,
    MarketSeriesRecord,
    TargetMappingRecord,
    TargetRecord,
)
from analyst_forecast.infrastructure.db.session import create_session_factory

# direction-v1.0.0: MFE/MAEは常に上昇方向の符号規約
#   favorable = (period_high - start) / start
#   adverse = (period_low - start) / start
# direction-v2.0.0: 予想方向に対応
#   up: v1と同じ
#   down: favorable = (start - period_low) / start
#         adverse = (period_high - start) / start
EVALUATION_METHOD_VERSION = "direction-v2.0.0"
LEGACY_EVALUATION_METHOD_VERSION = "direction-v1.0.0"


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    evaluation_id: str
    evaluation_status: str
    direction_result: str | None
    start_price: Decimal | None
    end_price: Decimal | None
    actual_return: Decimal | None
    max_favorable_excursion: Decimal | None
    max_adverse_excursion: Decimal | None
    unevaluable_reason: str | None = None
    method_version: str = EVALUATION_METHOD_VERSION
    provider_error_code: str | None = None


def evaluate_component(
    settings: AppSettings,
    *,
    component_id: str,
    provider: MarketDataProvider,
    as_of: date,
    run_id: str | None = None,
    method_version: str = EVALUATION_METHOD_VERSION,
) -> EvaluationResult:
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        component = session.get(ForecastComponentRecord, component_id)
        if component is None:
            raise ValueError(
                f"構成予想IDが存在しません: {component_id}\n"
                "次の操作: 状態表示に記載されたIDを使用してください。"
            )
        resolved_run_id = _resolve_run_id(session, component)
        if run_id is not None and resolved_run_id != run_id:
            raise ValueError(
                f"構成予想 {component_id} は案件 {run_id} に属していません。\n"
                "次の操作: status で正しいcomponent IDを確認してください。"
            )
        existing = session.scalar(
            select(EvaluationRecord).where(
                EvaluationRecord.forecast_component_id == component_id,
                EvaluationRecord.target_mapping_id == component.target_mapping_id,
                EvaluationRecord.evaluation_method_version == method_version,
                EvaluationRecord.evaluation_as_of == as_of,
            )
        )
        if existing is not None:
            return _to_result(existing)
        if component.target_mapping_id is None or component.target_id is None:
            raise ValueError(
                "予測対象の独立レビューとmapping固定が完了していません。"
                "次の操作: P11とP12、必要時P13を先に取り込んでください。"
            )
        mapping = session.get(TargetMappingRecord, component.target_mapping_id)
        target = session.get(TargetRecord, component.target_id)
        issuance = session.get(ForecastIssuanceRecord, component.forecast_issuance_id)
        if mapping is None or target is None or issuance is None:
            raise RuntimeError("予想、対象、マッピングのDB参照が破損しています")
        if (
            component.target_resolution_status
            in {"pending", "awaiting_review", "awaiting_adjudication", "proposed", "review_pending"}
            or mapping.locked_at is None
        ) and mapping.mapping_status != "unresolvable":
            raise ValueError(
                "予測対象の独立レビューとmapping固定が完了していません。"
                "次の操作: P11とP12、必要時P13を先に取り込んでください。"
            )
        mapping_id = mapping.target_mapping_id
        normalized_start = component.normalized_start
        normalized_end = component.normalized_end
        expected_direction = component.direction
        mapping_status = mapping.mapping_status
        mapping_reason = mapping.unevaluable_reason
        ticker = target.ticker
        currency = target.currency
        raw_instruments = list(mapping.evaluation_instruments or [])
        raw_weights = list(mapping.weights or []) if mapping.weights is not None else []

    if mapping_status not in {"verified", "corrected"}:
        return _store_without_values(
            settings,
            component_id=component_id,
            mapping_id=mapping_id,
            as_of=as_of,
            status="unevaluable",
            reason=mapping_reason or "固定済み・検証済みの対象マッピングがありません",
            run_id=resolved_run_id,
            method_version=method_version,
        )
    instruments = _normalize_instruments(
        raw_instruments,
        raw_weights,
        ticker=ticker,
        currency=currency,
    )
    if not instruments:
        return _store_without_values(
            settings,
            component_id=component_id,
            mapping_id=mapping_id,
            as_of=as_of,
            status="unevaluable",
            reason="mappingに市場symbolまたは通貨がありません",
            run_id=resolved_run_id,
            method_version=method_version,
        )
    instrument_error = _validate_instruments(instruments)
    if instrument_error is not None:
        return _store_without_values(
            settings,
            component_id=component_id,
            mapping_id=mapping_id,
            as_of=as_of,
            status="unevaluable",
            reason=instrument_error,
            run_id=resolved_run_id,
            method_version=method_version,
        )
    if normalized_start is None or normalized_end is None:
        return _store_without_values(
            settings,
            component_id=component_id,
            mapping_id=mapping_id,
            as_of=as_of,
            status="unevaluable",
            reason="最小方向評価に必要な開始日または終了日がありません",
            run_id=resolved_run_id,
            method_version=method_version,
        )
    if as_of < normalized_start:
        return _store_without_values(
            settings,
            component_id=component_id,
            mapping_id=mapping_id,
            as_of=as_of,
            status="not_started",
            reason=None,
            run_id=resolved_run_id,
            method_version=method_version,
        )

    effective_end = min(as_of, normalized_end)
    cache_hit = False
    attempt_count = 1
    provider_error_code: str | None = None
    provider_error_message: str | None = None
    retryable: str | None = None
    series_kind = "raw"
    series_identity: str | None = None
    mapping_hash: str | None = None
    input_series_hashes: list[str] | None = None
    basket_weights: list[float] | None = None
    common_date_rule: str | None = None
    store_provider: str
    store_symbol: str
    store_currency: str
    store_adjustment_type: str
    store_frequency: str
    store_retrieved_at: datetime
    raw_series_payloads: list[dict[str, object]] = []
    try:
        if len(instruments) == 1:
            instrument = instruments[0]
            request = MarketDataRequest(
                symbol=str(instrument["symbol"]),
                currency=str(instrument["currency"]),
                start=normalized_start,
                end=effective_end,
            )
            cached = _load_cached_series(settings, request, provider_name=provider.name)
            if cached is not None:
                series = cached
                cache_hit = True
            else:
                series = provider.fetch(request)
                _validate_provider_series(series, request)
            bars = tuple(
                bar for bar in series.bars if normalized_start <= bar.date <= effective_end
            )
            if not bars:
                raise MarketDataUnavailable("評価期間内の市場データが0件です")
            _validate_bars(bars)
            start_price = bars[0].adjusted_open
            end_price = bars[-1].adjusted_close
            if start_price <= 0:
                raise MarketDataUnavailable("開始値が0以下のため変化率を計算できません")
            series_kind = "raw"
            series_identity = request.symbol
            store_provider = series.provider
            store_symbol = request.symbol
            store_currency = series.currency
            store_adjustment_type = series.adjustment_type
            store_frequency = series.frequency
            store_retrieved_at = series.retrieved_at
        else:
            mapping_hash = _basket_mapping_hash(instruments)
            series_identity = f"BASKET:{mapping_hash}"
            series_kind = "basket"
            common_date_rule = "intersection_all_instruments_v1"
            basket_weights = [float(str(item["weight"])) for item in instruments]
            series_by_symbol: dict[str, MarketSeries] = {}
            raw_cache_hits: dict[str, bool] = {}
            for instrument in instruments:
                request = MarketDataRequest(
                    symbol=str(instrument["symbol"]),
                    currency=str(instrument["currency"]),
                    start=normalized_start,
                    end=effective_end,
                )
                cached = _load_cached_series(settings, request, provider_name=provider.name)
                if cached is not None:
                    series_by_symbol[str(instrument["symbol"])] = cached
                    raw_cache_hits[str(instrument["symbol"])] = True
                    cache_hit = True
                else:
                    fetched = provider.fetch(request)
                    _validate_provider_series(fetched, request)
                    series_by_symbol[str(instrument["symbol"])] = fetched
                    raw_cache_hits[str(instrument["symbol"])] = False
            common_dates = _common_bar_dates(series_by_symbol, normalized_start, effective_end)
            if len(common_dates) < 1:
                raise MarketDataUnavailable(
                    "共通取引日が不足しているためバスケットを評価できません"
                )
            bars = _build_basket_bars(series_by_symbol, instruments, common_dates)
            if not bars:
                raise MarketDataUnavailable("評価期間内の市場データが0件です")
            _validate_bars(bars)
            start_price = bars[0].adjusted_open
            end_price = bars[-1].adjusted_close
            if start_price <= 0:
                raise MarketDataUnavailable("開始値が0以下のため変化率を計算できません")
            first_series = series_by_symbol[str(instruments[0]["symbol"])]
            store_provider = provider.name
            store_symbol = series_identity
            store_currency = str(instruments[0]["currency"])
            store_adjustment_type = first_series.adjustment_type
            store_frequency = first_series.frequency
            store_retrieved_at = datetime.now(UTC)
            input_series_hashes = []
            for instrument in instruments:
                symbol = str(instrument["symbol"])
                raw_series = series_by_symbol[symbol]
                raw_bars = tuple(
                    bar for bar in raw_series.bars if normalized_start <= bar.date <= effective_end
                )
                raw_hash, raw_csv = _serialize_bars(
                    provider=raw_series.provider,
                    symbol=symbol,
                    currency=str(instrument["currency"]),
                    adjustment_type=raw_series.adjustment_type,
                    bars=raw_bars,
                )
                input_series_hashes.append(raw_hash)
                raw_series_payloads.append(
                    {
                        "provider": raw_series.provider,
                        "symbol": symbol,
                        "currency": str(instrument["currency"]),
                        "adjustment_type": raw_series.adjustment_type,
                        "frequency": raw_series.frequency,
                        "retrieved_at": raw_series.retrieved_at,
                        "start_date": raw_bars[0].date if raw_bars else bars[0].date,
                        "end_date": raw_bars[-1].date if raw_bars else bars[-1].date,
                        "data_hash": raw_hash,
                        "csv_content": raw_csv,
                        "series_kind": "raw",
                        "series_identity": symbol,
                        "mapping_hash": None,
                        "input_series_hashes": None,
                        "basket_weights": None,
                        "common_date_rule": None,
                        "cache_hit": "yes" if raw_cache_hits.get(symbol) else "no",
                        "attempt_count": attempt_count,
                    }
                )
    except ProviderError as error:
        return _store_without_values(
            settings,
            component_id=component_id,
            mapping_id=mapping_id,
            as_of=as_of,
            status="unevaluable",
            reason=error.guidance or str(error),
            run_id=resolved_run_id,
            method_version=method_version,
            provider_error_code=error.code,
            provider_error_message=str(error),
            retryable="yes" if error.retryable else "no",
            attempt_count=error.attempt_count,
        )
    except MarketDataUnavailable as error:
        return _store_without_values(
            settings,
            component_id=component_id,
            mapping_id=mapping_id,
            as_of=as_of,
            status="unevaluable",
            reason=str(error),
            run_id=resolved_run_id,
            method_version=method_version,
            provider_error_code=provider_error_code or "no_data",
            provider_error_message=provider_error_message or str(error),
            retryable=retryable or "no",
            attempt_count=attempt_count,
        )

    actual_return = (end_price - start_price) / start_price
    period_high = max(bar.high for bar in bars)
    period_low = min(bar.low for bar in bars)
    max_favorable, max_adverse = _direction_excursions(
        expected_direction,
        start_price=start_price,
        period_high=period_high,
        period_low=period_low,
        method_version=method_version,
    )
    actual_direction = (
        "up" if end_price > start_price else "down" if end_price < start_price else "flat"
    )
    direction_result = "hit" if actual_direction == expected_direction else "miss"
    evaluation_status = (
        f"expired_{direction_result}"
        if as_of >= normalized_end
        else "active_on_track"
        if direction_result == "hit"
        else "active_off_track"
    )

    data_hash, csv_content = _serialize_bars(
        provider=store_provider,
        symbol=store_symbol,
        currency=store_currency,
        adjustment_type=store_adjustment_type,
        bars=bars,
    )
    evaluation_payload = {
        "provider": store_provider,
        "symbol": store_symbol,
        "currency": store_currency,
        "adjustment_type": store_adjustment_type,
        "frequency": store_frequency,
        "retrieved_at": store_retrieved_at,
        "start_date": bars[0].date,
        "end_date": bars[-1].date,
        "data_hash": data_hash,
        "csv_content": csv_content,
        "series_kind": series_kind,
        "series_identity": series_identity,
        "mapping_hash": mapping_hash,
        "input_series_hashes": input_series_hashes,
        "basket_weights": basket_weights,
        "common_date_rule": common_date_rule,
        "cache_hit": "yes" if cache_hit else "no",
        "attempt_count": attempt_count,
    }

    with session_factory.begin() as session:
        for payload in raw_series_payloads:
            _upsert_market_series(session, settings, payload)
        existing_series = _upsert_market_series(session, settings, evaluation_payload)
        evaluation = EvaluationRecord(
            evaluation_id=next_id(session, "EVAL-", width=6, sequence_key="EVALUATION"),
            forecast_component_id=component_id,
            target_mapping_id=mapping_id,
            market_series_id=existing_series.market_series_id,
            evaluation_method_version=method_version,
            evaluation_as_of=as_of,
            start_price=start_price,
            end_price=end_price,
            current_price=end_price,
            period_high=period_high,
            period_low=period_low,
            actual_return=actual_return,
            direction_result=direction_result,
            evaluation_status=evaluation_status,
            max_favorable_excursion=max_favorable,
            max_adverse_excursion=max_adverse,
            cache_hit="yes" if cache_hit else "no",
            attempt_count=attempt_count,
        )
        session.add(evaluation)
        session.flush()
        session.add(
            EvaluationSnapshotRecord(
                evaluation_snapshot_id=next_id(
                    session,
                    "EVS-",
                    width=6,
                    sequence_key="EVALUATION_SNAPSHOT",
                ),
                evaluation_id=evaluation.evaluation_id,
                snapshot_at=as_of,
                status=evaluation_status,
                interim_return=actual_return,
                max_favorable_excursion=max_favorable,
                max_adverse_excursion=max_adverse,
                notes=f"最小方向評価 {method_version}",
            )
        )
        result = _to_result(evaluation)

    from analyst_forecast.application.workflow import refresh_workflow

    refresh_workflow(settings, resolved_run_id)
    return result


def _validate_instruments(instruments: list[dict[str, object]]) -> str | None:
    symbols = [str(item["symbol"]) for item in instruments]
    if len(symbols) != len(set(symbols)):
        return "同一symbolの重複instrumentは禁止です"
    currencies = {str(item["currency"]) for item in instruments}
    if len(currencies) > 1:
        return "unevaluable_mixed_currency"
    weights = [float(str(item["weight"])) for item in instruments]
    if any(weight <= 0 for weight in weights):
        return "weightは正の値である必要があります"
    weight_total = sum(weights)
    if abs(weight_total - 1.0) > 1e-6:
        return "instrument weight合計が1ではありません"
    return None


def _basket_mapping_hash(instruments: list[dict[str, object]]) -> str:
    payload = {
        "instruments": sorted(
            [
                {
                    "symbol": str(item["symbol"]),
                    "currency": str(item["currency"]),
                    "weight": float(str(item["weight"])),
                }
                for item in instruments
            ],
            key=lambda item: (item["symbol"], item["currency"]),
        ),
        "common_date_rule": "intersection_all_instruments_v1",
        "method": "weighted_adjusted_return_index_v1",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _validate_provider_series(series: MarketSeries, request: MarketDataRequest) -> None:
    if series.symbol != request.symbol:
        raise MarketDataUnavailable(
            f"providerのsymbolがrequestと不一致です: {series.symbol} != {request.symbol}"
        )
    if series.currency != request.currency:
        raise MarketDataUnavailable(
            f"providerのcurrencyがrequestと不一致です: {series.currency} != {request.currency}"
        )


def _normalize_instruments(
    evaluation_instruments: list[object] | None,
    weights: list[float] | None,
    *,
    ticker: str | None,
    currency: str | None,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    if evaluation_instruments:
        for index, raw in enumerate(evaluation_instruments):
            if isinstance(raw, dict):
                symbol = str(raw.get("symbol") or "")
                item_currency = str(raw.get("currency") or currency or "")
                weight = float(raw.get("weight") or (weights[index] if weights else 0))
                exchange = raw.get("exchange")
            else:
                symbol = str(raw)
                item_currency = str(currency or "")
                weight = float(weights[index]) if weights and index < len(weights) else 0.0
                exchange = None
            if not symbol or not item_currency:
                continue
            items.append(
                {
                    "symbol": symbol,
                    "currency": item_currency,
                    "weight": weight,
                    "exchange": exchange,
                }
            )
    elif ticker and currency:
        items.append({"symbol": ticker, "currency": currency, "weight": 1.0, "exchange": None})
    return items


def _common_bar_dates(
    series_by_symbol: dict[str, MarketSeries],
    start: date,
    end: date,
) -> list[date]:
    date_sets: list[set[date]] = []
    for series in series_by_symbol.values():
        date_sets.append({bar.date for bar in series.bars if start <= bar.date <= end})
    if not date_sets:
        return []
    common = set.intersection(*date_sets)
    return sorted(common)


def _build_basket_bars(
    series_by_symbol: dict[str, MarketSeries],
    instruments: list[dict[str, object]],
    common_dates: list[date],
) -> tuple[MarketBar, ...]:
    bars_by_symbol: dict[str, dict[date, MarketBar]] = {}
    for symbol, series in series_by_symbol.items():
        bars_by_symbol[symbol] = {bar.date: bar for bar in series.bars}
    # 基準値1のweight付き指数
    index_level = Decimal("1")
    previous_closes: dict[str, Decimal] = {}
    for instrument in instruments:
        symbol = str(instrument["symbol"])
        first = bars_by_symbol[symbol][common_dates[0]]
        previous_closes[symbol] = first.adjusted_open
    built: list[MarketBar] = []
    for day in common_dates:
        day_return = Decimal("0")
        day_high_ret = Decimal("0")
        day_low_ret = Decimal("0")
        for instrument in instruments:
            symbol = str(instrument["symbol"])
            weight = Decimal(str(instrument["weight"]))
            bar = bars_by_symbol[symbol][day]
            prev = previous_closes[symbol]
            if prev <= 0:
                raise MarketDataUnavailable("バスケット構成銘柄の基準値が不正です")
            close_ret = (bar.adjusted_close - prev) / prev
            high_ret = (bar.high - prev) / prev
            low_ret = (bar.low - prev) / prev
            day_return += weight * close_ret
            day_high_ret += weight * high_ret
            day_low_ret += weight * low_ret
            previous_closes[symbol] = bar.adjusted_close
        index_open = index_level
        index_close = index_level * (Decimal("1") + day_return)
        index_high = index_level * (Decimal("1") + day_high_ret)
        index_low = index_level * (Decimal("1") + day_low_ret)
        built.append(
            MarketBar(
                date=day,
                open=index_open,
                high=max(index_open, index_high, index_close, index_low),
                low=min(index_open, index_high, index_close, index_low),
                close=index_close,
                adjusted_open=index_open,
                adjusted_close=index_close,
            )
        )
        index_level = index_close
    return tuple(built)


def _direction_excursions(
    expected_direction: str,
    *,
    start_price: Decimal,
    period_high: Decimal,
    period_low: Decimal,
    method_version: str,
) -> tuple[Decimal, Decimal]:
    if method_version == LEGACY_EVALUATION_METHOD_VERSION:
        return (
            (period_high - start_price) / start_price,
            (period_low - start_price) / start_price,
        )
    if expected_direction == "down":
        return (
            (start_price - period_low) / start_price,
            (period_high - start_price) / start_price,
        )
    return (
        (period_high - start_price) / start_price,
        (period_low - start_price) / start_price,
    )


def _resolve_run_id(session: Session, component: ForecastComponentRecord) -> str:
    issuance = session.get(ForecastIssuanceRecord, component.forecast_issuance_id)
    if issuance is None:
        raise RuntimeError("予想表明のDB参照が破損しています")
    if issuance.ai_artifact_id is not None:
        ai_artifact = session.get(AiArtifactRecord, issuance.ai_artifact_id)
        if ai_artifact is None:
            raise RuntimeError("AI成果物のDB参照が破損しています")
        return str(ai_artifact.run_id)
    if issuance.ai_import_id is not None:
        ai_import = session.get(AiImportRecord, issuance.ai_import_id)
        if ai_import is None:
            raise RuntimeError("AI取込み記録のDB参照が破損しています")
        return str(ai_import.run_id)
    raise RuntimeError("予想表明にAI取込み元がありません")


def _store_without_values(
    settings: AppSettings,
    *,
    component_id: str,
    mapping_id: str,
    as_of: date,
    status: str,
    reason: str | None,
    run_id: str,
    method_version: str = EVALUATION_METHOD_VERSION,
    provider_error_code: str | None = None,
    provider_error_message: str | None = None,
    retryable: str | None = None,
    attempt_count: int | None = None,
) -> EvaluationResult:
    session_factory = create_session_factory(settings.database_file)
    with session_factory.begin() as session:
        existing = session.scalar(
            select(EvaluationRecord).where(
                EvaluationRecord.forecast_component_id == component_id,
                EvaluationRecord.target_mapping_id == mapping_id,
                EvaluationRecord.evaluation_method_version == method_version,
                EvaluationRecord.evaluation_as_of == as_of,
            )
        )
        if existing is not None:
            return _to_result(existing)
        evaluation = EvaluationRecord(
            evaluation_id=next_id(session, "EVAL-", width=6, sequence_key="EVALUATION"),
            forecast_component_id=component_id,
            target_mapping_id=mapping_id,
            evaluation_method_version=method_version,
            evaluation_as_of=as_of,
            evaluation_status=status,
            unevaluable_reason=reason,
            provider_error_code=provider_error_code,
            provider_error_message=provider_error_message,
            retryable=retryable,
            attempt_count=attempt_count,
        )
        session.add(evaluation)
        session.flush()
        session.add(
            EvaluationSnapshotRecord(
                evaluation_snapshot_id=next_id(
                    session,
                    "EVS-",
                    width=6,
                    sequence_key="EVALUATION_SNAPSHOT",
                ),
                evaluation_id=evaluation.evaluation_id,
                snapshot_at=as_of,
                status=status,
                notes=reason,
            )
        )
        # 親issuance状態は単一component評価で上書きしない
        result = _to_result(evaluation)

    from analyst_forecast.application.workflow import refresh_workflow

    refresh_workflow(settings, run_id)
    return result


def _validate_bars(bars: tuple[MarketBar, ...]) -> None:
    for bar in bars:
        prices = (
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.adjusted_open,
            bar.adjusted_close,
        )
        if any(price <= 0 for price in prices):
            raise MarketDataUnavailable("0以下の不正価格が含まれています")
        if bar.high < bar.low:
            raise MarketDataUnavailable("高値と安値の関係が不正です")


def _load_cached_series(
    settings: AppSettings,
    request: MarketDataRequest,
    *,
    provider_name: str,
) -> MarketSeries | None:
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        record = session.scalar(
            select(MarketSeriesRecord)
            .where(
                MarketSeriesRecord.provider == provider_name,
                MarketSeriesRecord.symbol == request.symbol,
                MarketSeriesRecord.currency == request.currency,
                MarketSeriesRecord.start_date <= request.start,
                MarketSeriesRecord.end_date >= request.end,
                MarketSeriesRecord.quality_status == "valid",
                or_(
                    MarketSeriesRecord.series_kind == "raw",
                    MarketSeriesRecord.series_kind.is_(None),
                ),
            )
            .order_by(MarketSeriesRecord.retrieved_at.desc())
        )
        if record is None:
            return None
        cache_path = settings.vault_root / Path(record.raw_cache_path)
        if not cache_path.is_file():
            return None
        from analyst_forecast.infrastructure.market.csv_provider import CsvMarketDataProvider

        return CsvMarketDataProvider(csv_path=cache_path).fetch(request)


def _upsert_market_series(
    session: Session,
    settings: AppSettings,
    payload: dict[str, object],
) -> MarketSeriesRecord:
    from typing import cast

    data_hash = str(payload["data_hash"])
    existing = session.scalar(
        select(MarketSeriesRecord).where(MarketSeriesRecord.data_hash == data_hash)
    )
    if existing is not None:
        return existing
    cache_path = _cache_market_series(
        settings,
        provider=str(payload["provider"]),
        symbol=str(payload["symbol"]),
        data_hash=data_hash,
        content=str(payload["csv_content"]),
    )
    raw_hashes = payload.get("input_series_hashes")
    raw_weights = payload.get("basket_weights")
    record = MarketSeriesRecord(
        market_series_id=next_id(session, "MKS-", width=6, sequence_key="MARKET_SERIES"),
        provider=str(payload["provider"]),
        symbol=str(payload["symbol"]),
        currency=str(payload["currency"]),
        adjustment_type=str(payload["adjustment_type"]),
        frequency=str(payload["frequency"]),
        start_date=cast(date, payload["start_date"]),
        end_date=cast(date, payload["end_date"]),
        retrieved_at=cast(datetime, payload["retrieved_at"]),
        raw_cache_path=cache_path.relative_to(settings.vault_root).as_posix(),
        data_hash=data_hash,
        quality_status="valid",
        series_kind=str(payload.get("series_kind") or "raw"),
        series_identity=(
            str(payload["series_identity"]) if payload.get("series_identity") is not None else None
        ),
        mapping_hash=(
            str(payload["mapping_hash"]) if payload.get("mapping_hash") is not None else None
        ),
        input_series_hashes=(
            [str(item) for item in cast(list[object], raw_hashes)]
            if isinstance(raw_hashes, list)
            else None
        ),
        basket_weights=(
            [float(cast(float | int | str, item)) for item in cast(list[object], raw_weights)]
            if isinstance(raw_weights, list)
            else None
        ),
        common_date_rule=(
            str(payload["common_date_rule"])
            if payload.get("common_date_rule") is not None
            else None
        ),
        cache_hit=str(payload.get("cache_hit") or "no"),
        attempt_count=(
            int(str(payload["attempt_count"])) if payload.get("attempt_count") is not None else None
        ),
    )
    session.add(record)
    session.flush()
    return record


def _serialize_bars(
    *,
    provider: str,
    symbol: str,
    currency: str,
    adjustment_type: str,
    bars: tuple[MarketBar, ...],
) -> tuple[str, str]:
    metadata = {
        "provider": provider,
        "symbol": symbol,
        "currency": currency,
        "adjustment_type": adjustment_type,
    }
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "adjusted_open",
            "adjusted_close",
        ]
    )
    for bar in bars:
        writer.writerow(
            [
                bar.date.isoformat(),
                str(bar.open),
                str(bar.high),
                str(bar.low),
                str(bar.close),
                str(bar.adjusted_open),
                str(bar.adjusted_close),
            ]
        )
    content = stream.getvalue()
    digest_input = json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n" + content
    return hashlib.sha256(digest_input.encode("utf-8")).hexdigest(), content


def _cache_market_series(
    settings: AppSettings,
    *,
    provider: str,
    symbol: str,
    data_hash: str,
    content: str,
) -> Path:
    safe_symbol = re.sub(r"[^A-Za-z0-9_.-]", "_", symbol)[:80] or "series"
    cache_path = (
        settings.vault_root
        / "_system"
        / "market_cache"
        / provider
        / f"{safe_symbol}__{data_hash[:12]}.csv"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        with cache_path.open("x", encoding="utf-8", newline="") as output:
            output.write(content)
    return cache_path


def _to_result(record: EvaluationRecord) -> EvaluationResult:
    return EvaluationResult(
        evaluation_id=record.evaluation_id,
        evaluation_status=record.evaluation_status,
        direction_result=record.direction_result,
        start_price=record.start_price,
        end_price=record.end_price,
        actual_return=record.actual_return,
        max_favorable_excursion=record.max_favorable_excursion,
        max_adverse_excursion=record.max_adverse_excursion,
        unevaluable_reason=record.unevaluable_reason,
        method_version=record.evaluation_method_version,
        provider_error_code=record.provider_error_code,
    )
