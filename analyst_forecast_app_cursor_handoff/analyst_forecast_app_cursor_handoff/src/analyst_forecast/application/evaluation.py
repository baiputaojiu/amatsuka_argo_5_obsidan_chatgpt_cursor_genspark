from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

from sqlalchemy import select
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
        mapping_id = mapping.target_mapping_id
        normalized_start = component.normalized_start
        normalized_end = component.normalized_end
        expected_direction = component.direction
        mapping_status = mapping.mapping_status
        mapping_reason = mapping.unevaluable_reason
        ticker = target.ticker
        currency = target.currency

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
    if ticker is None or currency is None:
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
    request = MarketDataRequest(
        symbol=ticker,
        currency=currency,
        start=normalized_start,
        end=effective_end,
    )
    cache_hit = False
    attempt_count = 1
    provider_error_code: str | None = None
    provider_error_message: str | None = None
    retryable: str | None = None
    try:
        cached = _load_cached_series(settings, request, provider_name=provider.name)
        if cached is not None:
            series = cached
            cache_hit = True
        else:
            series = provider.fetch(request)
        bars = tuple(bar for bar in series.bars if normalized_start <= bar.date <= effective_end)
        if not bars:
            raise MarketDataUnavailable("評価期間内の市場データが0件です")
        _validate_bars(bars)
        start_price = bars[0].adjusted_open
        end_price = bars[-1].adjusted_close
        if start_price <= 0:
            raise MarketDataUnavailable("開始値が0以下のため変化率を計算できません")
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
        provider=series.provider,
        symbol=series.symbol,
        currency=series.currency,
        adjustment_type=series.adjustment_type,
        bars=bars,
    )
    cache_path = _cache_market_series(
        settings,
        provider=series.provider,
        symbol=series.symbol,
        data_hash=data_hash,
        content=csv_content,
    )

    with session_factory.begin() as session:
        existing_series = session.scalar(
            select(MarketSeriesRecord).where(MarketSeriesRecord.data_hash == data_hash)
        )
        if existing_series is None:
            existing_series = MarketSeriesRecord(
                market_series_id=next_id(session, "MKS-", width=6, sequence_key="MARKET_SERIES"),
                provider=series.provider,
                symbol=series.symbol,
                currency=series.currency,
                adjustment_type=series.adjustment_type,
                frequency=series.frequency,
                start_date=bars[0].date,
                end_date=bars[-1].date,
                retrieved_at=series.retrieved_at,
                raw_cache_path=cache_path.relative_to(settings.vault_root).as_posix(),
                data_hash=data_hash,
                quality_status="valid",
                cache_hit="yes" if cache_hit else "no",
                attempt_count=attempt_count,
            )
            session.add(existing_series)
            session.flush()
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
