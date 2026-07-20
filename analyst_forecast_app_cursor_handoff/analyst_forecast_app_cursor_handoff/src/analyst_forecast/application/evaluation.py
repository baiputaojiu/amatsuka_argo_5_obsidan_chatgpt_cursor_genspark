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

from analyst_forecast.application.settings import AppSettings
from analyst_forecast.domain.market import (
    MarketBar,
    MarketDataProvider,
    MarketDataRequest,
    MarketDataUnavailable,
)
from analyst_forecast.infrastructure.db.ids import next_id
from analyst_forecast.infrastructure.db.models import (
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

EVALUATION_METHOD_VERSION = "direction-v1.0.0"


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


def evaluate_component(
    settings: AppSettings,
    *,
    component_id: str,
    provider: MarketDataProvider,
    as_of: date,
) -> EvaluationResult:
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        component = session.get(ForecastComponentRecord, component_id)
        if component is None:
            raise ValueError(
                f"構成予想IDが存在しません: {component_id}\n"
                "次の操作: 状態表示に記載されたIDを使用してください。"
            )
        existing = session.scalar(
            select(EvaluationRecord).where(
                EvaluationRecord.forecast_component_id == component_id,
                EvaluationRecord.target_mapping_id == component.target_mapping_id,
                EvaluationRecord.evaluation_method_version == EVALUATION_METHOD_VERSION,
                EvaluationRecord.evaluation_as_of == as_of,
            )
        )
        if existing is not None:
            return _to_result(existing)
        mapping = session.get(TargetMappingRecord, component.target_mapping_id)
        target = session.get(TargetRecord, component.target_id)
        issuance = session.get(ForecastIssuanceRecord, component.forecast_issuance_id)
        if mapping is None or target is None or issuance is None:
            raise RuntimeError("予想、対象、マッピングのDB参照が破損しています")
        ai_import = session.get(AiImportRecord, issuance.ai_import_id)
        if ai_import is None:
            raise RuntimeError("AI取込み記録のDB参照が破損しています")
        run_id = ai_import.run_id
        mapping_id = mapping.target_mapping_id
        normalized_start = component.normalized_start
        normalized_end = component.normalized_end
        expected_direction = component.direction
        symbol = target.ticker
        currency = target.currency
        mapping_status = mapping.mapping_status

    if mapping_status not in {"verified", "corrected"}:
        result = _store_without_values(
            settings,
            component_id=component_id,
            mapping_id=mapping_id,
            as_of=as_of,
            status="unevaluable",
            reason="固定済み・検証済みの対象マッピングがありません",
            run_id=run_id,
        )
        return result
    if normalized_start is None or normalized_end is None:
        return _store_without_values(
            settings,
            component_id=component_id,
            mapping_id=mapping_id,
            as_of=as_of,
            status="unevaluable",
            reason="最小方向評価に必要な開始日または終了日がありません",
            run_id=run_id,
        )
    if as_of < normalized_start:
        return _store_without_values(
            settings,
            component_id=component_id,
            mapping_id=mapping_id,
            as_of=as_of,
            status="not_started",
            reason=None,
            run_id=run_id,
        )

    effective_end = min(as_of, normalized_end)
    request = MarketDataRequest(
        symbol=symbol,
        currency=currency,
        start=normalized_start,
        end=effective_end,
    )
    try:
        series = provider.fetch(request)
        bars = tuple(bar for bar in series.bars if normalized_start <= bar.date <= effective_end)
        if not bars:
            raise MarketDataUnavailable("評価期間内の市場データが0件です")
        start_price = bars[0].adjusted_open
        end_price = bars[-1].adjusted_close
        if start_price == 0:
            raise MarketDataUnavailable("開始値が0のため変化率を計算できません")
    except MarketDataUnavailable as error:
        return _store_without_values(
            settings,
            component_id=component_id,
            mapping_id=mapping_id,
            as_of=as_of,
            status="unevaluable",
            reason=str(error),
            run_id=run_id,
        )

    actual_return = (end_price - start_price) / start_price
    period_high = max(bar.high for bar in bars)
    period_low = min(bar.low for bar in bars)
    max_favorable = (period_high - start_price) / start_price
    max_adverse = (period_low - start_price) / start_price
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
            )
            session.add(existing_series)
            session.flush()
        evaluation = EvaluationRecord(
            evaluation_id=next_id(session, "EVAL-", width=6, sequence_key="EVALUATION"),
            forecast_component_id=component_id,
            target_mapping_id=mapping_id,
            market_series_id=existing_series.market_series_id,
            evaluation_method_version=EVALUATION_METHOD_VERSION,
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
                notes="最小方向評価",
            )
        )
        issuance = session.get(
            ForecastIssuanceRecord,
            session.get(ForecastComponentRecord, component_id).forecast_issuance_id,  # type: ignore[union-attr]
        )
        if issuance is not None:
            issuance.current_status = evaluation_status
        result = _to_result(evaluation)

    from analyst_forecast.application.workflow import refresh_workflow

    refresh_workflow(settings, run_id)
    return result


def _store_without_values(
    settings: AppSettings,
    *,
    component_id: str,
    mapping_id: str,
    as_of: date,
    status: str,
    reason: str | None,
    run_id: str,
) -> EvaluationResult:
    session_factory = create_session_factory(settings.database_file)
    with session_factory.begin() as session:
        existing = session.scalar(
            select(EvaluationRecord).where(
                EvaluationRecord.forecast_component_id == component_id,
                EvaluationRecord.target_mapping_id == mapping_id,
                EvaluationRecord.evaluation_method_version == EVALUATION_METHOD_VERSION,
                EvaluationRecord.evaluation_as_of == as_of,
            )
        )
        if existing is not None:
            return _to_result(existing)
        evaluation = EvaluationRecord(
            evaluation_id=next_id(session, "EVAL-", width=6, sequence_key="EVALUATION"),
            forecast_component_id=component_id,
            target_mapping_id=mapping_id,
            evaluation_method_version=EVALUATION_METHOD_VERSION,
            evaluation_as_of=as_of,
            evaluation_status=status,
            unevaluable_reason=reason,
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
        component = session.get(ForecastComponentRecord, component_id)
        if component is not None:
            issuance = session.get(ForecastIssuanceRecord, component.forecast_issuance_id)
            if issuance is not None:
                issuance.current_status = status
        result = _to_result(evaluation)

    from analyst_forecast.application.workflow import refresh_workflow

    refresh_workflow(settings, run_id)
    return result


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
    )
