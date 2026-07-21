"""Round3 critical bug regression tests."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from analyst_forecast.application.ai_ingestion import AiIngestStatus, ingest_ai_output
from analyst_forecast.application.analysts import add_analyst_alias
from analyst_forecast.application.evaluation import evaluate_component
from analyst_forecast.application.raw_sources import RawSourceRequest, import_raw_source
from analyst_forecast.application.runs import CreateRunRequest, create_run
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.application.workflow import refresh_workflow
from analyst_forecast.domain.market import MarketBar, MarketDataRequest, MarketSeries
from analyst_forecast.domain.models import Medium
from analyst_forecast.infrastructure.db.models import (
    ForecastEvidenceRecord,
    ForecastIssuanceRecord,
    MarketSeriesRecord,
    SegmentRecord,
)
from analyst_forecast.infrastructure.db.session import create_session_factory
from conftest import RAW_TEXT
from helpers_pipeline_v2 import import_locked_component


def _write(tmp_path: Path, name: str, payload: dict[str, Any]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _prompt(prompt_id: str) -> dict[str, str]:
    return {
        "prompt_id": prompt_id,
        "prompt_version": "2.0.0",
        "environment": "cursor",
        "model": "high-performance-fixture",
        "executed_at": "2026-07-20T12:00:00+00:00",
    }


def _p05(
    run_id: str,
    source_id: str,
    raw_hash: str,
    *,
    speaker: str = "匿名アナリストA",
    cutoff: str = "2026-01-10T09:00:00+00:00",
    segment_ref: str = "seg-1",
) -> dict[str, Any]:
    return {
        "schema_version": "2.0.0",
        "run_id": run_id,
        "source_id": source_id,
        "prompt_execution": _prompt("P05"),
        "input_hash": raw_hash,
        "knowledge_cutoff": cutoff,
        "segments": [
            {
                "segment_ref": segment_ref,
                "sequence_number": 1,
                "raw_start_offset": 0,
                "raw_end_offset": len(RAW_TEXT),
                "raw_text": RAW_TEXT,
                "normalized_text": RAW_TEXT,
                "speaker_status": "identified",
                "speaker_candidate": speaker,
                "speaker_confidence": 0.95,
                "attribution_basis": "fixture",
                "review_status": "accepted",
                "importance": "normal",
                "high_importance_reason": None,
            }
        ],
    }


def _p08(
    run_id: str,
    source_id: str,
    upstream_id: str,
    input_hash: str,
    *,
    attribution: str = "target_confirmed",
    statement_kind: str = "direct_statement",
    segment_refs: list[str] | None = None,
    speaker: str = "匿名アナリストA",
) -> dict[str, Any]:
    quote = "日経平均は今後上昇する"
    return {
        "schema_version": "2.1.0",
        "run_id": run_id,
        "source_id": source_id,
        "upstream_artifact_id": upstream_id,
        "upstream_prompt_id": "P05",
        "prompt_execution": _prompt("P08"),
        "input_hash": input_hash,
        "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
        "processing_status": "processed_with_forecasts",
        "forecasts": [
            {
                "forecast_ref": "f1",
                "forecast_group_ref": "g1",
                "made_at": "2026-01-10T09:00:00+00:00",
                "publicly_available_at": "2026-01-10T10:00:00+00:00",
                "made_at_source": "explicit",
                "forecast_type": "directional",
                "commitment_strength": "explicit",
                "evidence_level": "A",
                "extraction_confidence": 0.95,
                "importance": "normal",
                "high_importance_reason": None,
                "human_readable_summary": quote,
                "relation_to_previous": "initial",
                "upstream_segment_refs": segment_refs if segment_refs is not None else ["seg-1"],
                "speaker_candidate": speaker,
                "speaker_attribution_status": attribution,
                "attribution_confidence": 0.95,
                "attribution_basis": "test",
                "statement_kind": statement_kind,
                "evidence": [
                    {
                        "source_id": source_id,
                        "quote": quote,
                        "start_offset": 0,
                        "end_offset": len(quote),
                        "role": "prediction",
                    }
                ],
                "components": [
                    {
                        "component_ref": "c1",
                        "sequence_number": 1,
                        "prediction_form": "period_direction",
                        "direction": "up",
                        "time_expression_raw": "今後3か月",
                        "time_source": "explicit",
                        "normalized_start": "2026-01-13",
                        "normalized_end": "2026-04-13",
                        "raw_target_label": "日経平均",
                        "target_resolution_status": "pending",
                    }
                ],
            }
        ],
    }


def test_host_target_confirmed_claim_not_formalized(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    p05 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p05-host.json",
            _p05(
                run_result.run_id,
                source_result.source_id,
                source_result.raw_hash,
                speaker="司会者",
            ),
        ),
    )
    assert p05.status is AiIngestStatus.ACCEPTED
    p08 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p08-host-claim.json",
            _p08(
                run_result.run_id,
                source_result.source_id,
                p05.artifact_ids[0],
                p05.output_hash,
                attribution="target_confirmed",
                speaker="匿名アナリストA",
            ),
        ),
    )
    assert p08.status is AiIngestStatus.REJECTED
    assert any(issue.code == "attribution_claim_rejected" for issue in p08.issues)
    assert len(p08.forecast_issuance_ids) == 0


def test_legacy_unknown_not_formalized(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    p05 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p05-legacy.json",
            _p05(run_result.run_id, source_result.source_id, source_result.raw_hash),
        ),
    )
    p08 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p08-legacy.json",
            _p08(
                run_result.run_id,
                source_result.source_id,
                p05.artifact_ids[0],
                p05.output_hash,
                attribution="legacy_unknown",
            ),
        ),
    )
    assert p08.status is AiIngestStatus.ACCEPTED, p08.issues
    assert len(p08.forecast_issuance_ids) == 0
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ForecastIssuanceRecord)) == 0


def test_canonical_and_alias_speaker_formalized(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    p05 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p05-canon.json",
            _p05(run_result.run_id, source_result.source_id, source_result.raw_hash),
        ),
    )
    p08 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p08-canon.json",
            _p08(
                run_result.run_id,
                source_result.source_id,
                p05.artifact_ids[0],
                p05.output_hash,
            ),
        ),
    )
    assert p08.status is AiIngestStatus.ACCEPTED, p08.issues
    assert len(p08.forecast_issuance_ids) == 1

    add_analyst_alias(
        settings,
        analyst_id=run_result.analyst_id,
        alias="別名アナリストＡ",  # fullwidth A → NFKC
    )
    run2 = create_run(
        settings,
        CreateRunRequest(
            canonical_name="匿名アナリストA",
            period_start=date(2026, 2, 1),
            period_end=date(2026, 6, 30),
            evaluation_as_of=date(2026, 7, 20),
            selected_media=[Medium.YOUTUBE],
        ),
        now=datetime(2026, 7, 20, 13, tzinfo=UTC),
    )
    raw2 = tmp_path / "raw2.txt"
    raw2.write_text(RAW_TEXT, encoding="utf-8")
    src2 = import_raw_source(
        settings,
        RawSourceRequest(
            run_id=run2.run_id,
            input_path=raw2,
            medium=Medium.YOUTUBE,
            url="https://example.invalid/video/2",
            title="alias",
            recorded_at=datetime(2026, 1, 10, 9, tzinfo=UTC),
            published_at=datetime(2026, 1, 10, 10, tzinfo=UTC),
            retrieved_at=datetime(2026, 7, 20, 12, tzinfo=UTC),
        ),
    )
    p05b = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p05-alias.json",
            _p05(
                run2.run_id,
                src2.source_id,
                src2.raw_hash,
                speaker="別名アナリストA",
            ),
        ),
    )
    assert p05b.status is AiIngestStatus.ACCEPTED, p05b.issues
    p08b = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p08-alias.json",
            _p08(
                run2.run_id,
                src2.source_id,
                p05b.artifact_ids[0],
                p05b.output_hash,
                speaker="別名アナリストA",
            ),
        ),
    )
    assert p08b.status is AiIngestStatus.ACCEPTED, p08b.issues
    assert len(p08b.forecast_issuance_ids) == 1


def test_p05_future_cutoff_rejected(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    result = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p05-future.json",
            _p05(
                run_result.run_id,
                source_result.source_id,
                source_result.raw_hash,
                cutoff="2026-01-11T00:00:00+00:00",
            ),
        ),
    )
    assert result.status is AiIngestStatus.REJECTED
    assert any(issue.code == "future_knowledge_cutoff" for issue in result.issues)


def test_p09_bad_offset_correct_rejected(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    p05 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p05-r.json",
            _p05(run_result.run_id, source_result.source_id, source_result.raw_hash),
        ),
    )
    low = _p08(
        run_result.run_id,
        source_result.source_id,
        p05.artifact_ids[0],
        p05.output_hash,
    )
    low["forecasts"][0]["extraction_confidence"] = 0.2
    p08 = ingest_ai_output(settings, _write(tmp_path, "p08-low.json", low))
    assert p08.status is AiIngestStatus.NEEDS_REVIEW

    corrected = _p08(
        run_result.run_id,
        source_result.source_id,
        p05.artifact_ids[0],
        p05.output_hash,
    )
    corrected["forecasts"][0]["evidence"][0]["start_offset"] = 1
    review = {
        "schema_version": "2.0.0",
        "run_id": run_result.run_id,
        "source_id": source_result.source_id,
        "reviewed_artifact_id": p08.artifact_ids[0],
        "prompt_execution": _prompt("P09"),
        "input_hash": p08.output_hash,
        "decision": "correct",
        "findings": [
            {
                "finding_ref": "f1",
                "severity": "error",
                "message": "offset修正",
                "evidence": "bad",
            }
        ],
        "corrected_payload": corrected,
        "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
    }
    p09 = ingest_ai_output(settings, _write(tmp_path, "p09-bad.json", review))
    assert p09.status is AiIngestStatus.REJECTED
    assert any(
        issue.code in {"quote_offset_mismatch", "evidence_outside_segment"} for issue in p09.issues
    )
    assert len(p09.forecast_issuance_ids) == 0


def test_multi_source_pending_p08_not_skipped(settings: AppSettings, tmp_path: Path) -> None:
    run = create_run(
        settings,
        CreateRunRequest(
            canonical_name="匿名アナリストA",
            period_start=date(2026, 1, 1),
            period_end=date(2026, 6, 30),
            evaluation_as_of=date(2026, 7, 20),
            selected_media=[Medium.BLOG],
        ),
        now=datetime(2026, 7, 20, 12, tzinfo=UTC),
    )
    sources = []
    for index in (1, 2):
        path = tmp_path / f"blog{index}.txt"
        path.write_text(RAW_TEXT, encoding="utf-8")
        sources.append(
            import_raw_source(
                settings,
                RawSourceRequest(
                    run_id=run.run_id,
                    input_path=path,
                    medium=Medium.BLOG,
                    url=f"https://example.invalid/blog/{index}",
                    title=f"blog{index}",
                    published_at=datetime(2026, 1, 10, 10, tzinfo=UTC),
                    retrieved_at=datetime(2026, 7, 20, 12, tzinfo=UTC),
                ),
            )
        )

    def _p07(source: Any, name: str) -> Any:
        return ingest_ai_output(
            settings,
            _write(
                tmp_path,
                name,
                {
                    "schema_version": "2.0.0",
                    "run_id": run.run_id,
                    "source_id": source.source_id,
                    "prompt_execution": _prompt("P07"),
                    "input_hash": source.raw_hash,
                    "knowledge_cutoff": "2026-01-10T10:00:00+00:00",
                    "segments": [
                        {
                            "segment_ref": "seg-1",
                            "sequence_number": 1,
                            "raw_start_offset": 0,
                            "raw_end_offset": len(RAW_TEXT),
                            "raw_text": RAW_TEXT,
                            "normalized_text": RAW_TEXT,
                            "author_status": "identified",
                            "author_candidate": "匿名アナリストA",
                            "author_confidence": 0.95,
                            "statement_kind": "author_own",
                            "attribution_basis": "署名",
                            "review_status": "accepted",
                            "importance": "normal",
                            "high_importance_reason": None,
                        }
                    ],
                },
            ),
        )

    p07a = _p07(sources[0], "p07a.json")
    p07b = _p07(sources[1], "p07b.json")
    assert p07a.status is AiIngestStatus.ACCEPTED
    assert p07b.status is AiIngestStatus.ACCEPTED

    p08_payload = _p08(
        run.run_id,
        sources[0].source_id,
        p07a.artifact_ids[0],
        p07a.output_hash,
    )
    p08_payload["upstream_prompt_id"] = "P07"
    p08_payload["upstream_artifact_id"] = p07a.artifact_ids[0]
    p08a = ingest_ai_output(settings, _write(tmp_path, "p08a.json", p08_payload))
    assert p08a.status is AiIngestStatus.ACCEPTED, p08a.issues
    assert len(p08a.forecast_issuance_ids) == 1

    state = refresh_workflow(settings, run.run_id)
    assert state.recommended_action.action_id == "EXTRACT_FORECASTS"
    assert state.recommended_action.action_id != "RUN_P11"


class _MockProvider:
    name = "mock"

    def __init__(self, series_by_symbol: dict[str, MarketSeries]) -> None:
        self.series_by_symbol = series_by_symbol

    def fetch(self, request: MarketDataRequest) -> MarketSeries:
        series = self.series_by_symbol[request.symbol]
        assert series.symbol == request.symbol
        assert series.currency == request.currency
        return series


def _series(symbol: str, open_: str, close: str) -> MarketSeries:
    return MarketSeries(
        provider="mock",
        symbol=symbol,
        currency="JPY",
        adjustment_type="split_adjusted_ohlc",
        frequency="1d",
        retrieved_at=datetime(2026, 7, 20, tzinfo=UTC),
        bars=(
            MarketBar(
                date=date(2026, 1, 13),
                open=Decimal(open_),
                high=Decimal(close) if Decimal(close) > Decimal(open_) else Decimal(open_),
                low=Decimal(close) if Decimal(close) < Decimal(open_) else Decimal(open_),
                close=Decimal(close),
                adjusted_open=Decimal(open_),
                adjusted_close=Decimal(close),
            ),
            MarketBar(
                date=date(2026, 4, 13),
                open=Decimal(close),
                high=Decimal(close),
                low=Decimal(close),
                close=Decimal(close),
                adjusted_open=Decimal(close),
                adjusted_close=Decimal(close),
            ),
        ),
    )


def test_basket_cache_does_not_pollute_single_symbol(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    """R4-042: basket cache must not pollute single-symbol AAA series cache."""
    from analyst_forecast.infrastructure.db.ids import next_id
    from analyst_forecast.infrastructure.db.models import (
        ForecastComponentRecord,
        TargetMappingRecord,
        TargetRecord,
    )

    component_id = import_locked_component(settings, run_result, source_result, tmp_path)
    session_factory = create_session_factory(settings.database_file)
    with session_factory.begin() as session:
        component = session.get(ForecastComponentRecord, component_id)
        assert component is not None
        mapping = session.get(TargetMappingRecord, component.target_mapping_id)
        assert mapping is not None
        mapping.evaluation_instruments = [
            {"symbol": "AAA", "currency": "JPY", "weight": 0.5, "exchange": None},
            {"symbol": "BBB", "currency": "JPY", "weight": 0.5, "exchange": None},
        ]
        mapping.weights = [0.5, 0.5]
        issuance_id = component.forecast_issuance_id

        aaa_target = TargetRecord(
            target_id=next_id(session, "TGT-", width=6, sequence_key="TARGET"),
            raw_label="AAA",
            canonical_name="AAA",
            target_type="equity",
            ticker="AAA",
            currency="JPY",
        )
        session.add(aaa_target)
        session.flush()
        aaa_mapping = TargetMappingRecord(
            target_mapping_id=next_id(session, "MAP-", width=6, sequence_key="TARGET_MAPPING"),
            target_id=aaa_target.target_id,
            mapping_method="explicit",
            evaluation_instruments=[{"symbol": "AAA", "currency": "JPY", "weight": 1.0}],
            weights=[1.0],
            knowledge_cutoff=datetime(2026, 1, 10, 9, tzinfo=UTC),
            source_evidence="fixture",
            mapping_status=mapping.mapping_status,
            mapping_hash="aaa-only-fixture-hash",
            locked_at=datetime(2026, 7, 20, tzinfo=UTC),
        )
        session.add(aaa_mapping)
        session.flush()
        aaa_component = ForecastComponentRecord(
            forecast_component_id=next_id(
                session, "FCC-", width=6, sequence_key="FORECAST_COMPONENT"
            ),
            forecast_issuance_id=issuance_id,
            local_ref="aaa-only",
            sequence_number=2,
            prediction_form="period_direction",
            direction="up",
            time_expression_raw="今後3か月",
            time_source="explicit",
            normalized_start=date(2026, 1, 13),
            normalized_end=date(2026, 4, 13),
            raw_target_label="AAA",
            target_resolution_status="locked",
            importance="normal",
            target_id=aaa_target.target_id,
            target_mapping_id=aaa_mapping.target_mapping_id,
        )
        session.add(aaa_component)
        session.flush()
        aaa_component_id = aaa_component.forecast_component_id

    class PollutionProvider:
        name = "pollution-fixture"

        def fetch(self, request: MarketDataRequest) -> MarketSeries:
            if request.symbol == "AAA":
                return _series("AAA", "100", "120")
            if request.symbol == "BBB":
                return _series("BBB", "100", "80")
            raise AssertionError(f"unexpected symbol {request.symbol}")

    provider = PollutionProvider()
    basket = evaluate_component(
        settings,
        component_id=component_id,
        provider=provider,
        as_of=date(2026, 4, 13),
        run_id=run_result.run_id,
    )
    assert basket.actual_return is not None, basket.unevaluable_reason
    assert abs(basket.actual_return - Decimal("0")) < Decimal("0.0001")

    aaa = evaluate_component(
        settings,
        component_id=aaa_component_id,
        provider=provider,
        as_of=date(2026, 4, 13),
        run_id=run_result.run_id,
    )
    assert aaa.actual_return is not None, aaa.unevaluable_reason
    assert abs(aaa.actual_return - Decimal("0.2")) < Decimal("0.0001")

    with session_factory() as session:
        kinds = set(session.scalars(select(MarketSeriesRecord.series_kind)))
        assert "basket" in kinds
        assert "raw" in kinds
        for record in session.scalars(select(MarketSeriesRecord)):
            if record.series_kind == "basket":
                assert str(record.symbol).startswith("BASKET:")
            if record.series_kind == "raw":
                assert not str(record.symbol).startswith("BASKET:")


def test_evidence_segment_lineage(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    p05 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p05-ev.json",
            _p05(run_result.run_id, source_result.source_id, source_result.raw_hash),
        ),
    )
    p08 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p08-ev.json",
            _p08(
                run_result.run_id,
                source_result.source_id,
                p05.artifact_ids[0],
                p05.output_hash,
            ),
        ),
    )
    assert p08.status is AiIngestStatus.ACCEPTED
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        evidence = session.scalar(select(ForecastEvidenceRecord))
        assert evidence is not None
        assert evidence.segment_id is not None
        segment = session.get(SegmentRecord, evidence.segment_id)
        assert segment is not None
        assert segment.local_ref == "seg-1"
