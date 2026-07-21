"""Round2: 非YouTube経路、再利用、AIレビュー、話者、バスケット。"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from analyst_forecast.application.ai_ingestion import AiIngestStatus, ingest_ai_output
from analyst_forecast.application.evaluation import evaluate_component
from analyst_forecast.application.raw_sources import RawSourceRequest, import_raw_source
from analyst_forecast.application.runs import CreateRunRequest, create_run
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.application.workflow import refresh_workflow
from analyst_forecast.domain.market import MarketBar, MarketDataRequest, MarketSeries
from analyst_forecast.domain.models import Medium
from analyst_forecast.infrastructure.db.models import (
    AiArtifactRecord,
    ForecastIssuanceRecord,
    RunSourceRecord,
    TargetMappingRecord,
)
from analyst_forecast.infrastructure.db.session import create_session_factory
from conftest import RAW_TEXT


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


def _import_medium(
    settings: AppSettings,
    run_id: str,
    tmp_path: Path,
    medium: Medium,
    *,
    name: str = "src.txt",
) -> Any:
    path = tmp_path / name
    path.write_text(RAW_TEXT, encoding="utf-8")
    return import_raw_source(
        settings,
        RawSourceRequest(
            run_id=run_id,
            input_path=path,
            medium=medium,
            url=f"https://example.invalid/{medium.value}/1",
            recorded_at=datetime(2026, 1, 10, 9, tzinfo=UTC),
            published_at=datetime(2026, 1, 10, 10, tzinfo=UTC),
        ),
    )


def _p07_payload(run_id: str, source_id: str, raw_hash: str) -> dict[str, Any]:
    return {
        "schema_version": "2.0.0",
        "run_id": run_id,
        "source_id": source_id,
        "prompt_execution": _prompt("P07"),
        "input_hash": raw_hash,
        "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
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
                "attribution_basis": "本人ブログの署名欄と一致",
                "review_status": "accepted",
                "importance": "normal",
                "high_importance_reason": None,
            }
        ],
    }


def _p08_from_upstream(
    run_id: str,
    source_id: str,
    upstream_id: str,
    input_hash: str,
    *,
    upstream_prompt: str = "P07",
    forecasts: bool = True,
    attribution: str = "target_confirmed",
    statement_kind: str = "direct_statement",
    segment_refs: list[str] | None = None,
) -> dict[str, Any]:
    quote = "日経平均は今後上昇する"
    payload: dict[str, Any] = {
        "schema_version": "2.1.0",
        "run_id": run_id,
        "source_id": source_id,
        "upstream_artifact_id": upstream_id,
        "upstream_prompt_id": upstream_prompt,
        "prompt_execution": _prompt("P08"),
        "input_hash": input_hash,
        "processing_status": "processed_with_forecasts" if forecasts else "processed_no_forecast",
        "forecasts": [],
    }
    if forecasts:
        payload["forecasts"] = [
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
                "upstream_segment_refs": segment_refs or ["seg-1"],
                "speaker_candidate": "匿名アナリストA",
                "speaker_attribution_status": attribution,
                "attribution_confidence": 0.95,
                "attribution_basis": "本人segment",
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
        ]
    return payload


def test_blog_x_web_reach_p08_and_zero_forecast(
    settings: AppSettings,
    tmp_path: Path,
) -> None:
    for medium in (Medium.BLOG, Medium.X, Medium.WEB):
        run = create_run(
            settings,
            CreateRunRequest(
                canonical_name="匿名アナリストA",
                period_start=date(2026, 1, 1),
                period_end=date(2026, 6, 30),
                evaluation_as_of=date(2026, 7, 20),
                selected_media=[medium],
            ),
            now=datetime(2026, 7, 20, 12, tzinfo=UTC),
        )
        source = _import_medium(settings, run.run_id, tmp_path, medium, name=f"{medium.value}.txt")
        state = refresh_workflow(settings, run.run_id)
        assert state.recommended_action.action_id == "RUN_PREPROCESS"
        p07 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                f"p07-{medium.value}.json",
                _p07_payload(run.run_id, source.source_id, source.raw_hash),
            ),
        )
        assert p07.status is AiIngestStatus.ACCEPTED, p07.issues
        p08 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                f"p08-{medium.value}.json",
                _p08_from_upstream(
                    run.run_id,
                    source.source_id,
                    p07.artifact_ids[0],
                    p07.output_hash,
                    forecasts=False,
                ),
            ),
        )
        assert p08.status is AiIngestStatus.ACCEPTED, p08.issues
        after = refresh_workflow(settings, run.run_id)
        assert after.stage == "processed_no_forecast"


def test_youtube_still_uses_p05(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    state = refresh_workflow(settings, run_result.run_id)
    assert "P05" in (state.recommended_action.command_or_prompt or "")
    p05 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p05.json",
            {
                "schema_version": "2.0.0",
                "run_id": run_result.run_id,
                "source_id": source_result.source_id,
                "prompt_execution": _prompt("P05"),
                "input_hash": source_result.raw_hash,
                "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                "segments": [
                    {
                        "segment_ref": "seg-1",
                        "sequence_number": 1,
                        "raw_start_offset": 0,
                        "raw_end_offset": len(RAW_TEXT),
                        "raw_text": RAW_TEXT,
                        "normalized_text": RAW_TEXT,
                        "speaker_status": "identified",
                        "speaker_candidate": "匿名アナリストA",
                        "speaker_confidence": 0.95,
                        "attribution_basis": "一致",
                        "review_status": "accepted",
                        "importance": "normal",
                        "high_importance_reason": None,
                    }
                ],
            },
        ),
    )
    assert p05.status is AiIngestStatus.ACCEPTED
    p08 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p08.json",
            _p08_from_upstream(
                run_result.run_id,
                source_result.source_id,
                p05.artifact_ids[0],
                p05.output_hash,
                upstream_prompt="P05",
                segment_refs=["seg-1"],
            ),
        ),
    )
    assert p08.status is AiIngestStatus.ACCEPTED


def test_p08_rejects_wrong_upstream_context(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    other = create_run(
        settings,
        CreateRunRequest(
            canonical_name="匿名アナリストA",
            period_start=date(2026, 2, 1),
            period_end=date(2026, 6, 30),
            evaluation_as_of=date(2026, 7, 20),
            selected_media=[Medium.YOUTUBE],
        ),
        now=datetime(2026, 7, 21, tzinfo=UTC),
    )
    p05 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p05b.json",
            {
                "schema_version": "2.0.0",
                "run_id": run_result.run_id,
                "source_id": source_result.source_id,
                "prompt_execution": _prompt("P05"),
                "input_hash": source_result.raw_hash,
                "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                "segments": [
                    {
                        "segment_ref": "seg-1",
                        "sequence_number": 1,
                        "raw_start_offset": 0,
                        "raw_end_offset": len(RAW_TEXT),
                        "raw_text": RAW_TEXT,
                        "normalized_text": RAW_TEXT,
                        "speaker_status": "identified",
                        "speaker_candidate": "匿名アナリストA",
                        "speaker_confidence": 0.95,
                        "attribution_basis": "一致",
                        "review_status": "accepted",
                        "importance": "normal",
                        "high_importance_reason": None,
                    }
                ],
            },
        ),
    )
    bad = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p08-bad.json",
            _p08_from_upstream(
                other.run_id,
                source_result.source_id,
                p05.artifact_ids[0],
                p05.output_hash,
                upstream_prompt="P05",
                forecasts=False,
            ),
        ),
    )
    assert bad.status is AiIngestStatus.REJECTED


def test_host_segment_forecast_not_formalized(
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
            {
                "schema_version": "2.0.0",
                "run_id": run_result.run_id,
                "source_id": source_result.source_id,
                "prompt_execution": _prompt("P05"),
                "input_hash": source_result.raw_hash,
                "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                "segments": [
                    {
                        "segment_ref": "host",
                        "sequence_number": 1,
                        "raw_start_offset": 0,
                        "raw_end_offset": len(RAW_TEXT),
                        "raw_text": RAW_TEXT,
                        "normalized_text": RAW_TEXT,
                        "speaker_status": "identified",
                        "speaker_candidate": "司会者",
                        "speaker_confidence": 0.95,
                        "attribution_basis": "司会紹介",
                        "review_status": "accepted",
                        "importance": "normal",
                        "high_importance_reason": None,
                    }
                ],
            },
        ),
    )
    assert p05.status is AiIngestStatus.ACCEPTED
    p08 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p08-host.json",
            _p08_from_upstream(
                run_result.run_id,
                source_result.source_id,
                p05.artifact_ids[0],
                p05.output_hash,
                upstream_prompt="P05",
                attribution="not_target",
                segment_refs=["host"],
            ),
        ),
    )
    assert p08.status is AiIngestStatus.ACCEPTED
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ForecastIssuanceRecord)) == 0


def test_p09_accept_resolves_needs_review_once(
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
            {
                "schema_version": "2.0.0",
                "run_id": run_result.run_id,
                "source_id": source_result.source_id,
                "prompt_execution": _prompt("P05"),
                "input_hash": source_result.raw_hash,
                "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                "segments": [
                    {
                        "segment_ref": "seg-1",
                        "sequence_number": 1,
                        "raw_start_offset": 0,
                        "raw_end_offset": len(RAW_TEXT),
                        "raw_text": RAW_TEXT,
                        "normalized_text": RAW_TEXT,
                        "speaker_status": "identified",
                        "speaker_candidate": "匿名アナリストA",
                        "speaker_confidence": 0.95,
                        "attribution_basis": "一致",
                        "review_status": "accepted",
                        "importance": "normal",
                        "high_importance_reason": None,
                    }
                ],
            },
        ),
    )
    low = _p08_from_upstream(
        run_result.run_id,
        source_result.source_id,
        p05.artifact_ids[0],
        p05.output_hash,
        upstream_prompt="P05",
        segment_refs=["seg-1"],
    )
    low["forecasts"][0]["extraction_confidence"] = 0.2
    p08 = ingest_ai_output(settings, _write(tmp_path, "p08-low.json", low))
    assert p08.status is AiIngestStatus.NEEDS_REVIEW
    review = {
        "schema_version": "2.0.0",
        "run_id": run_result.run_id,
        "source_id": source_result.source_id,
        "reviewed_artifact_id": p08.artifact_ids[0],
        "prompt_execution": _prompt("P09"),
        "input_hash": p08.output_hash,
        "decision": "accept",
        "findings": [
            {
                "finding_ref": "f1",
                "severity": "info",
                "message": "原文どおり",
                "evidence": "日経平均は今後上昇する",
            }
        ],
        "corrected_payload": None,
        "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
    }
    p09 = ingest_ai_output(settings, _write(tmp_path, "p09.json", review))
    assert p09.status is AiIngestStatus.ACCEPTED, p09.issues
    p09_again = ingest_ai_output(settings, _write(tmp_path, "p09-again.json", review))
    assert p09_again.status is AiIngestStatus.ALREADY_IMPORTED
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ForecastIssuanceRecord)) == 1
        reviewed = session.get(AiArtifactRecord, p08.artifact_ids[0])
        assert reviewed is not None
        assert reviewed.resolution_status == "resolved"
        assert reviewed.classification == "accepted"
    state = refresh_workflow(settings, run_result.run_id)
    assert state.counts["needs_review"] == 0
    assert state.recommended_action.action_id == "RUN_P11"


def test_safe_preprocess_reuse_across_runs(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    p05 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p05-reuse.json",
            {
                "schema_version": "2.0.0",
                "run_id": run_result.run_id,
                "source_id": source_result.source_id,
                "prompt_execution": _prompt("P05"),
                "input_hash": source_result.raw_hash,
                "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                "segments": [
                    {
                        "segment_ref": "seg-1",
                        "sequence_number": 1,
                        "raw_start_offset": 0,
                        "raw_end_offset": len(RAW_TEXT),
                        "raw_text": RAW_TEXT,
                        "normalized_text": RAW_TEXT,
                        "speaker_status": "identified",
                        "speaker_candidate": "匿名アナリストA",
                        "speaker_confidence": 0.95,
                        "attribution_basis": "一致",
                        "review_status": "accepted",
                        "importance": "normal",
                        "high_importance_reason": None,
                    }
                ],
            },
        ),
    )
    assert p05.status is AiIngestStatus.ACCEPTED
    other = create_run(
        settings,
        CreateRunRequest(
            canonical_name="匿名アナリストA",
            period_start=date(2026, 2, 1),
            period_end=date(2026, 6, 30),
            evaluation_as_of=date(2026, 7, 20),
            selected_media=[Medium.YOUTUBE],
        ),
        now=datetime(2026, 7, 21, tzinfo=UTC),
    )
    path = tmp_path / "same-raw.txt"
    path.write_text(RAW_TEXT, encoding="utf-8")
    imported = import_raw_source(
        settings,
        RawSourceRequest(
            run_id=other.run_id,
            input_path=path,
            medium=Medium.YOUTUBE,
            url="https://example.invalid/video/fixture",
            recorded_at=datetime(2026, 1, 10, 9, tzinfo=UTC),
            published_at=datetime(2026, 1, 10, 10, tzinfo=UTC),
        ),
    )
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        link = session.get(
            RunSourceRecord,
            {"run_id": other.run_id, "source_id": imported.source_id},
        )
        assert link is not None
        assert link.latest_ai_artifact_id == p05.artifact_ids[0]
        assert link.processing_status == "accepted"
    reuse_file = (
        other.run_path / "02_sources" / "youtube" / "processed" / f"P05-{p05.artifact_ids[0]}.json"
    )
    assert reuse_file.is_file()


class BasketProvider:
    name = "basket-fixture"

    def fetch(self, request: MarketDataRequest) -> MarketSeries:
        if request.symbol in {"AAA", "BBB"}:
            return MarketSeries(
                provider=self.name,
                symbol=request.symbol,
                currency="JPY",
                adjustment_type="split_adjusted_ohlc",
                frequency="1d",
                retrieved_at=datetime(2026, 7, 20, tzinfo=UTC),
                bars=(
                    MarketBar.from_prices(date(2026, 1, 13), "100", "110", high="110", low="100"),
                    MarketBar.from_prices(date(2026, 4, 13), "110", "120", high="120", low="110"),
                ),
            )
        return MarketSeries(
            provider=self.name,
            symbol=request.symbol,
            currency="USD",
            adjustment_type="split_adjusted_ohlc",
            frequency="1d",
            retrieved_at=datetime(2026, 7, 20, tzinfo=UTC),
            bars=(
                MarketBar.from_prices(date(2026, 1, 13), "10", "11", high="11", low="10"),
                MarketBar.from_prices(date(2026, 4, 13), "11", "12", high="12", low="11"),
            ),
        )


def test_equal_weight_basket_and_mixed_currency(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    from helpers_pipeline_v2 import import_locked_component

    component_id = import_locked_component(settings, run_result, source_result, tmp_path)
    session_factory = create_session_factory(settings.database_file)
    with session_factory.begin() as session:
        from analyst_forecast.infrastructure.db.models import ForecastComponentRecord

        component = session.get(ForecastComponentRecord, component_id)
        assert component is not None
        mapping = session.get(TargetMappingRecord, component.target_mapping_id)
        assert mapping is not None
        mapping.evaluation_instruments = [
            {"symbol": "AAA", "currency": "JPY", "weight": 0.5, "exchange": None},
            {"symbol": "BBB", "currency": "JPY", "weight": 0.5, "exchange": None},
        ]
        mapping.weights = [0.5, 0.5]
    result = evaluate_component(
        settings,
        component_id=component_id,
        provider=BasketProvider(),
        as_of=date(2026, 4, 13),
        run_id=run_result.run_id,
    )
    assert result.evaluation_status != "unevaluable"
    # 同一パスの2銘柄50/50は単一銘柄と同じリターン
    assert result.actual_return == Decimal("0.2")
    assert result.max_favorable_excursion == Decimal("0.2")

    with session_factory.begin() as session:
        component = session.get(ForecastComponentRecord, component_id)
        assert component is not None
        mapping = session.get(TargetMappingRecord, component.target_mapping_id)
        assert mapping is not None
        mapping.evaluation_instruments = [
            {"symbol": "AAA", "currency": "JPY", "weight": 0.5, "exchange": None},
            {"symbol": "USD1", "currency": "USD", "weight": 0.5, "exchange": None},
        ]
        mapping.weights = [0.5, 0.5]
        mapping.mapping_hash = mapping.mapping_hash[:-1] + (
            "0" if mapping.mapping_hash[-1] != "0" else "1"
        )
    mixed = evaluate_component(
        settings,
        component_id=component_id,
        provider=BasketProvider(),
        as_of=date(2026, 4, 14),
        run_id=run_result.run_id,
    )
    assert mixed.evaluation_status == "unevaluable"
    assert mixed.unevaluable_reason == "unevaluable_mixed_currency"
