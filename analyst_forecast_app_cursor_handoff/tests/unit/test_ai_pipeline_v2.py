from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select

from analyst_forecast.application.ai_ingestion import AiIngestStatus, ingest_ai_output
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.infrastructure.db.models import (
    AiArtifactRecord,
    ForecastComponentRecord,
    ForecastIssuanceRecord,
    PromptExecutionRecord,
    RunSourceRecord,
    SegmentRecord,
    TargetMappingRecord,
    TargetRecord,
    TargetResolutionCandidateRecord,
)
from analyst_forecast.infrastructure.db.session import create_session_factory
from analyst_forecast.schemas.pipeline import PIPELINE_MODELS, pipeline_schema_path
from conftest import RAW_TEXT


def _write(tmp_path: Path, name: str, payload: dict[str, Any]) -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _prompt(prompt_id: str) -> dict[str, str]:
    return {
        "prompt_id": prompt_id,
        "prompt_version": "2.0.0",
        "environment": "cursor",
        "model": "high-performance-fixture",
        "executed_at": "2026-07-20T12:00:00+00:00",
    }


def _p05_payload(
    run_id: str,
    source_id: str,
    raw_hash: str,
    *,
    unknown: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": "2.0.0",
        "run_id": run_id,
        "source_id": source_id,
        "prompt_execution": _prompt("P05"),
        "input_hash": raw_hash,
        "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
        "segments": [
            {
                "segment_ref": "segment-1",
                "sequence_number": 1,
                "raw_start_offset": 0,
                "raw_end_offset": len(RAW_TEXT),
                "raw_text": RAW_TEXT,
                "normalized_text": RAW_TEXT,
                "speaker_status": "unknown" if unknown else "identified",
                "speaker_candidate": None if unknown else "匿名アナリストA",
                "speaker_confidence": 0.20 if unknown else 0.95,
                "attribution_basis": (
                    "話者情報がないため判断不能" if unknown else "fixtureメタデータと発言者が一致"
                ),
                "review_status": "needs_review" if unknown else "accepted",
                "importance": "normal",
                "high_importance_reason": None,
            }
        ],
    }


def _p08_payload(
    run_id: str,
    source_id: str,
    p05_artifact_id: str,
    input_hash: str,
    *,
    forecasts: bool = True,
    confidence: float = 0.95,
    importance: str = "normal",
) -> dict[str, Any]:
    quote = "日経平均は今後上昇する"
    payload: dict[str, Any] = {
        "schema_version": "2.0.0",
        "run_id": run_id,
        "source_id": source_id,
        "p05_artifact_id": p05_artifact_id,
        "prompt_execution": _prompt("P08"),
        "input_hash": input_hash,
        "processing_status": ("processed_with_forecasts" if forecasts else "processed_no_forecast"),
        "forecasts": [],
    }
    if forecasts:
        payload["forecasts"] = [
            {
                "forecast_ref": "forecast-1",
                "forecast_group_ref": "group-1",
                "made_at": "2026-01-10T09:00:00+00:00",
                "publicly_available_at": "2026-01-10T10:00:00+00:00",
                "made_at_source": "explicit",
                "forecast_type": "directional",
                "commitment_strength": "explicit",
                "evidence_level": "A",
                "extraction_confidence": confidence,
                "importance": importance,
                "high_importance_reason": (
                    "市場全体に影響する予想" if importance == "high" else None
                ),
                "human_readable_summary": quote,
                "relation_to_previous": "initial",
                "upstream_segment_refs": ["segment-1"],
                "speaker_candidate": "匿名アナリストA",
                "speaker_attribution_status": "target_confirmed",
                "attribution_confidence": 0.95,
                "attribution_basis": "本人segment",
                "statement_kind": "direct_statement",
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
                        "component_ref": "component-1",
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


def _candidate(
    ref: str,
    rank: int,
    symbol: str,
    *,
    instruments: int = 1,
) -> dict[str, Any]:
    values = [
        {
            "symbol": symbol if index == 0 else f"{symbol}-{index + 1}",
            "exchange": "JPX",
            "currency": "JPY",
            "weight": 1 / instruments,
        }
        for index in range(instruments)
    ]
    return {
        "candidate_ref": ref,
        "rank": rank,
        "canonical_name": f"候補{rank}",
        "target_type": "index",
        "mapping_method": "explicit",
        "instruments": values,
        "existed_at": "2026-01-10",
        "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
        "source_evidence": "発言日時点の資料",
        "confidence": 0.90,
    }


def _p11_payload(
    run_id: str,
    source_id: str,
    component_id: str,
    input_hash: str,
    *,
    candidates: list[dict[str, Any]] | None = None,
    unresolvable: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": "2.0.0",
        "run_id": run_id,
        "source_id": source_id,
        "forecast_component_id": component_id,
        "prompt_execution": _prompt("P11"),
        "input_hash": input_hash,
        "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
        "resolution_status": "unresolvable" if unresolvable else "proposed",
        "candidates": []
        if unresolvable
        else (candidates or [_candidate("candidate-1", 1, "^N225")]),
        "unevaluable_reason": ("発言日時点に適切な評価指標が存在しない" if unresolvable else None),
    }


def _p12_payload(
    run_id: str,
    source_id: str,
    component_id: str,
    proposal_artifact_id: str,
    input_hash: str,
    *,
    resolution_status: str = "agreed",
    candidate_ref: str | None = "candidate-1",
) -> dict[str, Any]:
    return {
        "schema_version": "2.0.0",
        "run_id": run_id,
        "source_id": source_id,
        "forecast_component_id": component_id,
        "proposal_artifact_id": proposal_artifact_id,
        "prompt_execution": _prompt("P12"),
        "input_hash": input_hash,
        "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
        "resolution_status": resolution_status,
        "reviews": (
            []
            if candidate_ref is None
            else [
                {
                    "candidate_ref": candidate_ref,
                    "decision": ("accept" if resolution_status == "agreed" else "reject"),
                    "confidence": 0.92,
                    "rationale": "原文と当時資料を独立確認",
                }
            ]
        ),
        "recommended_candidate_ref": (candidate_ref if resolution_status == "agreed" else None),
        "unevaluable_reason": (
            "独立レビューでも評価指標を特定不能" if resolution_status == "unresolved" else None
        ),
    }


def _p13_payload(
    run_id: str,
    source_id: str,
    component_id: str,
    proposal_artifact_id: str,
    review_artifact_id: str,
    input_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": "2.0.0",
        "run_id": run_id,
        "source_id": source_id,
        "forecast_component_id": component_id,
        "proposal_artifact_id": proposal_artifact_id,
        "review_artifact_id": review_artifact_id,
        "prompt_execution": _prompt("P13"),
        "input_hash": input_hash,
        "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
        "final_status": "verified",
        "selected_candidate_ref": "candidate-1",
        "rationale": "提案と独立レビューを比較し、発言時点の指数を採用",
        "unevaluable_reason": None,
    }


def _ingest_p05_p08(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
    *,
    forecasts: bool = True,
) -> tuple[Any, Any]:
    p05 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p05.json",
            _p05_payload(
                run_result.run_id,
                source_result.source_id,
                source_result.raw_hash,
            ),
        ),
    )
    assert p05.status is AiIngestStatus.ACCEPTED, p05.issues
    p08 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p08.json",
            _p08_payload(
                run_result.run_id,
                source_result.source_id,
                p05.artifact_ids[0],
                p05.output_hash,
                forecasts=forecasts,
            ),
        ),
    )
    return p05, p08


@pytest.mark.parametrize(
    "prompt_id",
    ["P05", "P06", "P07", "P08", "P09", "P11", "P12", "P13"],
)
def test_pipeline_fixed_schema_matches_pydantic_model(prompt_id: str) -> None:
    fixed = json.loads(pipeline_schema_path(prompt_id).read_text(encoding="utf-8"))
    generated = PIPELINE_MODELS[prompt_id].model_json_schema()
    if prompt_id == "P09":
        # Round6: fixed Schema carries Draft 2020-12 allOf conditionals that
        # model_json_schema() cannot express; dual-validator matrix covers them.
        assert "allOf" in fixed
        assert set(fixed["properties"]) >= set(generated["properties"])
        assert fixed["title"] == generated["title"]
    else:
        assert fixed == generated


def test_p05_unknown_speaker_is_saved_without_forcing_analyst(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    result = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p05.json",
            _p05_payload(
                run_result.run_id,
                source_result.source_id,
                source_result.raw_hash,
                unknown=True,
            ),
        ),
    )

    assert result.status is AiIngestStatus.NEEDS_REVIEW
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        segment = session.scalar(select(SegmentRecord))
        assert segment is not None
        assert segment.speaker_candidate is None
        assert segment.speaker_status == "unknown"
        artifact = session.get(AiArtifactRecord, result.artifact_ids[0])
        assert artifact is not None
        assert artifact.prompt_id == "P05"
    assert list((run_result.run_path / "02_sources" / "youtube" / "processed").glob("P05-*.json"))


def test_p08_accepts_processed_source_with_zero_forecasts(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    _, result = _ingest_p05_p08(
        settings,
        run_result,
        source_result,
        tmp_path,
        forecasts=False,
    )

    assert result.status is AiIngestStatus.ACCEPTED, result.issues
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        link = session.get(
            RunSourceRecord,
            {"run_id": run_result.run_id, "source_id": source_result.source_id},
        )
        assert link is not None
        assert link.processing_status == "processed_no_forecast"
        assert session.scalar(select(func.count()).select_from(ForecastIssuanceRecord)) == 0


def test_p11_alone_never_verifies_mapping_and_saves_three_candidates(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    _, p08 = _ingest_p05_p08(settings, run_result, source_result, tmp_path)
    candidates = [
        _candidate("candidate-1", 1, "^N225", instruments=2),
        _candidate("candidate-2", 2, "1321.T"),
        _candidate("candidate-3", 3, "1306.T"),
    ]
    p11 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p11.json",
            _p11_payload(
                run_result.run_id,
                source_result.source_id,
                p08.component_ids[0],
                p08.output_hash,
                candidates=candidates,
            ),
        ),
    )

    assert p11.status is AiIngestStatus.ACCEPTED
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        component = session.get(ForecastComponentRecord, p08.component_ids[0])
        assert component is not None
        assert component.target_mapping_id is None
        assert component.target_resolution_status == "awaiting_review"
        saved = list(session.scalars(select(TargetResolutionCandidateRecord)))
        assert len(saved) == 3
        assert len(saved[0].instruments) == 2
        assert session.scalar(select(func.count()).select_from(TargetMappingRecord)) == 0


def test_separate_p12_execution_is_required_before_mapping_lock(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    _, p08 = _ingest_p05_p08(settings, run_result, source_result, tmp_path)
    p11 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p11.json",
            _p11_payload(
                run_result.run_id,
                source_result.source_id,
                p08.component_ids[0],
                p08.output_hash,
            ),
        ),
    )
    p12 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p12.json",
            _p12_payload(
                run_result.run_id,
                source_result.source_id,
                p08.component_ids[0],
                p11.artifact_ids[0],
                p11.output_hash,
            ),
        ),
    )

    assert p12.status is AiIngestStatus.ACCEPTED
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        artifacts = list(
            session.scalars(
                select(AiArtifactRecord).where(AiArtifactRecord.prompt_id.in_(["P11", "P12"]))
            )
        )
        assert len(artifacts) == 2
        assert artifacts[0].prompt_execution_id != artifacts[1].prompt_execution_id
        executions = list(
            session.scalars(
                select(PromptExecutionRecord).where(
                    PromptExecutionRecord.prompt_id.in_(["P11", "P12"])
                )
            )
        )
        assert {execution.prompt_id for execution in executions} == {"P11", "P12"}
        component = session.get(ForecastComponentRecord, p08.component_ids[0])
        assert component is not None
        assert component.target_resolution_status == "locked"
        mapping = session.get(TargetMappingRecord, component.target_mapping_id)
        assert mapping is not None
        assert mapping.mapping_status == "verified"
        assert mapping.locked_at is not None


def test_p12_reference_to_non_p11_artifact_is_rejected(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    p05, p08 = _ingest_p05_p08(settings, run_result, source_result, tmp_path)
    result = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "bad-p12.json",
            _p12_payload(
                run_result.run_id,
                source_result.source_id,
                p08.component_ids[0],
                p05.artifact_ids[0],
                p05.output_hash,
            ),
        ),
    )

    assert result.status is AiIngestStatus.REJECTED
    assert any("P11" in issue.message for issue in result.issues)


def test_p11_p12_disagreement_waits_for_p13_adjudication(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    _, p08 = _ingest_p05_p08(settings, run_result, source_result, tmp_path)
    p11 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p11.json",
            _p11_payload(
                run_result.run_id,
                source_result.source_id,
                p08.component_ids[0],
                p08.output_hash,
            ),
        ),
    )
    p12 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p12-disagreed.json",
            _p12_payload(
                run_result.run_id,
                source_result.source_id,
                p08.component_ids[0],
                p11.artifact_ids[0],
                p11.output_hash,
                resolution_status="disagreed",
            ),
        ),
    )
    assert p12.status is AiIngestStatus.ACCEPTED

    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        component = session.get(ForecastComponentRecord, p08.component_ids[0])
        assert component is not None
        assert component.target_resolution_status == "awaiting_adjudication"
        assert component.target_mapping_id is None

    p13 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p13.json",
            _p13_payload(
                run_result.run_id,
                source_result.source_id,
                p08.component_ids[0],
                p11.artifact_ids[0],
                p12.artifact_ids[0],
                p12.output_hash,
            ),
        ),
    )

    assert p13.status is AiIngestStatus.ACCEPTED
    with session_factory() as session:
        component = session.get(ForecastComponentRecord, p08.component_ids[0])
        assert component is not None
        assert component.target_resolution_status == "locked"
        assert component.target_mapping_id is not None


def test_unresolvable_target_is_saved_without_dummy_symbol(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    _, p08 = _ingest_p05_p08(settings, run_result, source_result, tmp_path)
    p11 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p11-unresolvable.json",
            _p11_payload(
                run_result.run_id,
                source_result.source_id,
                p08.component_ids[0],
                p08.output_hash,
                unresolvable=True,
            ),
        ),
    )
    p12 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p12-unresolvable.json",
            _p12_payload(
                run_result.run_id,
                source_result.source_id,
                p08.component_ids[0],
                p11.artifact_ids[0],
                p11.output_hash,
                resolution_status="unresolved",
                candidate_ref=None,
            ),
        ),
    )

    assert p12.status is AiIngestStatus.ACCEPTED
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        target = session.scalar(select(TargetRecord))
        mapping = session.scalar(select(TargetMappingRecord))
        assert target is not None
        assert target.ticker is None
        assert target.currency is None
        assert mapping is not None
        assert mapping.mapping_status == "unresolvable"
        assert mapping.unevaluable_reason


def test_low_confidence_or_high_importance_p08_waits_for_review(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    p05 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p05.json",
            _p05_payload(
                run_result.run_id,
                source_result.source_id,
                source_result.raw_hash,
            ),
        ),
    )
    low = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p08-low.json",
            _p08_payload(
                run_result.run_id,
                source_result.source_id,
                p05.artifact_ids[0],
                p05.output_hash,
                confidence=0.30,
            ),
        ),
    )
    high = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p08-high.json",
            _p08_payload(
                run_result.run_id,
                source_result.source_id,
                p05.artifact_ids[0],
                p05.output_hash,
                importance="high",
            ),
        ),
    )

    assert low.status is AiIngestStatus.NEEDS_REVIEW
    assert high.status is AiIngestStatus.NEEDS_REVIEW
    assert any("高重要度" in issue.message for issue in high.issues)


def test_future_knowledge_cutoff_and_market_outcome_are_rejected(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    _, p08 = _ingest_p05_p08(settings, run_result, source_result, tmp_path)
    future = _p11_payload(
        run_result.run_id,
        source_result.source_id,
        p08.component_ids[0],
        p08.output_hash,
    )
    future["knowledge_cutoff"] = "2026-01-11T09:00:00+00:00"
    future_result = ingest_ai_output(
        settings,
        _write(tmp_path, "p11-future.json", future),
    )

    leaked = _p11_payload(
        run_result.run_id,
        source_result.source_id,
        p08.component_ids[0],
        p08.output_hash,
    )
    leaked["actual_return"] = 0.50
    leaked_result = ingest_ai_output(
        settings,
        _write(tmp_path, "p11-leaked.json", leaked),
    )

    assert future_result.status is AiIngestStatus.REJECTED
    assert any("発言日時" in issue.message for issue in future_result.issues)
    assert leaked_result.status is AiIngestStatus.REJECTED
