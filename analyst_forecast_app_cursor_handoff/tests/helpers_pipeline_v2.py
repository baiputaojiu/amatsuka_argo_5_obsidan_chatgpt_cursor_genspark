"""Schema 2.0.0 取込みヘルパー（評価・workflow試験用）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from analyst_forecast.application.ai_ingestion import AiIngestStatus, ingest_ai_output
from analyst_forecast.application.settings import AppSettings
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


def import_locked_component(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
    *,
    direction: str = "up",
    label: str = "main",
) -> str:
    """P05→P08→P11→P12でmapping固定済みのcomponent IDを返す。"""
    from analyst_forecast.infrastructure.db.models import AnalystRecord, RunRecord
    from analyst_forecast.infrastructure.db.session import create_session_factory

    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        run = session.get(RunRecord, run_result.run_id)
        assert run is not None
        analyst = session.get(AnalystRecord, run.analyst_id)
        assert analyst is not None
        speaker_name = analyst.canonical_name

    quote = "日経平均は今後上昇する"
    p05 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            f"p05-{label}.json",
            {
                "schema_version": "2.0.0",
                "run_id": run_result.run_id,
                "source_id": source_result.source_id,
                "prompt_execution": _prompt("P05"),
                "input_hash": source_result.raw_hash,
                "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                "segments": [
                    {
                        "segment_ref": f"segment-{label}",
                        "sequence_number": 1,
                        "raw_start_offset": 0,
                        "raw_end_offset": len(RAW_TEXT),
                        "raw_text": RAW_TEXT,
                        "normalized_text": RAW_TEXT,
                        "speaker_status": "identified",
                        "speaker_candidate": speaker_name,
                        "speaker_confidence": 0.95,
                        "attribution_basis": "fixtureメタデータと発言者が一致",
                        "review_status": "accepted",
                        "importance": "normal",
                        "high_importance_reason": None,
                    }
                ],
            },
        ),
    )
    assert p05.status is AiIngestStatus.ACCEPTED, p05.issues

    p08 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            f"p08-{label}.json",
            {
                "schema_version": "2.0.0",
                "run_id": run_result.run_id,
                "source_id": source_result.source_id,
                "p05_artifact_id": p05.artifact_ids[0],
                "prompt_execution": _prompt("P08"),
                "input_hash": p05.output_hash,
                "processing_status": "processed_with_forecasts",
                "forecasts": [
                    {
                        "forecast_ref": f"forecast-{label}",
                        "forecast_group_ref": f"group-{label}",
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
                        "upstream_segment_refs": [f"segment-{label}"],
                        "speaker_candidate": speaker_name,
                        "speaker_attribution_status": "target_confirmed",
                        "attribution_confidence": 0.95,
                        "attribution_basis": "本人segment",
                        "statement_kind": "direct_statement",
                        "evidence": [
                            {
                                "source_id": source_result.source_id,
                                "quote": quote,
                                "start_offset": 0,
                                "end_offset": len(quote),
                                "role": "prediction",
                            }
                        ],
                        "components": [
                            {
                                "component_ref": f"component-{label}",
                                "sequence_number": 1,
                                "prediction_form": "period_direction",
                                "direction": direction,
                                "time_source": "explicit",
                                "time_expression_raw": "今後3か月",
                                "normalized_start": "2026-01-13",
                                "normalized_end": "2026-04-13",
                                "raw_target_label": "日経平均",
                                "target_resolution_status": "pending",
                            }
                        ],
                    }
                ],
            },
        ),
    )
    assert p08.status is AiIngestStatus.ACCEPTED, p08.issues
    component_id = p08.component_ids[0]

    p11 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            f"p11-{label}.json",
            {
                "schema_version": "2.0.0",
                "run_id": run_result.run_id,
                "source_id": source_result.source_id,
                "forecast_component_id": component_id,
                "prompt_execution": _prompt("P11"),
                "input_hash": p08.output_hash,
                "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                "resolution_status": "proposed",
                "candidates": [
                    {
                        "candidate_ref": "candidate-1",
                        "rank": 1,
                        "canonical_name": "日経平均株価",
                        "target_type": "index",
                        "mapping_method": "explicit",
                        "instruments": [
                            {
                                "symbol": "^N225",
                                "exchange": "JPX",
                                "currency": "JPY",
                                "weight": 1.0,
                            }
                        ],
                        "existed_at": "2026-01-10",
                        "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                        "source_evidence": "原文で明示",
                        "confidence": 0.95,
                    }
                ],
                "unevaluable_reason": None,
            },
        ),
    )
    assert p11.status is AiIngestStatus.ACCEPTED, p11.issues

    p12 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            f"p12-{label}.json",
            {
                "schema_version": "2.0.0",
                "run_id": run_result.run_id,
                "source_id": source_result.source_id,
                "forecast_component_id": component_id,
                "proposal_artifact_id": p11.artifact_ids[0],
                "prompt_execution": _prompt("P12"),
                "input_hash": p11.output_hash,
                "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                "resolution_status": "agreed",
                "reviews": [
                    {
                        "candidate_ref": "candidate-1",
                        "decision": "accept",
                        "confidence": 0.92,
                        "rationale": "原文と当時資料を独立確認",
                    }
                ],
                "recommended_candidate_ref": "candidate-1",
                "unevaluable_reason": None,
            },
        ),
    )
    assert p12.status is AiIngestStatus.ACCEPTED, p12.issues
    return component_id
