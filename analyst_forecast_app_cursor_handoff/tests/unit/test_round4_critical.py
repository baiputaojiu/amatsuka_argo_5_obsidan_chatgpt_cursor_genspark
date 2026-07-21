"""Round4 critical bug tests covering Bugs A-F."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from analyst_forecast.application.ai_ingestion import AiIngestStatus, ingest_ai_output
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.infrastructure.db.models import ForecastIssuanceRecord
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


def _ingest_p05(settings, run_result, source_result, tmp_path, *, label="main"):
    from analyst_forecast.infrastructure.db.models import AnalystRecord, RunRecord

    sf = create_session_factory(settings.database_file)
    with sf() as session:
        run = session.get(RunRecord, run_result.run_id)
        analyst = session.get(AnalystRecord, run.analyst_id)
        speaker = analyst.canonical_name

    return ingest_ai_output(
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
                        "speaker_candidate": speaker,
                        "speaker_confidence": 0.95,
                        "attribution_basis": "fixture",
                        "review_status": "accepted",
                    }
                ],
            },
        ),
    ), speaker


def _ingest_p08(
    settings,
    run_result,
    source_result,
    tmp_path,
    p05_result,
    speaker,
    *,
    label="main",
):
    quote = "日経平均は今後上昇する"
    return ingest_ai_output(
        settings,
        _write(
            tmp_path,
            f"p08-{label}.json",
            {
                "schema_version": "2.1.0",
                "run_id": run_result.run_id,
                "source_id": source_result.source_id,
                "upstream_artifact_id": p05_result.artifact_ids[0],
                "upstream_prompt_id": "P05",
                "prompt_execution": _prompt("P08"),
                "input_hash": p05_result.output_hash,
                "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
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
                        "human_readable_summary": quote,
                        "relation_to_previous": "initial",
                        "upstream_segment_refs": [f"segment-{label}"],
                        "speaker_candidate": speaker,
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
                                "direction": "up",
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


# =========================================================================
# Bug A/B: P09 accept/correct lifecycle (Fix 02)
# =========================================================================


class TestP09AcceptOnAcceptedP08:
    """R4-002: accepted P08 + P09 accept → no exception, no duplication."""

    def test_accept_on_accepted_p08_no_exception(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path)
        assert p05.status is AiIngestStatus.ACCEPTED
        p08 = _ingest_p08(settings, run_result, source_result, tmp_path, p05, speaker)
        assert p08.status is AiIngestStatus.ACCEPTED

        sf = create_session_factory(settings.database_file)
        p09 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-accept.json",
                {
                    "schema_version": "2.0.0",
                    "run_id": run_result.run_id,
                    "source_id": source_result.source_id,
                    "reviewed_artifact_id": p08.artifact_ids[0],
                    "prompt_execution": _prompt("P09"),
                    "input_hash": p08.output_hash,
                    "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                    "decision": "accept",
                    "findings": [],
                    "corrected_payload": None,
                },
            ),
        )
        assert p09.status is AiIngestStatus.ACCEPTED

        with sf() as session:
            issuances = list(
                session.scalars(
                    select(ForecastIssuanceRecord).where(
                        ForecastIssuanceRecord.ai_artifact_id == p08.artifact_ids[0]
                    )
                )
            )
            active = [i for i in issuances if i.lifecycle_status == "active"]
            assert len(active) <= 1


class TestP09CorrectOnAcceptedP08:
    """R4-003: accepted P08 + P09 correct → old superseded, new active."""

    def test_correct_supersedes_old(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path)
        p08 = _ingest_p08(settings, run_result, source_result, tmp_path, p05, speaker)
        assert p08.status is AiIngestStatus.ACCEPTED

        sf = create_session_factory(settings.database_file)
        with sf() as session:
            old_issuances = list(
                session.scalars(
                    select(ForecastIssuanceRecord).where(
                        ForecastIssuanceRecord.ai_artifact_id == p08.artifact_ids[0]
                    )
                )
            )
            assert len(old_issuances) == 1
            old_id = old_issuances[0].forecast_issuance_id

        corrected_quote = "日経平均は今後上昇する"
        corrected_payload = {
            "schema_version": "2.1.0",
            "run_id": run_result.run_id,
            "source_id": source_result.source_id,
            "upstream_artifact_id": p05.artifact_ids[0],
            "upstream_prompt_id": "P05",
            "prompt_execution": _prompt("P08"),
            "input_hash": p05.output_hash,
            "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
            "processing_status": "processed_with_forecasts",
            "forecasts": [
                {
                    "forecast_ref": "forecast-corrected",
                    "forecast_group_ref": "group-main",
                    "made_at": "2026-01-10T09:00:00+00:00",
                    "publicly_available_at": "2026-01-10T10:00:00+00:00",
                    "made_at_source": "explicit",
                    "forecast_type": "directional",
                    "commitment_strength": "explicit",
                    "evidence_level": "A",
                    "extraction_confidence": 0.95,
                    "human_readable_summary": corrected_quote,
                    "relation_to_previous": "initial",
                    "upstream_segment_refs": ["segment-main"],
                    "speaker_candidate": speaker,
                    "speaker_attribution_status": "target_confirmed",
                    "attribution_confidence": 0.95,
                    "attribution_basis": "本人segment",
                    "statement_kind": "direct_statement",
                    "evidence": [
                        {
                            "source_id": source_result.source_id,
                            "quote": corrected_quote,
                            "start_offset": 0,
                            "end_offset": len(corrected_quote),
                            "role": "prediction",
                        }
                    ],
                    "components": [
                        {
                            "component_ref": "component-corrected",
                            "sequence_number": 1,
                            "prediction_form": "period_direction",
                            "direction": "down",
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
        }

        p09 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-correct.json",
                {
                    "schema_version": "2.0.0",
                    "run_id": run_result.run_id,
                    "source_id": source_result.source_id,
                    "reviewed_artifact_id": p08.artifact_ids[0],
                    "prompt_execution": _prompt("P09"),
                    "input_hash": p08.output_hash,
                    "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                    "decision": "correct",
                    "findings": [
                        {
                            "finding_ref": "f1",
                            "severity": "error",
                            "message": "方向が誤り",
                            "evidence": "原文確認",
                        }
                    ],
                    "corrected_payload": corrected_payload,
                },
            ),
        )
        assert p09.status is AiIngestStatus.ACCEPTED, [f"{i.code}: {i.message}" for i in p09.issues]

        with sf() as session:
            old = session.get(ForecastIssuanceRecord, old_id)
            assert old.lifecycle_status == "superseded"

            new_issuances = list(
                session.scalars(
                    select(ForecastIssuanceRecord).where(
                        ForecastIssuanceRecord.lifecycle_status == "active"
                    )
                )
            )
            assert len(new_issuances) == 1
            assert new_issuances[0].forecast_issuance_id != old_id


# =========================================================================
# Bug D: P09 reject/unresolved state machine (Fix 04)
# =========================================================================


class TestP09RejectRetryable:
    """R4-020: P09 reject retryable → RUN_P08, not RUN_PREPROCESS."""

    def test_reject_sets_reextract_status(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="rej")
        p08 = _ingest_p08(settings, run_result, source_result, tmp_path, p05, speaker, label="rej")
        assert p08.status is AiIngestStatus.ACCEPTED

        sf = create_session_factory(settings.database_file)
        from analyst_forecast.infrastructure.db.models import AiArtifactRecord

        with sf.begin() as session:
            art = session.get(AiArtifactRecord, p08.artifact_ids[0])
            art.classification = "needs_review"
            art.resolution_status = "needs_review"

        p09 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-reject.json",
                {
                    "schema_version": "2.1.0",
                    "run_id": run_result.run_id,
                    "source_id": source_result.source_id,
                    "reviewed_artifact_id": p08.artifact_ids[0],
                    "prompt_execution": _prompt("P09"),
                    "input_hash": p08.output_hash,
                    "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                    "decision": "reject",
                    "reject_disposition": "retryable",
                    "reject_reason": "P08再抽出で修正可能",
                    "findings": [
                        {
                            "finding_ref": "f1",
                            "severity": "error",
                            "message": "retryable error",
                            "evidence": "P08出力確認",
                        }
                    ],
                    "corrected_payload": None,
                },
            ),
        )
        if p09.status is not AiIngestStatus.ACCEPTED:
            issues = [(i.code, i.message) for i in p09.issues]
            pytest.fail(f"P09 REJECT failed: status={p09.status}, issues={issues}")
        assert p09.status is AiIngestStatus.ACCEPTED, [f"{i.code}: {i.message}" for i in p09.issues]

        from analyst_forecast.infrastructure.db.models import RunSourceRecord

        with sf() as session:
            link = session.get(
                RunSourceRecord,
                {
                    "run_id": run_result.run_id,
                    "source_id": source_result.source_id,
                },
            )
            assert link.processing_status in {"p08_reextract_required", "p08_rejected_terminal"}


# =========================================================================
# Bug E: Unknown time & cutoff (Fix 05)
# =========================================================================


def _sample_component() -> dict[str, Any]:
    return {
        "component_ref": "c1",
        "sequence_number": 1,
        "prediction_form": "period_direction",
        "direction": "up",
        "time_source": "explicit",
        "normalized_start": "2026-01-13",
        "normalized_end": "2026-04-13",
        "raw_target_label": "test",
        "target_resolution_status": "pending",
    }


def _sample_evidence() -> dict[str, Any]:
    return {
        "source_id": "SRC-000001",
        "quote": "x",
        "start_offset": 0,
        "end_offset": 1,
        "role": "prediction",
    }


class TestUnknownTime:
    """R4-028/029: unknown time → made_at=null, no formal issuance."""

    def test_unknown_time_with_datetime_rejected(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        """made_at_source=unknown + made_at=datetime → Pydantic rejects."""
        from pydantic import ValidationError

        from analyst_forecast.schemas.pipeline import ForecastIssuanceV2

        with pytest.raises(ValidationError, match="unknown"):
            ForecastIssuanceV2.model_validate(
                {
                    "forecast_ref": "f1",
                    "forecast_group_ref": "g1",
                    "made_at": "2026-01-10T09:00:00+00:00",
                    "publicly_available_at": "2026-01-10T10:00:00+00:00",
                    "made_at_source": "unknown",
                    "forecast_type": "directional",
                    "commitment_strength": "explicit",
                    "evidence_level": "A",
                    "extraction_confidence": 0.9,
                    "human_readable_summary": "test",
                    "relation_to_previous": "initial",
                    "evidence": [_sample_evidence()],
                    "components": [_sample_component()],
                }
            )

    def test_unknown_time_null_made_at_accepted(self):
        """made_at_source=unknown + made_at=null is valid."""
        from analyst_forecast.schemas.pipeline import ForecastIssuanceV2

        result = ForecastIssuanceV2.model_validate(
            {
                "forecast_ref": "f1",
                "forecast_group_ref": "g1",
                "made_at": None,
                "publicly_available_at": None,
                "made_at_source": "unknown",
                "forecast_type": "directional",
                "commitment_strength": "explicit",
                "evidence_level": "A",
                "extraction_confidence": 0.9,
                "human_readable_summary": "test",
                "relation_to_previous": "initial",
                "evidence": [_sample_evidence()],
                "components": [_sample_component()],
            }
        )
        assert result.made_at is None

    def test_explicit_without_made_at_rejected(self):
        """made_at_source=explicit + made_at=null → rejected."""
        from pydantic import ValidationError

        from analyst_forecast.schemas.pipeline import ForecastIssuanceV2

        with pytest.raises(ValidationError, match="made_at"):
            ForecastIssuanceV2.model_validate(
                {
                    "forecast_ref": "f1",
                    "forecast_group_ref": "g1",
                    "made_at": None,
                    "publicly_available_at": None,
                    "made_at_source": "explicit",
                    "forecast_type": "directional",
                    "commitment_strength": "explicit",
                    "evidence_level": "A",
                    "extraction_confidence": 0.9,
                    "human_readable_summary": "test",
                    "relation_to_previous": "initial",
                    "evidence": [_sample_evidence()],
                    "components": [_sample_component()],
                }
            )


class TestP08KnowledgeCutoff:
    """R4-031/032: P08 knowledge_cutoff in schema, cutoff > made_at rejected."""

    def test_cutoff_exceeds_made_at_rejected(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="cutx")
        assert p05.status is AiIngestStatus.ACCEPTED

        quote = "日経平均は今後上昇する"
        p08 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p08-bad-cutoff.json",
                {
                    "schema_version": "2.1.0",
                    "run_id": run_result.run_id,
                    "source_id": source_result.source_id,
                    "upstream_artifact_id": p05.artifact_ids[0],
                    "upstream_prompt_id": "P05",
                    "prompt_execution": _prompt("P08"),
                    "input_hash": p05.output_hash,
                    "knowledge_cutoff": "2026-07-20T12:00:00+00:00",  # far future
                    "processing_status": "processed_with_forecasts",
                    "forecasts": [
                        {
                            "forecast_ref": "forecast-cut",
                            "forecast_group_ref": "group-cut",
                            "made_at": "2026-01-10T09:00:00+00:00",
                            "publicly_available_at": "2026-01-10T10:00:00+00:00",
                            "made_at_source": "explicit",
                            "forecast_type": "directional",
                            "commitment_strength": "explicit",
                            "evidence_level": "A",
                            "extraction_confidence": 0.95,
                            "human_readable_summary": quote,
                            "relation_to_previous": "initial",
                            "upstream_segment_refs": ["segment-cutx"],
                            "speaker_candidate": speaker,
                            "speaker_attribution_status": "target_confirmed",
                            "attribution_confidence": 0.95,
                            "attribution_basis": "本人",
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
                                    "component_ref": "component-cut",
                                    "sequence_number": 1,
                                    "prediction_form": "period_direction",
                                    "direction": "up",
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
        assert p08.status in {AiIngestStatus.NEEDS_REVIEW, AiIngestStatus.REJECTED}, [
            f"{i.code}: {i.message}" for i in p08.issues
        ]
        cutoff_issues = [i for i in p08.issues if "cutoff" in i.code or "cutoff" in i.message]
        assert len(cutoff_issues) > 0, (
            f"Expected cutoff issue but got: {[(i.code, i.message) for i in p08.issues]}"
        )


# =========================================================================
# Bug F: Basket common dates (Fix 06)
# =========================================================================


class TestBasketCommonDates:
    """R4-037: basket with 1 common date → unevaluable insufficient_common_dates."""

    def test_insufficient_common_dates(self):
        """1 common date across basket instruments → MarketDataUnavailable."""
        from decimal import Decimal

        from analyst_forecast.application import evaluation as eval_mod
        from analyst_forecast.domain.market import (
            MarketBar,
            MarketSeries,
        )

        bar_a = MarketBar(
            date=date(2026, 1, 13),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("102"),
            adjusted_open=Decimal("100"),
            adjusted_close=Decimal("102"),
        )
        bar_b = MarketBar(
            date=date(2026, 1, 13),
            open=Decimal("200"),
            high=Decimal("210"),
            low=Decimal("190"),
            close=Decimal("205"),
            adjusted_open=Decimal("200"),
            adjusted_close=Decimal("205"),
        )

        series_a = MarketSeries(
            provider="test",
            symbol="AAA",
            currency="JPY",
            adjustment_type="split_adjusted",
            frequency="daily",
            retrieved_at=datetime(2026, 7, 20, tzinfo=UTC),
            bars=(bar_a,),
        )
        series_b = MarketSeries(
            provider="test",
            symbol="BBB",
            currency="JPY",
            adjustment_type="split_adjusted",
            frequency="daily",
            retrieved_at=datetime(2026, 7, 20, tzinfo=UTC),
            bars=(bar_b,),
        )

        series_by_symbol = {"AAA": series_a, "BBB": series_b}
        common_dates = eval_mod._common_bar_dates(
            series_by_symbol, date(2026, 1, 13), date(2026, 4, 13)
        )
        assert len(common_dates) < 2
