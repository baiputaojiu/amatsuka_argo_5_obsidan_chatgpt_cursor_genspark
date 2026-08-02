"""Round4 acceptance matrix tests (R4-001..041 subset)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select

from analyst_forecast.application.ai_ingestion import AiIngestStatus, ingest_ai_output
from analyst_forecast.application.artifact_reuse import (
    can_reuse_processed_artifact,
    is_artifact_applicable_for_source,
    reuse_artifact_for_source,
)
from analyst_forecast.application.evaluation import evaluate_component
from analyst_forecast.application.raw_sources import RawSourceRequest, import_raw_source
from analyst_forecast.application.results import generate_run_results
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.application.workflow import refresh_workflow
from analyst_forecast.domain.market import (
    MarketBar,
    MarketDataRequest,
    MarketDataUnavailable,
    MarketSeries,
)
from analyst_forecast.domain.models import Medium
from analyst_forecast.infrastructure.db.models import (
    ArtifactApplicabilityRecord,
    EvaluationRecord,
    ForecastComponentRecord,
    ForecastIssuanceRecord,
    RunSourceRecord,
    SourceRecord,
    TargetMappingRecord,
)
from analyst_forecast.infrastructure.db.session import create_session_factory
from analyst_forecast.schemas.pipeline import P08Output
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


def _speaker(settings: AppSettings, run_id: str) -> str:
    from analyst_forecast.infrastructure.db.models import AnalystRecord, RunRecord

    sf = create_session_factory(settings.database_file)
    with sf() as session:
        run = session.get(RunRecord, run_id)
        assert run is not None
        analyst = session.get(AnalystRecord, run.analyst_id)
        assert analyst is not None
        return analyst.canonical_name


def _ingest_p05(settings, run_result, source_result, tmp_path, *, label="main"):
    speaker = _speaker(settings, run_result.run_id)
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


def _p08_payload(
    *,
    run_id: str,
    source_id: str,
    upstream_artifact_id: str,
    input_hash: str,
    speaker: str,
    label: str = "main",
    segment_label: str | None = None,
    confidence: float = 0.95,
    made_at: str | None = "2026-01-10T09:00:00+00:00",
    made_at_source: str = "explicit",
    publicly_available_at: str | None = "2026-01-10T10:00:00+00:00",
    knowledge_cutoff: str | None = "2026-01-10T09:00:00+00:00",
    schema_version: str = "2.1.0",
    direction: str = "up",
) -> dict[str, Any]:
    quote = "日経平均は今後上昇する"
    seg = segment_label or label
    forecast: dict[str, Any] = {
        "forecast_ref": f"forecast-{label}",
        "forecast_group_ref": f"group-{label}",
        "made_at": made_at,
        "publicly_available_at": publicly_available_at,
        "made_at_source": made_at_source,
        "forecast_type": "directional",
        "commitment_strength": "explicit",
        "evidence_level": "A",
        "extraction_confidence": confidence,
        "human_readable_summary": quote,
        "relation_to_previous": "initial",
        "upstream_segment_refs": [f"segment-{seg}"],
        "speaker_candidate": speaker,
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
    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "run_id": run_id,
        "source_id": source_id,
        "upstream_artifact_id": upstream_artifact_id,
        "upstream_prompt_id": "P05",
        "prompt_execution": _prompt("P08"),
        "input_hash": input_hash,
        "processing_status": "processed_with_forecasts",
        "forecasts": [forecast],
    }
    if knowledge_cutoff is not None:
        payload["knowledge_cutoff"] = knowledge_cutoff
    return payload


def _ingest_p08(
    settings,
    run_result,
    source_result,
    tmp_path,
    p05_result,
    speaker,
    *,
    label="main",
    confidence: float = 0.95,
    made_at: str | None = "2026-01-10T09:00:00+00:00",
    made_at_source: str = "explicit",
    publicly_available_at: str | None = "2026-01-10T10:00:00+00:00",
    knowledge_cutoff: str | None = "2026-01-10T09:00:00+00:00",
    direction: str = "up",
):
    return ingest_ai_output(
        settings,
        _write(
            tmp_path,
            f"p08-{label}.json",
            _p08_payload(
                run_id=run_result.run_id,
                source_id=source_result.source_id,
                upstream_artifact_id=p05_result.artifact_ids[0],
                input_hash=p05_result.output_hash,
                speaker=speaker,
                label=label,
                confidence=confidence,
                made_at=made_at,
                made_at_source=made_at_source,
                publicly_available_at=publicly_available_at,
                knowledge_cutoff=knowledge_cutoff,
                direction=direction,
            ),
        ),
    )


def _corrected_p08(
    *,
    run_id: str,
    source_id: str,
    upstream_artifact_id: str,
    input_hash: str,
    speaker: str,
    label: str = "corrected",
    segment_label: str | None = None,
    direction: str = "down",
    made_at: str | None = "2026-01-10T09:00:00+00:00",
    made_at_source: str = "explicit",
    publicly_available_at: str | None = "2026-01-10T10:00:00+00:00",
    knowledge_cutoff: str = "2026-01-10T09:00:00+00:00",
    confidence: float = 0.95,
) -> dict[str, Any]:
    return _p08_payload(
        run_id=run_id,
        source_id=source_id,
        upstream_artifact_id=upstream_artifact_id,
        input_hash=input_hash,
        speaker=speaker,
        label=label,
        segment_label=segment_label or label,
        confidence=confidence,
        made_at=made_at,
        made_at_source=made_at_source,
        publicly_available_at=publicly_available_at,
        knowledge_cutoff=knowledge_cutoff,
        direction=direction,
    )


def _p09(
    *,
    run_id: str,
    source_id: str,
    reviewed_artifact_id: str,
    input_hash: str,
    decision: str,
    label: str = "p09",
    findings: list[dict[str, Any]] | None = None,
    corrected_payload: dict[str, Any] | None = None,
    knowledge_cutoff: str = "2026-01-10T09:00:00+00:00",
    reject_terminal: bool = False,
    reject_reason: str | None = None,
    reject_disposition: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "2.1.0",
        "run_id": run_id,
        "source_id": source_id,
        "reviewed_artifact_id": reviewed_artifact_id,
        "prompt_execution": _prompt("P09"),
        "input_hash": input_hash,
        "knowledge_cutoff": knowledge_cutoff,
        "decision": decision,
        "findings": findings or [],
        "corrected_payload": corrected_payload,
    }
    if decision == "reject":
        disposition = reject_disposition
        if disposition is None:
            disposition = "terminal" if reject_terminal else "retryable"
        payload["reject_disposition"] = disposition
        payload["reject_reason"] = reject_reason or (
            "terminal reject" if disposition == "terminal" else "retryable reject"
        )
    return payload


def _active_issuances(session) -> list[ForecastIssuanceRecord]:
    return list(
        session.scalars(
            select(ForecastIssuanceRecord).where(
                ForecastIssuanceRecord.lifecycle_status == "active"
            )
        )
    )


def _mark_p08_needs_review(settings: AppSettings, artifact_id: str) -> None:
    from analyst_forecast.infrastructure.db.models import AiArtifactRecord

    sf = create_session_factory(settings.database_file)
    with sf.begin() as session:
        art = session.get(AiArtifactRecord, artifact_id)
        assert art is not None
        art.classification = "needs_review"
        art.resolution_status = "needs_review"


def _import_second_source(
    settings: AppSettings,
    run_id: str,
    tmp_path: Path,
    *,
    url: str,
    text: str = RAW_TEXT,
    label: str = "src2",
):
    path = tmp_path / f"{label}.txt"
    path.write_text(text, encoding="utf-8")
    return import_raw_source(
        settings,
        RawSourceRequest(
            run_id=run_id,
            input_path=path,
            medium=Medium.YOUTUBE,
            url=url,
            title=label,
            recorded_at=datetime(2026, 1, 10, 9, 0, tzinfo=UTC),
            published_at=datetime(2026, 1, 10, 10, 0, tzinfo=UTC),
            retrieved_at=datetime(2026, 7, 20, 12, 30, tzinfo=UTC),
        ),
    )


def _set_basket_instruments(settings: AppSettings, component_id: str) -> None:
    sf = create_session_factory(settings.database_file)
    with sf.begin() as session:
        component = session.get(ForecastComponentRecord, component_id)
        assert component is not None
        mapping = session.get(TargetMappingRecord, component.target_mapping_id)
        assert mapping is not None
        mapping.evaluation_instruments = [
            {"symbol": "AAA", "currency": "JPY", "weight": 0.5, "exchange": None},
            {"symbol": "BBB", "currency": "JPY", "weight": 0.5, "exchange": None},
        ]
        mapping.weights = [0.5, 0.5]


def _bar(d: date, open_: str, close: str) -> MarketBar:
    o = Decimal(open_)
    c = Decimal(close)
    return MarketBar(
        date=d,
        open=o,
        high=max(o, c),
        low=min(o, c),
        close=c,
        adjusted_open=o,
        adjusted_close=c,
    )


class SymbolProvider:
    name = "basket-fixture"

    def __init__(self, series: dict[str, MarketSeries]) -> None:
        self.series = series

    def fetch(self, request: MarketDataRequest) -> MarketSeries:
        if request.symbol not in self.series:
            raise MarketDataUnavailable(f"missing symbol {request.symbol}")
        return self.series[request.symbol]


def _series(symbol: str, bars: tuple[MarketBar, ...]) -> MarketSeries:
    return MarketSeries(
        provider="basket-fixture",
        symbol=symbol,
        currency="JPY",
        adjustment_type="split_adjusted_ohlc",
        frequency="1d",
        retrieved_at=datetime(2026, 7, 20, 13, tzinfo=UTC),
        bars=bars,
    )


# =============================================================================
# Lifecycle
# =============================================================================


class TestR4001NeedsReviewAccept:
    """R4-001: P08 low confidence → NEEDS_REVIEW; P09 accept → 1 active issuance."""

    def test_r4_001_needs_review_then_accept(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="r4001")
        p08 = _ingest_p08(
            settings,
            run_result,
            source_result,
            tmp_path,
            p05,
            speaker,
            label="r4001",
            confidence=0.3,
        )
        assert p08.status is AiIngestStatus.NEEDS_REVIEW
        assert not p08.forecast_issuance_ids

        p09 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-r4001.json",
                _p09(
                    run_id=run_result.run_id,
                    source_id=source_result.source_id,
                    reviewed_artifact_id=p08.artifact_ids[0],
                    input_hash=p08.output_hash,
                    decision="accept",
                ),
            ),
        )
        assert p09.status is AiIngestStatus.ACCEPTED, [f"{i.code}: {i.message}" for i in p09.issues]

        sf = create_session_factory(settings.database_file)
        with sf() as session:
            active = _active_issuances(session)
            assert len(active) == 1


class TestR4004LineageActive:
    """R4-004: after correct, only 1 active in lineage (generation, supersedes)."""

    def test_r4_004_lineage_single_active(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="r4004")
        p08 = _ingest_p08(
            settings, run_result, source_result, tmp_path, p05, speaker, label="r4004"
        )
        assert p08.status is AiIngestStatus.ACCEPTED
        old_id = p08.forecast_issuance_ids[0]

        corrected = _corrected_p08(
            run_id=run_result.run_id,
            source_id=source_result.source_id,
            upstream_artifact_id=p05.artifact_ids[0],
            input_hash=p05.output_hash,
            speaker=speaker,
            label="r4004c",
            segment_label="r4004",
        )
        p09 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-r4004.json",
                _p09(
                    run_id=run_result.run_id,
                    source_id=source_result.source_id,
                    reviewed_artifact_id=p08.artifact_ids[0],
                    input_hash=p08.output_hash,
                    decision="correct",
                    findings=[
                        {
                            "finding_ref": "f1",
                            "severity": "error",
                            "message": "方向訂正",
                            "evidence": "原文",
                        }
                    ],
                    corrected_payload=corrected,
                ),
            ),
        )
        assert p09.status is AiIngestStatus.ACCEPTED, [f"{i.code}: {i.message}" for i in p09.issues]

        sf = create_session_factory(settings.database_file)
        with sf() as session:
            old = session.get(ForecastIssuanceRecord, old_id)
            assert old is not None
            assert old.lifecycle_status == "superseded"
            active = _active_issuances(session)
            assert len(active) == 1
            new = active[0]
            assert new.generation == (old.generation or 1) + 1
            assert new.supersedes_forecast_issuance_id == old_id
            root = old.lineage_root_id or old.forecast_issuance_id
            assert new.lineage_root_id == root
            lineage_active = [
                i
                for i in session.scalars(
                    select(ForecastIssuanceRecord).where(
                        (ForecastIssuanceRecord.lineage_root_id == root)
                        | (ForecastIssuanceRecord.forecast_issuance_id == root)
                    )
                )
                if i.lifecycle_status == "active"
            ]
            assert len(lineage_active) == 1


class TestR4005CorrectIdempotent:
    """R4-005: re-ingest same P09 correct → ALREADY_IMPORTED; count unchanged."""

    def test_r4_005_already_imported(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="r4005")
        p08 = _ingest_p08(
            settings, run_result, source_result, tmp_path, p05, speaker, label="r4005"
        )
        corrected = _corrected_p08(
            run_id=run_result.run_id,
            source_id=source_result.source_id,
            upstream_artifact_id=p05.artifact_ids[0],
            input_hash=p05.output_hash,
            speaker=speaker,
            label="r4005c",
            segment_label="r4005",
        )
        payload = _p09(
            run_id=run_result.run_id,
            source_id=source_result.source_id,
            reviewed_artifact_id=p08.artifact_ids[0],
            input_hash=p08.output_hash,
            decision="correct",
            findings=[
                {
                    "finding_ref": "f1",
                    "severity": "error",
                    "message": "訂正",
                    "evidence": "原文",
                }
            ],
            corrected_payload=corrected,
        )
        path = _write(tmp_path, "p09-r4005.json", payload)
        first = ingest_ai_output(settings, path)
        assert first.status is AiIngestStatus.ACCEPTED

        sf = create_session_factory(settings.database_file)
        with sf() as session:
            before = session.scalar(select(func.count()).select_from(ForecastIssuanceRecord))

        second = ingest_ai_output(settings, path)
        assert second.status is AiIngestStatus.ALREADY_IMPORTED

        with sf() as session:
            after = session.scalar(select(func.count()).select_from(ForecastIssuanceRecord))
            assert after == before
            assert len(_active_issuances(session)) == 1


class TestR4006SupersededRetained:
    """R4-006: superseded issuance and components remain in DB."""

    def test_r4_006_history_retained(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="r4006")
        p08 = _ingest_p08(
            settings, run_result, source_result, tmp_path, p05, speaker, label="r4006"
        )
        old_id = p08.forecast_issuance_ids[0]
        old_comp = p08.component_ids[0]

        corrected = _corrected_p08(
            run_id=run_result.run_id,
            source_id=source_result.source_id,
            upstream_artifact_id=p05.artifact_ids[0],
            input_hash=p05.output_hash,
            speaker=speaker,
            label="r4006c",
            segment_label="r4006",
        )
        p09 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-r4006.json",
                _p09(
                    run_id=run_result.run_id,
                    source_id=source_result.source_id,
                    reviewed_artifact_id=p08.artifact_ids[0],
                    input_hash=p08.output_hash,
                    decision="correct",
                    findings=[
                        {
                            "finding_ref": "f1",
                            "severity": "error",
                            "message": "訂正",
                            "evidence": "原文",
                        }
                    ],
                    corrected_payload=corrected,
                ),
            ),
        )
        assert p09.status is AiIngestStatus.ACCEPTED

        sf = create_session_factory(settings.database_file)
        with sf() as session:
            old = session.get(ForecastIssuanceRecord, old_id)
            assert old is not None
            assert old.lifecycle_status == "superseded"
            comps = list(
                session.scalars(
                    select(ForecastComponentRecord).where(
                        ForecastComponentRecord.forecast_issuance_id == old_id
                    )
                )
            )
            assert any(c.forecast_component_id == old_comp for c in comps)


class TestR4007ResultsExcludeSuperseded:
    """R4-007: generate_run_results excludes superseded from active forecasts."""

    def test_r4_007_results_active_only(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="r4007")
        p08 = _ingest_p08(
            settings, run_result, source_result, tmp_path, p05, speaker, label="r4007"
        )
        old_id = p08.forecast_issuance_ids[0]
        corrected = _corrected_p08(
            run_id=run_result.run_id,
            source_id=source_result.source_id,
            upstream_artifact_id=p05.artifact_ids[0],
            input_hash=p05.output_hash,
            speaker=speaker,
            label="r4007c",
            segment_label="r4007",
        )
        ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-r4007.json",
                _p09(
                    run_id=run_result.run_id,
                    source_id=source_result.source_id,
                    reviewed_artifact_id=p08.artifact_ids[0],
                    input_hash=p08.output_hash,
                    decision="correct",
                    findings=[
                        {
                            "finding_ref": "f1",
                            "severity": "error",
                            "message": "訂正",
                            "evidence": "原文",
                        }
                    ],
                    corrected_payload=corrected,
                ),
            ),
        )
        paths = generate_run_results(settings, run_result.run_id)
        csv_text = paths["forecasts_csv"].read_text(encoding="utf-8")
        md_text = paths["forecasts_md"].read_text(encoding="utf-8")
        assert old_id not in csv_text
        assert old_id not in md_text

        sf = create_session_factory(settings.database_file)
        with sf() as session:
            active = _active_issuances(session)
            assert len(active) == 1
            assert active[0].forecast_issuance_id in csv_text


class TestR4009SummaryActiveCountOnly:
    """R4-009: summary forecast count uses active generation only (not superseded)."""

    def test_r4_009_summary_forecast_count_active_only(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="r4009")
        p08 = _ingest_p08(
            settings, run_result, source_result, tmp_path, p05, speaker, label="r4009"
        )
        corrected = _corrected_p08(
            run_id=run_result.run_id,
            source_id=source_result.source_id,
            upstream_artifact_id=p05.artifact_ids[0],
            input_hash=p05.output_hash,
            speaker=speaker,
            label="r4009c",
            segment_label="r4009",
        )
        p09 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-r4009.json",
                _p09(
                    run_id=run_result.run_id,
                    source_id=source_result.source_id,
                    reviewed_artifact_id=p08.artifact_ids[0],
                    input_hash=p08.output_hash,
                    decision="correct",
                    findings=[
                        {
                            "finding_ref": "f1",
                            "severity": "error",
                            "message": "訂正",
                            "evidence": "原文",
                        }
                    ],
                    corrected_payload=corrected,
                ),
            ),
        )
        assert p09.status is AiIngestStatus.ACCEPTED, [f"{i.code}: {i.message}" for i in p09.issues]

        sf = create_session_factory(settings.database_file)
        with sf() as session:
            total = session.scalar(select(func.count()).select_from(ForecastIssuanceRecord))
            assert total == 2
            assert len(_active_issuances(session)) == 1

        paths = generate_run_results(settings, run_result.run_id)
        summary = paths["summary_md"].read_text(encoding="utf-8")
        assert "- 予想構成数: 1\n" in summary
        assert "- 予想構成数: 2\n" not in summary


class TestR4008EvalRefusesSuperseded:
    """R4-008: evaluate_component on superseded component refuses."""

    def test_r4_008_superseded_eval_refused(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        component_id = import_locked_component(
            settings, run_result, source_result, tmp_path, label="r4008"
        )
        sf = create_session_factory(settings.database_file)
        with sf.begin() as session:
            component = session.get(ForecastComponentRecord, component_id)
            assert component is not None
            issuance = session.get(ForecastIssuanceRecord, component.forecast_issuance_id)
            assert issuance is not None
            issuance.lifecycle_status = "superseded"

        with pytest.raises(ValueError, match=r"superseded|excluded|active"):
            evaluate_component(
                settings,
                component_id=component_id,
                provider=SymbolProvider({}),
                as_of=date(2026, 4, 13),
                run_id=run_result.run_id,
            )


class TestR4010InvalidCorrectPayload:
    """R4-010: invalid corrected_payload → P09 rejected; original active unchanged."""

    def test_r4_010_invalid_correct_keeps_original(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="r4010")
        p08 = _ingest_p08(
            settings, run_result, source_result, tmp_path, p05, speaker, label="r4010"
        )
        assert p08.status is AiIngestStatus.ACCEPTED
        old_id = p08.forecast_issuance_ids[0]

        bad_payload = {
            "schema_version": "2.1.0",
            "run_id": run_result.run_id,
            "source_id": source_result.source_id,
            "upstream_artifact_id": p05.artifact_ids[0],
            "upstream_prompt_id": "P05",
            "prompt_execution": _prompt("P08"),
            "input_hash": p05.output_hash,
            "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
            "processing_status": "processed_with_forecasts",
            # forecasts missing → invalid
        }
        p09 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-r4010.json",
                _p09(
                    run_id=run_result.run_id,
                    source_id=source_result.source_id,
                    reviewed_artifact_id=p08.artifact_ids[0],
                    input_hash=p08.output_hash,
                    decision="correct",
                    findings=[
                        {
                            "finding_ref": "f1",
                            "severity": "error",
                            "message": "bad",
                            "evidence": "x",
                        }
                    ],
                    corrected_payload=bad_payload,
                ),
            ),
        )
        assert p09.status in {AiIngestStatus.REJECTED, AiIngestStatus.NEEDS_REVIEW}

        sf = create_session_factory(settings.database_file)
        with sf() as session:
            old = session.get(ForecastIssuanceRecord, old_id)
            assert old is not None
            assert old.lifecycle_status == "active"
            assert len(_active_issuances(session)) == 1


# =============================================================================
# Reuse
# =============================================================================


class TestR4011to014ReuseVertical:
    """R4-011/012/013/014: same RAW, same analyst, different URL → reuse + P08."""

    def test_r4_011_014_reuse_applicability_and_p08(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="reuse1")
        assert p05.status is AiIngestStatus.ACCEPTED
        origin_p05_id = p05.artifact_ids[0]

        src2 = _import_second_source(
            settings,
            run_result.run_id,
            tmp_path,
            url="https://example.invalid/video/reuse-2",
            label="reuse2",
        )
        assert src2.source_id != source_result.source_id

        sf = create_session_factory(settings.database_file)
        with sf() as session:
            apps = list(
                session.scalars(
                    select(ArtifactApplicabilityRecord).where(
                        ArtifactApplicabilityRecord.ai_artifact_id == origin_p05_id,
                        ArtifactApplicabilityRecord.target_source_id == src2.source_id,
                    )
                )
            )
            assert len(apps) >= 1
            assert is_artifact_applicable_for_source(
                session, artifact_id=origin_p05_id, source_id=src2.source_id
            )
            link2 = session.get(
                RunSourceRecord,
                {"run_id": run_result.run_id, "source_id": src2.source_id},
            )
            assert link2 is not None
            assert link2.latest_ai_artifact_id == origin_p05_id
            applicability_id = apps[0].applicability_id
            reused_from = apps[0].reused_from_artifact_id

        # R4-012: P08 for source2 with origin P05 upstream (same segment refs)
        p08b = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p08-reuse2.json",
                _p08_payload(
                    run_id=run_result.run_id,
                    source_id=src2.source_id,
                    upstream_artifact_id=origin_p05_id,
                    input_hash=p05.output_hash,
                    speaker=speaker,
                    label="reuse2",
                    segment_label="reuse1",
                ),
            ),
        )
        assert p08b.status is AiIngestStatus.ACCEPTED, [
            f"{i.code}: {i.message}" for i in p08b.issues
        ]
        assert len(p08b.forecast_issuance_ids) == 1

        # R4-013 lineage: applicability points to origin
        assert reused_from == origin_p05_id

        # R4-014: reuse twice → same applicability count
        with sf.begin() as session:
            again = reuse_artifact_for_source(
                session,
                original_artifact_id=origin_p05_id,
                target_run_id=run_result.run_id,
                target_source_id=src2.source_id,
            )
            assert again.applicability_id == applicability_id
            count = session.scalar(
                select(func.count())
                .select_from(ArtifactApplicabilityRecord)
                .where(
                    ArtifactApplicabilityRecord.ai_artifact_id == origin_p05_id,
                    ArtifactApplicabilityRecord.target_source_id == src2.source_id,
                )
            )
            assert count == 1


class TestR4015DifferentRawNoReuse:
    """R4-015: different raw text → no reuse."""

    def test_r4_015_no_reuse_on_different_text(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, _speaker_name = _ingest_p05(
            settings, run_result, source_result, tmp_path, label="diffraw"
        )
        assert p05.status is AiIngestStatus.ACCEPTED

        other = _import_second_source(
            settings,
            run_result.run_id,
            tmp_path,
            url="https://example.invalid/video/different-raw",
            text="全く別の原文です。予想はありません。",
            label="diffraw2",
        )
        sf = create_session_factory(settings.database_file)
        with sf() as session:
            apps = list(
                session.scalars(
                    select(ArtifactApplicabilityRecord).where(
                        ArtifactApplicabilityRecord.target_source_id == other.source_id
                    )
                )
            )
            assert apps == []
            link = session.get(
                RunSourceRecord,
                {"run_id": run_result.run_id, "source_id": other.source_id},
            )
            assert link is not None
            assert link.latest_ai_artifact_id != p05.artifact_ids[0]


class TestR4016PromptModelMismatch:
    """R4-016: prompt_version/model mismatch refuses reuse lookup."""

    def test_r4_016_model_and_prompt_version_mismatch(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        from analyst_forecast.application.raw_sources import (
            can_reuse_processed_artifact as lookup_reuse,
        )
        from analyst_forecast.infrastructure.db.models import PromptExecutionRecord

        p05, _ = _ingest_p05(settings, run_result, source_result, tmp_path, label="r4016")
        assert p05.status is AiIngestStatus.ACCEPTED

        sf = create_session_factory(settings.database_file)
        with sf() as session:
            from analyst_forecast.infrastructure.db.models import AiArtifactRecord

            art = session.get(AiArtifactRecord, p05.artifact_ids[0])
            assert art is not None
            execution = session.get(PromptExecutionRecord, art.prompt_execution_id)
            assert execution is not None
            content_hash = art.input_hash
            analyst_id = session.get(SourceRecord, source_result.source_id).analyst_id  # type: ignore[union-attr]
            assert analyst_id is not None

            wrong_model = lookup_reuse(
                session,
                content_hash=content_hash,
                prompt_id="P05",
                prompt_version=execution.prompt_version,
                model="different-model-v999",
                analyst_id=analyst_id,
            )
            assert wrong_model is None

            wrong_version = lookup_reuse(
                session,
                content_hash=content_hash,
                prompt_id="P05",
                prompt_version="9.9.9",
                model=execution.model,
                analyst_id=analyst_id,
            )
            assert wrong_version is None

            matching = lookup_reuse(
                session,
                content_hash=content_hash,
                prompt_id="P05",
                prompt_version=execution.prompt_version,
                model=execution.model,
                analyst_id=analyst_id,
            )
            assert matching is not None
            assert matching.ai_artifact_id == p05.artifact_ids[0]


class TestR4017CutoffMismatch:
    """R4-017: can_reuse_processed_artifact rejects cutoff mismatch."""

    def test_r4_017_cutoff_exceeds_boundary(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        from analyst_forecast.infrastructure.db.models import AiArtifactRecord

        p05, _ = _ingest_p05(settings, run_result, source_result, tmp_path, label="cutreuse")
        src2 = _import_second_source(
            settings,
            run_result.run_id,
            tmp_path,
            url="https://example.invalid/video/cutoff-reuse",
            label="cutreuse2",
        )
        sf = create_session_factory(settings.database_file)
        with sf.begin() as session:
            art = session.get(AiArtifactRecord, p05.artifact_ids[0])
            assert art is not None
            art.knowledge_cutoff = datetime(2026, 7, 1, tzinfo=UTC)
            target = session.get(SourceRecord, src2.source_id)
            assert target is not None
            # SQLite may strip tzinfo; restore aware datetimes for the check.
            if target.recorded_at is not None and target.recorded_at.tzinfo is None:
                target.recorded_at = target.recorded_at.replace(tzinfo=UTC)
            if target.published_at is not None and target.published_at.tzinfo is None:
                target.published_at = target.published_at.replace(tzinfo=UTC)
            for row in list(
                session.scalars(
                    select(ArtifactApplicabilityRecord).where(
                        ArtifactApplicabilityRecord.target_source_id == src2.source_id
                    )
                )
            ):
                session.delete(row)
            session.flush()
            # Avoid target_has_own_accepted if auto-reuse created accepted link status only
            ok, reason = can_reuse_processed_artifact(
                session,
                original_artifact=art,
                target_source=target,
                target_run_id=run_result.run_id,
            )
            assert ok is False
            assert reason == "cutoff_exceeds_target_boundary"


class TestR4018InvalidUpstreamWithoutApplicability:
    """R4-018: P08 referencing other source artifact without applicability → reject."""

    def test_r4_018_invalid_upstream_reference(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="upx")
        # Different raw so no auto-reuse applicability
        src2 = _import_second_source(
            settings,
            run_result.run_id,
            tmp_path,
            url="https://example.invalid/video/no-applicability",
            text="別原文B。再利用対象外。",
            label="upx2",
        )
        # Still need a local P05 for src2 segments? Upstream check fails before that
        # if we point at source1's P05 without applicability.
        p08 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p08-bad-upstream.json",
                _p08_payload(
                    run_id=run_result.run_id,
                    source_id=src2.source_id,
                    upstream_artifact_id=p05.artifact_ids[0],
                    input_hash=p05.output_hash,
                    speaker=speaker,
                    label="upx-bad",
                ),
            ),
        )
        assert p08.status is AiIngestStatus.REJECTED
        codes = [i.code for i in p08.issues]
        assert "invalid_upstream_reference" in codes, codes


class TestR4019WorkflowSharesApplicability:
    """R4-019: workflow accepted_preprocess uses same applicability as P08 helper."""

    def test_r4_019_pending_p08_after_reuse_matches_applicability(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, _ = _ingest_p05(settings, run_result, source_result, tmp_path, label="r4019")
        assert p05.status is AiIngestStatus.ACCEPTED
        origin_p05_id = p05.artifact_ids[0]

        src2 = _import_second_source(
            settings,
            run_result.run_id,
            tmp_path,
            url="https://example.invalid/video/r4019-reuse",
            label="r4019b",
        )
        sf = create_session_factory(settings.database_file)
        with sf() as session:
            assert is_artifact_applicable_for_source(
                session, artifact_id=origin_p05_id, source_id=src2.source_id
            )

        state = refresh_workflow(settings, run_result.run_id)
        assert state.recommended_action.action_id == "EXTRACT_FORECASTS"
        assert (
            src2.source_id in state.recommended_action.reason
            or "pending_source_ids" in state.recommended_action.reason
        )
        # inputs are local vault paths for pending P08 sources (reuse target included)
        assert state.recommended_action.inputs
        assert any(
            src2.source_id in path or "r4019b" in path or Path(path).name.endswith(".txt")
            for path in state.recommended_action.inputs
        )
        # Shared applicability: reused P05 remains applicable for target source
        with sf() as session:
            assert (
                is_artifact_applicable_for_source(
                    session, artifact_id=origin_p05_id, source_id=src2.source_id
                )
                is True
            )
            link = session.get(
                RunSourceRecord,
                {"run_id": run_result.run_id, "source_id": src2.source_id},
            )
            assert link is not None
            assert link.latest_ai_artifact_id == origin_p05_id
            assert is_artifact_applicable_for_source(
                session,
                artifact_id=link.latest_ai_artifact_id,
                source_id=src2.source_id,
            )


# =============================================================================
# State machine
# =============================================================================


class TestR4020RejectRetryableExtract:
    """R4-020: after reject retryable, refresh → EXTRACT_FORECASTS / P08, not RUN_PREPROCESS."""

    def test_r4_020_refresh_extracts_not_preprocess(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="r4020")
        p08 = _ingest_p08(
            settings, run_result, source_result, tmp_path, p05, speaker, label="r4020"
        )
        _mark_p08_needs_review(settings, p08.artifact_ids[0])
        p09 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-r4020.json",
                _p09(
                    run_id=run_result.run_id,
                    source_id=source_result.source_id,
                    reviewed_artifact_id=p08.artifact_ids[0],
                    input_hash=p08.output_hash,
                    decision="reject",
                    findings=[
                        {
                            "finding_ref": "f1",
                            "severity": "error",
                            "message": "retryable",
                            "evidence": "P08",
                        }
                    ],
                ),
            ),
        )
        assert p09.status is AiIngestStatus.ACCEPTED, [f"{i.code}: {i.message}" for i in p09.issues]

        state = refresh_workflow(settings, run_result.run_id)
        assert state.recommended_action.action_id == "EXTRACT_FORECASTS"
        assert state.recommended_action.action_id != "RUN_PREPROCESS"
        reason = state.recommended_action.reason.lower()
        assert "p08" in reason or "抽出" in state.recommended_action.reason


class TestR4021LatestPointsToP05:
    """R4-021: after reject retryable, latest_ai_artifact_id points to accepted P05."""

    def test_r4_021_latest_restored_to_p05(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="r4021")
        p08 = _ingest_p08(
            settings, run_result, source_result, tmp_path, p05, speaker, label="r4021"
        )
        _mark_p08_needs_review(settings, p08.artifact_ids[0])
        ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-r4021.json",
                _p09(
                    run_id=run_result.run_id,
                    source_id=source_result.source_id,
                    reviewed_artifact_id=p08.artifact_ids[0],
                    input_hash=p08.output_hash,
                    decision="reject",
                    findings=[
                        {
                            "finding_ref": "f1",
                            "severity": "error",
                            "message": "retry",
                            "evidence": "x",
                        }
                    ],
                ),
            ),
        )
        sf = create_session_factory(settings.database_file)
        with sf() as session:
            link = session.get(
                RunSourceRecord,
                {"run_id": run_result.run_id, "source_id": source_result.source_id},
            )
            assert link is not None
            assert link.processing_status == "p08_reextract_required"
            assert link.latest_ai_artifact_id == p05.artifact_ids[0]


class TestR4022RejectTerminal:
    """R4-022: P09 reject_terminal → p08_rejected_terminal; no REVIEW_AI_OUTPUT."""

    def test_r4_022_terminal_reject_no_review_loop(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="r4022")
        p08 = _ingest_p08(
            settings, run_result, source_result, tmp_path, p05, speaker, label="r4022"
        )
        _mark_p08_needs_review(settings, p08.artifact_ids[0])
        p09 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-r4022.json",
                _p09(
                    run_id=run_result.run_id,
                    source_id=source_result.source_id,
                    reviewed_artifact_id=p08.artifact_ids[0],
                    input_hash=p08.output_hash,
                    decision="reject",
                    findings=[
                        {
                            "finding_ref": "f1",
                            "severity": "error",
                            "message": "not speaker",
                            "evidence": "原文",
                        }
                    ],
                    reject_terminal=True,
                    reject_reason="not speaker",
                ),
            ),
        )
        assert p09.status is AiIngestStatus.ACCEPTED, [f"{i.code}: {i.message}" for i in p09.issues]

        sf = create_session_factory(settings.database_file)
        with sf() as session:
            link = session.get(
                RunSourceRecord,
                {"run_id": run_result.run_id, "source_id": source_result.source_id},
            )
            assert link is not None
            assert link.processing_status == "p08_rejected_terminal"

        state = refresh_workflow(settings, run_result.run_id)
        assert state.recommended_action.action_id != "REVIEW_AI_OUTPUT"
        assert (
            state.recommended_action.action_id
            in {
                "COMPLETE_NO_ACTIVE_FORECAST",
                "REVIEW_NO_FORECAST",
                "ADD_ANOTHER_SOURCE",
            }
            or state.stage == "complete_no_active_forecast"
        )


class TestR4023UnresolvedTerminal:
    """R4-023: P09 unresolved → terminal; refresh → no infinite REVIEW."""

    def test_r4_023_unresolved_no_review_loop(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="r4023")
        p08 = _ingest_p08(
            settings, run_result, source_result, tmp_path, p05, speaker, label="r4023"
        )
        _mark_p08_needs_review(settings, p08.artifact_ids[0])
        ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-r4023.json",
                _p09(
                    run_id=run_result.run_id,
                    source_id=source_result.source_id,
                    reviewed_artifact_id=p08.artifact_ids[0],
                    input_hash=p08.output_hash,
                    decision="unresolved",
                    findings=[
                        {
                            "finding_ref": "f1",
                            "severity": "warning",
                            "message": "cannot decide",
                            "evidence": "x",
                        }
                    ],
                ),
            ),
        )
        state = refresh_workflow(settings, run_result.run_id)
        assert state.recommended_action.action_id != "REVIEW_AI_OUTPUT"


class TestR4024UnresolvedExcluded:
    """R4-024: after unresolved on accepted P08, lifecycle review_unresolved_excluded."""

    def test_r4_024_unresolved_excluded_from_results(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="r4024")
        p08 = _ingest_p08(
            settings, run_result, source_result, tmp_path, p05, speaker, label="r4024"
        )
        assert p08.status is AiIngestStatus.ACCEPTED
        issuance_id = p08.forecast_issuance_ids[0]
        # accepted P08 is reviewable for unresolved
        p09 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-r4024.json",
                _p09(
                    run_id=run_result.run_id,
                    source_id=source_result.source_id,
                    reviewed_artifact_id=p08.artifact_ids[0],
                    input_hash=p08.output_hash,
                    decision="unresolved",
                    findings=[
                        {
                            "finding_ref": "f1",
                            "severity": "warning",
                            "message": "unresolved",
                            "evidence": "x",
                        }
                    ],
                ),
            ),
        )
        assert p09.status is AiIngestStatus.ACCEPTED, [f"{i.code}: {i.message}" for i in p09.issues]

        sf = create_session_factory(settings.database_file)
        with sf() as session:
            iss = session.get(ForecastIssuanceRecord, issuance_id)
            assert iss is not None
            assert iss.lifecycle_status == "review_unresolved_excluded"
            assert _active_issuances(session) == []

        paths = generate_run_results(settings, run_result.run_id)
        assert issuance_id not in paths["forecasts_csv"].read_text(encoding="utf-8")


class TestR4025TwoSourceWorkflow:
    """R4-025: source A unresolved terminal, B only P05 → EXTRACT for B."""

    def test_r4_025_other_source_not_blocked(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05a, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="r4025a")
        p08a = _ingest_p08(
            settings, run_result, source_result, tmp_path, p05a, speaker, label="r4025a"
        )
        _mark_p08_needs_review(settings, p08a.artifact_ids[0])
        ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-r4025a.json",
                _p09(
                    run_id=run_result.run_id,
                    source_id=source_result.source_id,
                    reviewed_artifact_id=p08a.artifact_ids[0],
                    input_hash=p08a.output_hash,
                    decision="unresolved",
                    findings=[
                        {
                            "finding_ref": "f1",
                            "severity": "warning",
                            "message": "A unresolved",
                            "evidence": "x",
                        }
                    ],
                ),
            ),
        )

        src_b = _import_second_source(
            settings,
            run_result.run_id,
            tmp_path,
            url="https://example.invalid/video/r4025-b",
            text="ソースB用の別原文。日経平均は今後上昇する。これは現状分析ではなく予想です。",
            label="r4025b",
        )
        # Own P05 for B (different raw → no auto reuse)
        p05b = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p05-r4025b.json",
                {
                    "schema_version": "2.0.0",
                    "run_id": run_result.run_id,
                    "source_id": src_b.source_id,
                    "prompt_execution": _prompt("P05"),
                    "input_hash": src_b.raw_hash,
                    "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                    "segments": [
                        {
                            "segment_ref": "segment-r4025b",
                            "sequence_number": 1,
                            "raw_start_offset": 0,
                            "raw_end_offset": len(
                                "ソースB用の別原文。日経平均は今後上昇する。これは現状分析ではなく予想です。"
                            ),
                            "raw_text": (
                                "ソースB用の別原文。日経平均は今後上昇する。"
                                "これは現状分析ではなく予想です。"
                            ),
                            "normalized_text": (
                                "ソースB用の別原文。日経平均は今後上昇する。"
                                "これは現状分析ではなく予想です。"
                            ),
                            "speaker_status": "identified",
                            "speaker_candidate": speaker,
                            "speaker_confidence": 0.95,
                            "attribution_basis": "fixture",
                            "review_status": "accepted",
                        }
                    ],
                },
            ),
        )
        assert p05b.status is AiIngestStatus.ACCEPTED, [
            f"{i.code}: {i.message}" for i in p05b.issues
        ]

        state = refresh_workflow(settings, run_result.run_id)
        assert state.recommended_action.action_id == "EXTRACT_FORECASTS"
        assert state.recommended_action.action_id != "REVIEW_AI_OUTPUT"
        assert src_b.source_id in state.recommended_action.reason or "pending" in (
            state.recommended_action.reason.lower()
        )


class TestR4026CompleteNoActive:
    """R4-026: all sources terminal, 0 active → COMPLETE_NO_ACTIVE_FORECAST."""

    def test_r4_026_complete_no_active_forecast(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="r4026")
        p08 = _ingest_p08(
            settings, run_result, source_result, tmp_path, p05, speaker, label="r4026"
        )
        _mark_p08_needs_review(settings, p08.artifact_ids[0])
        ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-r4026.json",
                _p09(
                    run_id=run_result.run_id,
                    source_id=source_result.source_id,
                    reviewed_artifact_id=p08.artifact_ids[0],
                    input_hash=p08.output_hash,
                    decision="reject",
                    findings=[
                        {
                            "finding_ref": "f1",
                            "severity": "error",
                            "message": "terminal",
                            "evidence": "x",
                        }
                    ],
                    reject_terminal=True,
                    reject_reason="not speaker",
                ),
            ),
        )
        state = refresh_workflow(settings, run_result.run_id)
        assert state.recommended_action.action_id == "COMPLETE_NO_ACTIVE_FORECAST"
        assert state.stage == "complete_no_active_forecast"


class TestR4027NextActionsContent:
    """R4-027: NEXT_ACTIONS.md contains source_id or reason after reject."""

    def test_r4_027_next_actions_mentions_source_or_reason(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="r4027")
        p08 = _ingest_p08(
            settings, run_result, source_result, tmp_path, p05, speaker, label="r4027"
        )
        _mark_p08_needs_review(settings, p08.artifact_ids[0])
        ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-r4027.json",
                _p09(
                    run_id=run_result.run_id,
                    source_id=source_result.source_id,
                    reviewed_artifact_id=p08.artifact_ids[0],
                    input_hash=p08.output_hash,
                    decision="reject",
                    findings=[
                        {
                            "finding_ref": "f1",
                            "severity": "error",
                            "message": "retry",
                            "evidence": "x",
                        }
                    ],
                ),
            ),
        )
        refresh_workflow(settings, run_result.run_id)
        from analyst_forecast.infrastructure.db.models import RunRecord

        sf = create_session_factory(settings.database_file)
        with sf() as session:
            run = session.get(RunRecord, run_result.run_id)
            assert run is not None
            next_path = settings.vault_root / Path(run.run_path) / "NEXT_ACTIONS.md"
        text = next_path.read_text(encoding="utf-8")
        assert "EXTRACT_FORECASTS" in text or "P08" in text
        assert source_result.source_id in text or "理由" in text


# =============================================================================
# Time
# =============================================================================


class TestR4030UnknownNotFormal:
    """R4-030: made_at_source=unknown, made_at=null → no formal issuance."""

    def test_r4_030_unknown_not_formalized(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="r4030")
        p08 = _ingest_p08(
            settings,
            run_result,
            source_result,
            tmp_path,
            p05,
            speaker,
            label="r4030",
            made_at=None,
            made_at_source="unknown",
            publicly_available_at=None,
        )
        assert p08.status in {AiIngestStatus.ACCEPTED, AiIngestStatus.NEEDS_REVIEW}
        assert list(p08.forecast_issuance_ids) == []


class TestR4031CutoffRequired:
    """R4-031: P08 2.1.0 without knowledge_cutoff → ValidationError."""

    def test_r4_031_schema_requires_cutoff(self):
        with pytest.raises(ValidationError, match="knowledge_cutoff"):
            P08Output.model_validate(
                {
                    "schema_version": "2.1.0",
                    "run_id": "RUN-20260720-001",
                    "source_id": "SRC-000001",
                    "upstream_artifact_id": "AIF-000001",
                    "upstream_prompt_id": "P05",
                    "prompt_execution": _prompt("P08"),
                    "input_hash": "a" * 64,
                    "processing_status": "processed_no_forecast",
                    "forecasts": [],
                }
            )


class TestR4033P09CutoffExceedsMadeAt:
    """R4-033: P09 knowledge_cutoff after reviewed made_at → rejected."""

    def test_r4_033_p09_cutoff_exceeds(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="r4033")
        p08 = _ingest_p08(
            settings,
            run_result,
            source_result,
            tmp_path,
            p05,
            speaker,
            label="r4033",
            confidence=0.3,
        )
        assert p08.status is AiIngestStatus.NEEDS_REVIEW
        p09 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-r4033.json",
                _p09(
                    run_id=run_result.run_id,
                    source_id=source_result.source_id,
                    reviewed_artifact_id=p08.artifact_ids[0],
                    input_hash=p08.output_hash,
                    decision="accept",
                    knowledge_cutoff="2026-01-11T00:00:00+00:00",
                ),
            ),
        )
        assert p09.status is AiIngestStatus.REJECTED
        assert any(i.code == "p09_cutoff_exceeds_reviewed_made_at" for i in p09.issues)


class TestR4034CorrectedCutoffExceeds:
    """R4-034: P09 correct with corrected_payload cutoff > made_at → rejected."""

    def test_r4_034_corrected_cutoff_rejected(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="r4034")
        p08 = _ingest_p08(
            settings, run_result, source_result, tmp_path, p05, speaker, label="r4034"
        )
        corrected = _corrected_p08(
            run_id=run_result.run_id,
            source_id=source_result.source_id,
            upstream_artifact_id=p05.artifact_ids[0],
            input_hash=p05.output_hash,
            speaker=speaker,
            label="r4034c",
            segment_label="r4034",
            knowledge_cutoff="2026-06-01T00:00:00+00:00",
            made_at="2026-01-10T09:00:00+00:00",
        )
        p09 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-r4034.json",
                _p09(
                    run_id=run_result.run_id,
                    source_id=source_result.source_id,
                    reviewed_artifact_id=p08.artifact_ids[0],
                    input_hash=p08.output_hash,
                    decision="correct",
                    findings=[
                        {
                            "finding_ref": "f1",
                            "severity": "error",
                            "message": "cut",
                            "evidence": "x",
                        }
                    ],
                    corrected_payload=corrected,
                ),
            ),
        )
        assert p09.status is AiIngestStatus.REJECTED
        assert any("cutoff" in i.code for i in p09.issues)


class TestR4035UnknownFormalizedByCorrect:
    """R4-035: unknown P08 + P09 correct with made_at → formalizes once."""

    def test_r4_035_correct_formalizes_unknown(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="r4035")
        p08 = _ingest_p08(
            settings,
            run_result,
            source_result,
            tmp_path,
            p05,
            speaker,
            label="r4035",
            made_at=None,
            made_at_source="unknown",
            publicly_available_at=None,
            confidence=0.3,
        )
        assert p08.status is AiIngestStatus.NEEDS_REVIEW
        assert not p08.forecast_issuance_ids

        corrected = _corrected_p08(
            run_id=run_result.run_id,
            source_id=source_result.source_id,
            upstream_artifact_id=p05.artifact_ids[0],
            input_hash=p05.output_hash,
            speaker=speaker,
            label="r4035c",
            segment_label="r4035",
            made_at="2026-01-10T09:00:00+00:00",
            made_at_source="explicit",
        )
        p09 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-r4035.json",
                _p09(
                    run_id=run_result.run_id,
                    source_id=source_result.source_id,
                    reviewed_artifact_id=p08.artifact_ids[0],
                    input_hash=p08.output_hash,
                    decision="correct",
                    findings=[
                        {
                            "finding_ref": "f1",
                            "severity": "error",
                            "message": "時刻確定",
                            "evidence": "発言日時明示",
                        }
                    ],
                    corrected_payload=corrected,
                ),
            ),
        )
        assert p09.status is AiIngestStatus.ACCEPTED, [f"{i.code}: {i.message}" for i in p09.issues]
        sf = create_session_factory(settings.database_file)
        with sf() as session:
            assert len(_active_issuances(session)) == 1


class TestR4036LegacyNullMadeAtExcluded:
    """R4-036: lifecycle=active but made_at NULL is excluded from active query/results."""

    def test_r4_036_null_made_at_not_in_active_aggregation(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        from analyst_forecast.application.active_forecast_query import (
            active_issuances_query,
            is_active_issuance,
        )

        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="r4036")
        p08 = _ingest_p08(
            settings, run_result, source_result, tmp_path, p05, speaker, label="r4036"
        )
        assert p08.status is AiIngestStatus.ACCEPTED
        issuance_id = p08.forecast_issuance_ids[0]

        sf = create_session_factory(settings.database_file)
        with sf.begin() as session:
            iss = session.get(ForecastIssuanceRecord, issuance_id)
            assert iss is not None
            # Simulate legacy/unknown time row that 0008 may have marked active
            iss.made_at = None
            iss.made_at_source = "unknown"
            iss.lifecycle_status = "active"

        with sf() as session:
            assert is_active_issuance(session, issuance_id) is False
            active_rows = list(session.scalars(active_issuances_query()))
            assert all(row.forecast_issuance_id != issuance_id for row in active_rows)
            assert issuance_id not in [r.forecast_issuance_id for r in active_rows]

        paths = generate_run_results(settings, run_result.run_id)
        summary = paths["summary_md"].read_text(encoding="utf-8")
        assert "- 予想構成数: 0\n" in summary
        assert issuance_id not in paths["forecasts_csv"].read_text(encoding="utf-8")


# =============================================================================
# Basket
# =============================================================================


class TestR4038InsufficientCommonDates:
    """R4-038: 1 common date → insufficient_common_dates; no hit/miss."""

    def test_r4_038_insufficient_common_dates(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        component_id = import_locked_component(
            settings, run_result, source_result, tmp_path, label="r4038"
        )
        _set_basket_instruments(settings, component_id)
        provider = SymbolProvider(
            {
                "AAA": _series("AAA", (_bar(date(2026, 1, 13), "100", "102"),)),
                "BBB": _series("BBB", (_bar(date(2026, 1, 13), "200", "205"),)),
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
        assert result.unevaluable_reason is not None
        assert "insufficient_common_dates" in result.unevaluable_reason
        assert result.direction_result is None
        assert result.actual_return is None
        assert result.max_favorable_excursion is None
        assert result.max_adverse_excursion is None


class TestR4039BasketReturn:
    """R4-039: 2 common dates AAA +20% BBB -20% 50/50 → ~0."""

    def test_r4_039_weighted_return_near_zero(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        component_id = import_locked_component(
            settings, run_result, source_result, tmp_path, label="r4039"
        )
        _set_basket_instruments(settings, component_id)
        provider = SymbolProvider(
            {
                "AAA": _series(
                    "AAA",
                    (
                        _bar(date(2026, 1, 13), "100", "100"),
                        _bar(date(2026, 4, 13), "120", "120"),
                    ),
                ),
                "BBB": _series(
                    "BBB",
                    (
                        _bar(date(2026, 1, 13), "100", "100"),
                        _bar(date(2026, 4, 13), "80", "80"),
                    ),
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
        assert result.actual_return is not None, result.unevaluable_reason
        assert abs(result.actual_return - Decimal("0")) < Decimal("0.0001")

        sf = create_session_factory(settings.database_file)
        with sf() as session:
            ev = session.scalar(
                select(EvaluationRecord).where(
                    EvaluationRecord.forecast_component_id == component_id
                )
            )
            assert ev is not None
            assert ev.coverage_audit is not None


class TestR4040MissingSymbol:
    """R4-040: missing one symbol → unevaluable, not partial basket."""

    def test_r4_040_missing_symbol_unevaluable(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        component_id = import_locked_component(
            settings, run_result, source_result, tmp_path, label="r4040"
        )
        _set_basket_instruments(settings, component_id)
        provider = SymbolProvider(
            {
                "AAA": _series(
                    "AAA",
                    (
                        _bar(date(2026, 1, 13), "100", "100"),
                        _bar(date(2026, 4, 13), "120", "120"),
                    ),
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
        assert result.direction_result is None
        assert result.actual_return is None


class TestR4041CoverageAuditKeys:
    """R4-041: coverage_audit keys on successful basket eval."""

    def test_r4_041_coverage_audit_keys(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        component_id = import_locked_component(
            settings, run_result, source_result, tmp_path, label="r4041"
        )
        _set_basket_instruments(settings, component_id)
        provider = SymbolProvider(
            {
                "AAA": _series(
                    "AAA",
                    (
                        _bar(date(2026, 1, 13), "100", "100"),
                        _bar(date(2026, 4, 13), "110", "110"),
                    ),
                ),
                "BBB": _series(
                    "BBB",
                    (
                        _bar(date(2026, 1, 13), "100", "100"),
                        _bar(date(2026, 4, 13), "105", "105"),
                    ),
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
        assert result.actual_return is not None, result.unevaluable_reason

        sf = create_session_factory(settings.database_file)
        with sf() as session:
            ev = session.scalar(
                select(EvaluationRecord).where(
                    EvaluationRecord.forecast_component_id == component_id
                )
            )
            assert ev is not None
            audit = ev.coverage_audit
            assert isinstance(audit, dict)
            for key in (
                "common_date_count",
                "selected_start_date",
                "selected_end_date",
                "basket_weights",
                "mapping_hash",
                "input_series_hashes",
                "common_date_rule",
                "evaluation_method_version",
            ):
                assert key in audit
