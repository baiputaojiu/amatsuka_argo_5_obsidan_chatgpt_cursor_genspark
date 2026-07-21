"""Round5 functional tests: lineage, active gate, time boundary, coverage, reject."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select

from analyst_forecast.application.active_forecast_query import (
    InactiveComponentError,
    require_active_component_context,
)
from analyst_forecast.application.ai_ingestion import AiIngestStatus, ingest_ai_output
from analyst_forecast.application.evaluation import evaluate_component
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.domain.market import MarketBar, MarketSeries
from analyst_forecast.infrastructure.db.models import (
    EvaluationRecord,
    ForecastComponentRecord,
    ForecastCorrectionOperationRecord,
    ForecastIssuanceRecord,
    TargetResolutionCandidateRecord,
)
from analyst_forecast.infrastructure.db.session import create_session_factory
from analyst_forecast.schemas.pipeline import P09Output
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


def _forecast_block(
    *,
    source_id: str,
    speaker: str,
    label: str,
    made_at: str | None = "2026-01-10T09:00:00+00:00",
    made_at_source: str = "explicit",
    publicly_available_at: str | None = "2026-01-10T10:00:00+00:00",
    start: str = "2026-01-13",
    end: str = "2026-04-13",
    segment_label: str = "main",
) -> dict[str, Any]:
    quote = "日経平均は今後上昇する"
    return {
        "forecast_ref": f"forecast-{label}",
        "forecast_group_ref": f"group-{label}",
        "made_at": made_at,
        "publicly_available_at": publicly_available_at,
        "made_at_source": made_at_source,
        "forecast_type": "directional",
        "commitment_strength": "explicit",
        "evidence_level": "A",
        "extraction_confidence": 0.95,
        "human_readable_summary": quote,
        "relation_to_previous": "initial",
        "upstream_segment_refs": [f"segment-{segment_label}"],
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
                "direction": "up",
                "time_source": "explicit",
                "time_expression_raw": "今後3か月",
                "normalized_start": start,
                "normalized_end": end,
                "raw_target_label": "日経平均",
                "target_resolution_status": "pending",
            }
        ],
    }


def _ingest_multi_p08(
    settings,
    run_result,
    source_result,
    tmp_path,
    p05_result,
    speaker,
    *,
    labels: tuple[str, ...] = ("a", "b"),
    made_at: str | None = "2026-01-10T09:00:00+00:00",
    made_at_source: str = "explicit",
    segment_label: str = "main",
):
    payload = {
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
            _forecast_block(
                source_id=source_result.source_id,
                speaker=speaker,
                label=lab,
                made_at=made_at,
                made_at_source=made_at_source,
                segment_label=segment_label,
            )
            for lab in labels
        ],
    }
    return ingest_ai_output(
        settings, _write(tmp_path, f"p08-multi-{'-'.join(labels)}.json", payload)
    )


def _mark_needs_review(settings: AppSettings, artifact_id: str) -> None:
    from analyst_forecast.infrastructure.db.models import AiArtifactRecord

    sf = create_session_factory(settings.database_file)
    with sf.begin() as session:
        art = session.get(AiArtifactRecord, artifact_id)
        assert art is not None
        art.classification = "needs_review"
        art.resolution_status = "needs_review"


class SymbolProvider:
    name = "fixture"

    def __init__(self, series_by_symbol: dict[str, MarketSeries]) -> None:
        self._series = series_by_symbol

    def fetch(self, request):  # type: ignore[no-untyped-def]
        series = self._series.get(request.symbol)
        if series is None:
            from analyst_forecast.domain.market import MarketDataUnavailable

            raise MarketDataUnavailable(f"missing {request.symbol}")
        return series


def _bar(day: date, open_: str, close: str) -> MarketBar:
    price_o = Decimal(open_)
    price_c = Decimal(close)
    return MarketBar(
        date=day,
        open=price_o,
        high=max(price_o, price_c),
        low=min(price_o, price_c),
        close=price_c,
        adjusted_open=price_o,
        adjusted_close=price_c,
    )


def _series(symbol: str, bars: tuple[MarketBar, ...]) -> MarketSeries:
    return MarketSeries(
        provider="fixture",
        symbol=symbol,
        currency="JPY",
        adjustment_type="split_dividend",
        frequency="daily",
        retrieved_at=datetime(2026, 7, 20, tzinfo=UTC),
        bars=bars,
    )


# --- Fix 02 lineage ---


class TestR5MultiForecastLineage:
    def test_r5_009_pairwise_lineage(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="main")
        p08 = _ingest_multi_p08(
            settings, run_result, source_result, tmp_path, p05, speaker, labels=("a", "b")
        )
        assert p08.status is AiIngestStatus.ACCEPTED, [i.message for i in p08.issues]
        _mark_needs_review(settings, p08.artifact_ids[0])

        corrected = {
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
                _forecast_block(source_id=source_result.source_id, speaker=speaker, label="a-new"),
                _forecast_block(source_id=source_result.source_id, speaker=speaker, label="b-new"),
            ],
        }
        # Reorder corrected payload B then A — operations still map correctly.
        corrected["forecasts"] = list(reversed(corrected["forecasts"]))
        p09 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-correct.json",
                {
                    "schema_version": "2.1.0",
                    "run_id": run_result.run_id,
                    "source_id": source_result.source_id,
                    "reviewed_artifact_id": p08.artifact_ids[0],
                    "prompt_execution": _prompt("P09"),
                    "input_hash": p08.output_hash,
                    "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                    "decision": "correct",
                    "findings": [],
                    "corrected_payload": corrected,
                    "forecast_operations": [
                        {
                            "action": "update",
                            "reviewed_forecast_ref": "forecast-a",
                            "corrected_forecast_ref": "forecast-a-new",
                            "reason": "fix a",
                        },
                        {
                            "action": "update",
                            "reviewed_forecast_ref": "forecast-b",
                            "corrected_forecast_ref": "forecast-b-new",
                            "reason": "fix b",
                        },
                    ],
                },
            ),
        )
        assert p09.status is AiIngestStatus.ACCEPTED, [f"{i.code}: {i.message}" for i in p09.issues]

        sf = create_session_factory(settings.database_file)
        with sf() as session:
            actives = list(
                session.scalars(
                    select(ForecastIssuanceRecord).where(
                        ForecastIssuanceRecord.lifecycle_status == "active"
                    )
                )
            )
            assert len(actives) == 2
            roots = {iss.lineage_root_id for iss in actives}
            assert len(roots) == 2
            for iss in actives:
                assert iss.supersedes_forecast_issuance_id is not None
                old = session.get(ForecastIssuanceRecord, iss.supersedes_forecast_issuance_id)
                assert old is not None
                assert old.superseded_by_issuance_id == iss.forecast_issuance_id
                assert old.lifecycle_status == "superseded"
            ops = list(session.scalars(select(ForecastCorrectionOperationRecord)))
            assert len(ops) == 2

    def test_r5_016_ambiguous_multi_without_ops_rejected(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="main")
        p08 = _ingest_multi_p08(
            settings, run_result, source_result, tmp_path, p05, speaker, labels=("a", "b")
        )
        _mark_needs_review(settings, p08.artifact_ids[0])
        corrected = {
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
                _forecast_block(source_id=source_result.source_id, speaker=speaker, label="a-new"),
                _forecast_block(source_id=source_result.source_id, speaker=speaker, label="b-new"),
            ],
        }
        result = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-ambiguous.json",
                {
                    "schema_version": "2.1.0",
                    "run_id": run_result.run_id,
                    "source_id": source_result.source_id,
                    "reviewed_artifact_id": p08.artifact_ids[0],
                    "prompt_execution": _prompt("P09"),
                    "input_hash": p08.output_hash,
                    "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                    "decision": "correct",
                    "findings": [],
                    "corrected_payload": corrected,
                },
            ),
        )
        assert result.status is not AiIngestStatus.ACCEPTED
        assert any(i.code == "ambiguous_forecast_correction" for i in result.issues)
        sf = create_session_factory(settings.database_file)
        with sf() as session:
            actives = list(
                session.scalars(
                    select(ForecastIssuanceRecord).where(
                        ForecastIssuanceRecord.lifecycle_status == "active"
                    )
                )
            )
            assert len(actives) == 2
            roots = {a.lineage_root_id for a in actives}
            assert len(roots) == 2


# --- Fix 03 active gate ---


class TestR5ActiveComponentGate:
    def test_r5_019_superseded_p11_rejected(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        from tests.unit.test_round4_acceptance import (
            _ingest_p05 as r4_p05,
        )
        from tests.unit.test_round4_acceptance import (
            _ingest_p08 as r4_p08,
        )
        from tests.unit.test_round4_acceptance import (
            _mark_p08_needs_review,
            _p09,
        )

        p05, speaker = r4_p05(settings, run_result, source_result, tmp_path, label="gate")
        p08 = r4_p08(settings, run_result, source_result, tmp_path, p05, speaker, label="gate")
        sf = create_session_factory(settings.database_file)
        with sf() as session:
            old_comp = session.scalar(select(ForecastComponentRecord))
            assert old_comp is not None
            old_component_id = old_comp.forecast_component_id
            old_iss = session.get(ForecastIssuanceRecord, old_comp.forecast_issuance_id)
            assert old_iss is not None
            old_ref = old_iss.local_ref

        _mark_p08_needs_review(settings, p08.artifact_ids[0])
        from tests.unit.test_round4_acceptance import _p08_payload

        corrected = _p08_payload(
            run_id=run_result.run_id,
            source_id=source_result.source_id,
            upstream_artifact_id=p05.artifact_ids[0],
            input_hash=p05.output_hash,
            speaker=speaker,
            label="gate-new",
            segment_label="gate",
        )
        p09 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-gate.json",
                {
                    **_p09(
                        run_id=run_result.run_id,
                        source_id=source_result.source_id,
                        reviewed_artifact_id=p08.artifact_ids[0],
                        input_hash=p08.output_hash,
                        decision="correct",
                        corrected_payload=corrected,
                        knowledge_cutoff="2026-01-10T09:00:00+00:00",
                    ),
                    "forecast_operations": [
                        {
                            "action": "update",
                            "reviewed_forecast_ref": old_ref,
                            "corrected_forecast_ref": "forecast-gate-new",
                            "reason": "supersede for gate test",
                        }
                    ],
                },
            ),
        )
        assert p09.status is AiIngestStatus.ACCEPTED, [f"{i.code}: {i.message}" for i in p09.issues]

        before = 0
        with sf() as session:
            before = (
                session.scalar(select(func.count()).select_from(TargetResolutionCandidateRecord))
                or 0
            )
            old_iss_after = session.get(
                ForecastIssuanceRecord,
                session.get(ForecastComponentRecord, old_component_id).forecast_issuance_id,
            )
            assert old_iss_after is not None
            assert old_iss_after.lifecycle_status == "superseded", old_iss_after.lifecycle_status
            gate = require_active_component_context(session, old_component_id)
            assert isinstance(gate, InactiveComponentError)
            assert gate.code == "inactive_forecast_component"

        # Use real P08 output hash from original artifact for context match paths;
        # gate must still fire before hash checks once lifecycle is superseded.
        from analyst_forecast.infrastructure.db.models import AiArtifactRecord

        with sf() as session:
            art = session.get(AiArtifactRecord, p08.artifact_ids[0])
            assert art is not None
            p08_hash = art.output_hash

        p11 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p11-old.json",
                {
                    "schema_version": "2.0.0",
                    "run_id": run_result.run_id,
                    "source_id": source_result.source_id,
                    "forecast_component_id": old_component_id,
                    "prompt_execution": _prompt("P11"),
                    "input_hash": p08_hash,
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
        assert p11.status is not AiIngestStatus.ACCEPTED, p11.status
        codes = [i.code for i in p11.issues]
        assert "inactive_forecast_component" in codes, codes
        with sf() as session:
            after = (
                session.scalar(select(func.count()).select_from(TargetResolutionCandidateRecord))
                or 0
            )
            assert after == before


# --- Fix 04 time boundary ---


class TestR5CorrectedTimeBoundary:
    def test_r5_027_p09_cutoff_after_corrected_made_at_rejected(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        from tests.unit.test_round4_acceptance import _ingest_p05 as r4_p05
        from tests.unit.test_round4_acceptance import _ingest_p08 as r4_p08
        from tests.unit.test_round4_acceptance import _mark_p08_needs_review, _p08_payload

        p05, speaker = r4_p05(settings, run_result, source_result, tmp_path, label="time")
        p08 = r4_p08(
            settings,
            run_result,
            source_result,
            tmp_path,
            p05,
            speaker,
            label="time",
            made_at=None,
            made_at_source="unknown",
            publicly_available_at=None,
            knowledge_cutoff="2026-01-10T09:00:00+00:00",
            confidence=0.4,
        )
        # Unknown-time P08 may be needs_review / non-formal; force review path.
        _mark_p08_needs_review(settings, p08.artifact_ids[0])
        corrected = _p08_payload(
            run_id=run_result.run_id,
            source_id=source_result.source_id,
            upstream_artifact_id=p05.artifact_ids[0],
            input_hash=p05.output_hash,
            speaker=speaker,
            label="time-known",
            made_at="2026-01-10T08:00:00+00:00",
            knowledge_cutoff="2026-01-10T08:00:00+00:00",
        )
        # Add time evidence if required by P08 validation — keep minimal.
        bad = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-cutoff-bad.json",
                {
                    "schema_version": "2.1.0",
                    "run_id": run_result.run_id,
                    "source_id": source_result.source_id,
                    "reviewed_artifact_id": p08.artifact_ids[0],
                    "prompt_execution": _prompt("P09"),
                    "input_hash": p08.output_hash,
                    "knowledge_cutoff": "2026-01-10T08:30:00+00:00",
                    "decision": "correct",
                    "findings": [],
                    "corrected_payload": corrected,
                },
            ),
        )
        assert bad.status is not AiIngestStatus.ACCEPTED
        assert any(i.code == "p09_cutoff_exceeds_corrected_made_at" for i in bad.issues), [
            f"{i.code}:{i.message}" for i in bad.issues
        ]

        good = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-cutoff-good.json",
                {
                    "schema_version": "2.1.0",
                    "run_id": run_result.run_id,
                    "source_id": source_result.source_id,
                    "reviewed_artifact_id": p08.artifact_ids[0],
                    "prompt_execution": _prompt("P09"),
                    "input_hash": p08.output_hash,
                    "knowledge_cutoff": "2026-01-10T08:00:00+00:00",
                    "decision": "correct",
                    "findings": [],
                    "corrected_payload": corrected,
                },
            ),
        )
        # May still fail for other unknown→known evidence rules; cutoff itself must pass.
        assert not any(i.code == "p09_cutoff_exceeds_corrected_made_at" for i in good.issues)


# --- Fix 05 coverage ---


class TestR5Coverage:
    def test_r5_034_single_symbol_one_day_unevaluable(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        from tests.unit.test_round4_acceptance import import_locked_component

        component_id = import_locked_component(
            settings, run_result, source_result, tmp_path, label="cov1"
        )
        provider = SymbolProvider(
            {"N225": _series("N225", (_bar(date(2026, 1, 13), "100", "102"),))}
        )
        # Locked component may use ticker from mapping — set instruments to N225 if needed.
        sf = create_session_factory(settings.database_file)
        with sf.begin() as session:
            from analyst_forecast.infrastructure.db.models import TargetMappingRecord, TargetRecord

            comp = session.get(ForecastComponentRecord, component_id)
            assert comp is not None and comp.target_mapping_id
            mapping = session.get(TargetMappingRecord, comp.target_mapping_id)
            target = session.get(TargetRecord, comp.target_id)
            assert mapping is not None and target is not None
            symbol = target.ticker or "N225"
            mapping.evaluation_instruments = [
                {"symbol": symbol, "currency": target.currency or "JPY", "weight": 1.0}
            ]
            mapping.weights = [1.0]

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
        assert result.unevaluable_reason is not None
        assert "insufficient_trading_dates" in result.unevaluable_reason
        with sf() as session:
            ev = session.scalar(
                select(EvaluationRecord).where(
                    EvaluationRecord.forecast_component_id == component_id
                )
            )
            assert ev is not None
            assert ev.coverage_audit is not None
            assert ev.coverage_audit.get("reason_code") == "insufficient_trading_dates"


# --- Fix 06 reject disposition ---


class TestR5RejectDisposition:
    def test_r5_043_missing_disposition_rejected(self) -> None:
        with pytest.raises(ValidationError):
            P09Output.model_validate(
                {
                    "schema_version": "2.1.0",
                    "run_id": "RUN-20260101-001",
                    "source_id": "SRC-000001",
                    "reviewed_artifact_id": "AIF-000001",
                    "prompt_execution": _prompt("P09"),
                    "input_hash": "a" * 64,
                    "decision": "reject",
                    "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                    "reject_reason": "missing disposition",
                }
            )

    def test_r5_044_blank_reason_rejected(self) -> None:
        with pytest.raises(ValidationError):
            P09Output.model_validate(
                {
                    "schema_version": "2.1.0",
                    "run_id": "RUN-20260101-001",
                    "source_id": "SRC-000001",
                    "reviewed_artifact_id": "AIF-000001",
                    "prompt_execution": _prompt("P09"),
                    "input_hash": "a" * 64,
                    "decision": "reject",
                    "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                    "reject_disposition": "retryable",
                    "reject_reason": "   ",
                }
            )

    def test_r5_045_reject_terminal_forbidden_on_2_1(self) -> None:
        with pytest.raises(ValidationError):
            P09Output.model_validate(
                {
                    "schema_version": "2.1.0",
                    "run_id": "RUN-20260101-001",
                    "source_id": "SRC-000001",
                    "reviewed_artifact_id": "AIF-000001",
                    "prompt_execution": _prompt("P09"),
                    "input_hash": "a" * 64,
                    "decision": "reject",
                    "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                    "reject_disposition": "terminal",
                    "reject_reason": "ok",
                    "reject_terminal": True,
                }
            )

    def test_r5_046_schema_matches_pydantic(self) -> None:
        from analyst_forecast.schemas.pipeline import pipeline_schema_path

        fixed = json.loads(pipeline_schema_path("P09").read_text(encoding="utf-8"))
        assert fixed == P09Output.model_json_schema()

    def test_prompt_mentions_disposition(self) -> None:
        prompt = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "analyst_forecast"
            / "resources"
            / "prompts"
            / "P09.md.j2"
        ).read_text(encoding="utf-8")
        assert "reject_disposition" in prompt
        assert "retryable" in prompt
        assert "terminal" in prompt
        assert "forecast_operations" in prompt


# --- Fix 02 add / remove / idempotent ---


class TestR5AddRemoveOperations:
    def test_r5_014_add_creates_new_lineage_generation_1(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="add")
        p08 = _ingest_multi_p08(
            settings,
            run_result,
            source_result,
            tmp_path,
            p05,
            speaker,
            labels=("a",),
            segment_label="add",
        )
        assert p08.status is AiIngestStatus.ACCEPTED, [f"{i.code}: {i.message}" for i in p08.issues]
        _mark_needs_review(settings, p08.artifact_ids[0])
        corrected = {
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
                _forecast_block(
                    source_id=source_result.source_id,
                    speaker=speaker,
                    label="a-new",
                    segment_label="add",
                ),
                _forecast_block(
                    source_id=source_result.source_id,
                    speaker=speaker,
                    label="c-new",
                    segment_label="add",
                ),
            ],
        }
        p09 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-add.json",
                {
                    "schema_version": "2.1.0",
                    "run_id": run_result.run_id,
                    "source_id": source_result.source_id,
                    "reviewed_artifact_id": p08.artifact_ids[0],
                    "prompt_execution": _prompt("P09"),
                    "input_hash": p08.output_hash,
                    "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                    "decision": "correct",
                    "findings": [],
                    "corrected_payload": corrected,
                    "forecast_operations": [
                        {
                            "action": "update",
                            "reviewed_forecast_ref": "forecast-a",
                            "corrected_forecast_ref": "forecast-a-new",
                            "reason": "keep a",
                        },
                        {
                            "action": "add",
                            "corrected_forecast_ref": "forecast-c-new",
                            "reason": "add c",
                        },
                    ],
                },
            ),
        )
        assert p09.status is AiIngestStatus.ACCEPTED, [f"{i.code}: {i.message}" for i in p09.issues]
        sf = create_session_factory(settings.database_file)
        with sf() as session:
            actives = list(
                session.scalars(
                    select(ForecastIssuanceRecord).where(
                        ForecastIssuanceRecord.lifecycle_status == "active"
                    )
                )
            )
            assert len(actives) == 2
            added = next(i for i in actives if i.local_ref == "forecast-c-new")
            assert added.generation == 1
            assert added.lineage_root_id == added.forecast_issuance_id
            assert added.supersedes_forecast_issuance_id is None
            ops = list(session.scalars(select(ForecastCorrectionOperationRecord)))
            assert any(o.action == "add" for o in ops)

    def test_r5_015_remove_withdraws_active_keeps_history(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="rm")
        p08 = _ingest_multi_p08(
            settings,
            run_result,
            source_result,
            tmp_path,
            p05,
            speaker,
            labels=("a", "b"),
            segment_label="rm",
        )
        assert p08.status is AiIngestStatus.ACCEPTED, [f"{i.code}: {i.message}" for i in p08.issues]
        sf = create_session_factory(settings.database_file)
        with sf() as session:
            old_b = session.scalar(
                select(ForecastIssuanceRecord).where(
                    ForecastIssuanceRecord.local_ref == "forecast-b"
                )
            )
            assert old_b is not None
            old_b_id = old_b.forecast_issuance_id
        _mark_needs_review(settings, p08.artifact_ids[0])
        corrected = {
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
                _forecast_block(
                    source_id=source_result.source_id,
                    speaker=speaker,
                    label="a-new",
                    segment_label="rm",
                ),
            ],
        }
        p09 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-remove.json",
                {
                    "schema_version": "2.1.0",
                    "run_id": run_result.run_id,
                    "source_id": source_result.source_id,
                    "reviewed_artifact_id": p08.artifact_ids[0],
                    "prompt_execution": _prompt("P09"),
                    "input_hash": p08.output_hash,
                    "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                    "decision": "correct",
                    "findings": [],
                    "corrected_payload": corrected,
                    "forecast_operations": [
                        {
                            "action": "update",
                            "reviewed_forecast_ref": "forecast-a",
                            "corrected_forecast_ref": "forecast-a-new",
                            "reason": "keep a",
                        },
                        {
                            "action": "remove",
                            "reviewed_forecast_ref": "forecast-b",
                            "reason": "drop b",
                        },
                    ],
                },
            ),
        )
        assert p09.status is AiIngestStatus.ACCEPTED, [f"{i.code}: {i.message}" for i in p09.issues]
        with sf() as session:
            removed = session.get(ForecastIssuanceRecord, old_b_id)
            assert removed is not None
            assert removed.lifecycle_status == "withdrawn_by_correction"
            assert removed.lifecycle_reason == "removed_by_p09_correction"
            actives = list(
                session.scalars(
                    select(ForecastIssuanceRecord).where(
                        ForecastIssuanceRecord.lifecycle_status == "active"
                    )
                )
            )
            assert len(actives) == 1
            assert actives[0].local_ref == "forecast-a-new"
            total = session.scalar(select(func.count()).select_from(ForecastIssuanceRecord))
            assert total == 3  # old a, old b, new a


class TestR5CorrectIdempotent:
    def test_r5_017_same_p09_correct_already_imported(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="idem")
        p08 = _ingest_multi_p08(
            settings,
            run_result,
            source_result,
            tmp_path,
            p05,
            speaker,
            labels=("a", "b"),
            segment_label="idem",
        )
        assert p08.status is AiIngestStatus.ACCEPTED, [f"{i.code}: {i.message}" for i in p08.issues]
        _mark_needs_review(settings, p08.artifact_ids[0])
        corrected = {
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
                _forecast_block(
                    source_id=source_result.source_id,
                    speaker=speaker,
                    label="a-new",
                    segment_label="idem",
                ),
                _forecast_block(
                    source_id=source_result.source_id,
                    speaker=speaker,
                    label="b-new",
                    segment_label="idem",
                ),
            ],
        }
        path = _write(
            tmp_path,
            "p09-idem.json",
            {
                "schema_version": "2.1.0",
                "run_id": run_result.run_id,
                "source_id": source_result.source_id,
                "reviewed_artifact_id": p08.artifact_ids[0],
                "prompt_execution": _prompt("P09"),
                "input_hash": p08.output_hash,
                "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                "decision": "correct",
                "findings": [],
                "corrected_payload": corrected,
                "forecast_operations": [
                    {
                        "action": "update",
                        "reviewed_forecast_ref": "forecast-a",
                        "corrected_forecast_ref": "forecast-a-new",
                        "reason": "fix a",
                    },
                    {
                        "action": "update",
                        "reviewed_forecast_ref": "forecast-b",
                        "corrected_forecast_ref": "forecast-b-new",
                        "reason": "fix b",
                    },
                ],
            },
        )
        first = ingest_ai_output(settings, path)
        assert first.status is AiIngestStatus.ACCEPTED, [
            f"{i.code}: {i.message}" for i in first.issues
        ]
        sf = create_session_factory(settings.database_file)
        with sf() as session:
            before_total = session.scalar(select(func.count()).select_from(ForecastIssuanceRecord))
            before_active = session.scalar(
                select(func.count())
                .select_from(ForecastIssuanceRecord)
                .where(ForecastIssuanceRecord.lifecycle_status == "active")
            )
            before_ops = session.scalar(
                select(func.count()).select_from(ForecastCorrectionOperationRecord)
            )
        second = ingest_ai_output(settings, path)
        assert second.status is AiIngestStatus.ALREADY_IMPORTED
        with sf() as session:
            assert (
                session.scalar(select(func.count()).select_from(ForecastIssuanceRecord))
                == before_total
            )
            assert (
                session.scalar(
                    select(func.count())
                    .select_from(ForecastIssuanceRecord)
                    .where(ForecastIssuanceRecord.lifecycle_status == "active")
                )
                == before_active
            )
            assert (
                session.scalar(select(func.count()).select_from(ForecastCorrectionOperationRecord))
                == before_ops
            )


class TestR5StaleP12:
    def test_r5_021_stale_p12_after_supersede_rejected(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        from tests.unit.test_round4_acceptance import (
            _ingest_p05 as r4_p05,
        )
        from tests.unit.test_round4_acceptance import (
            _ingest_p08 as r4_p08,
        )
        from tests.unit.test_round4_acceptance import (
            _mark_p08_needs_review,
            _p08_payload,
            _p09,
        )

        p05, speaker = r4_p05(settings, run_result, source_result, tmp_path, label="stale12")
        p08 = r4_p08(settings, run_result, source_result, tmp_path, p05, speaker, label="stale12")
        assert p08.status is AiIngestStatus.ACCEPTED
        old_component_id = p08.component_ids[0]
        with create_session_factory(settings.database_file)() as session:
            old_iss = session.get(
                ForecastIssuanceRecord,
                session.get(ForecastComponentRecord, old_component_id).forecast_issuance_id,
            )
            assert old_iss is not None
            old_ref = old_iss.local_ref

        p11 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p11-stale12.json",
                {
                    "schema_version": "2.0.0",
                    "run_id": run_result.run_id,
                    "source_id": source_result.source_id,
                    "forecast_component_id": old_component_id,
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
        assert p11.status is AiIngestStatus.ACCEPTED, [f"{i.code}: {i.message}" for i in p11.issues]

        _mark_p08_needs_review(settings, p08.artifact_ids[0])
        corrected = _p08_payload(
            run_id=run_result.run_id,
            source_id=source_result.source_id,
            upstream_artifact_id=p05.artifact_ids[0],
            input_hash=p05.output_hash,
            speaker=speaker,
            label="stale12-new",
            segment_label="stale12",
        )
        p09 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-stale12.json",
                {
                    **_p09(
                        run_id=run_result.run_id,
                        source_id=source_result.source_id,
                        reviewed_artifact_id=p08.artifact_ids[0],
                        input_hash=p08.output_hash,
                        decision="correct",
                        corrected_payload=corrected,
                        knowledge_cutoff="2026-01-10T09:00:00+00:00",
                    ),
                    "forecast_operations": [
                        {
                            "action": "update",
                            "reviewed_forecast_ref": old_ref,
                            "corrected_forecast_ref": "forecast-stale12-new",
                            "reason": "supersede after p11",
                        }
                    ],
                },
            ),
        )
        assert p09.status is AiIngestStatus.ACCEPTED, [f"{i.code}: {i.message}" for i in p09.issues]

        p12 = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p12-stale.json",
                {
                    "schema_version": "2.0.0",
                    "run_id": run_result.run_id,
                    "source_id": source_result.source_id,
                    "forecast_component_id": old_component_id,
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
                            "rationale": "stale after supersede",
                        }
                    ],
                    "recommended_candidate_ref": "candidate-1",
                    "unevaluable_reason": None,
                },
            ),
        )
        assert p12.status is not AiIngestStatus.ACCEPTED
        assert "inactive_forecast_component" in [i.code for i in p12.issues]


class TestR5ExactTimeBoundary:
    def test_r5_028_equal_ok_plus_one_us_rejected(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        from tests.unit.test_round4_acceptance import _ingest_p05 as r4_p05
        from tests.unit.test_round4_acceptance import _ingest_p08 as r4_p08
        from tests.unit.test_round4_acceptance import _mark_p08_needs_review, _p08_payload

        p05, speaker = r4_p05(settings, run_result, source_result, tmp_path, label="bound")
        p08 = r4_p08(
            settings,
            run_result,
            source_result,
            tmp_path,
            p05,
            speaker,
            label="bound",
            made_at=None,
            made_at_source="unknown",
            publicly_available_at=None,
            knowledge_cutoff="2026-01-10T09:00:00+00:00",
            confidence=0.4,
        )
        _mark_p08_needs_review(settings, p08.artifact_ids[0])
        made_at = "2026-01-10T08:00:00+00:00"
        corrected = _p08_payload(
            run_id=run_result.run_id,
            source_id=source_result.source_id,
            upstream_artifact_id=p05.artifact_ids[0],
            input_hash=p05.output_hash,
            speaker=speaker,
            label="bound-known",
            made_at=made_at,
            knowledge_cutoff=made_at,
        )
        plus_us = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-bound-plus.json",
                {
                    "schema_version": "2.1.0",
                    "run_id": run_result.run_id,
                    "source_id": source_result.source_id,
                    "reviewed_artifact_id": p08.artifact_ids[0],
                    "prompt_execution": _prompt("P09"),
                    "input_hash": p08.output_hash,
                    "knowledge_cutoff": "2026-01-10T08:00:00.000001+00:00",
                    "decision": "correct",
                    "findings": [],
                    "corrected_payload": corrected,
                },
            ),
        )
        assert plus_us.status is not AiIngestStatus.ACCEPTED
        assert any(i.code == "p09_cutoff_exceeds_corrected_made_at" for i in plus_us.issues), [
            f"{i.code}:{i.message}" for i in plus_us.issues
        ]

        equal = ingest_ai_output(
            settings,
            _write(
                tmp_path,
                "p09-bound-eq.json",
                {
                    "schema_version": "2.1.0",
                    "run_id": run_result.run_id,
                    "source_id": source_result.source_id,
                    "reviewed_artifact_id": p08.artifact_ids[0],
                    "prompt_execution": _prompt("P09"),
                    "input_hash": p08.output_hash,
                    "knowledge_cutoff": made_at,
                    "decision": "correct",
                    "findings": [],
                    "corrected_payload": corrected,
                },
            ),
        )
        assert not any(i.code == "p09_cutoff_exceeds_corrected_made_at" for i in equal.issues)


class TestR5BasketAuditCount:
    def test_r5_036_basket_common_date_count_one_saved(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        from tests.unit.test_round4_acceptance import (
            _set_basket_instruments,
            import_locked_component,
        )

        component_id = import_locked_component(
            settings, run_result, source_result, tmp_path, label="r5036"
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
        sf = create_session_factory(settings.database_file)
        with sf() as session:
            ev = session.scalar(
                select(EvaluationRecord).where(
                    EvaluationRecord.forecast_component_id == component_id
                )
            )
            assert ev is not None
            assert ev.common_date_count == 1
            assert ev.coverage_audit is not None
            assert ev.coverage_audit.get("common_date_count") == 1
            assert ev.coverage_audit.get("reason_code") == "insufficient_common_dates"
