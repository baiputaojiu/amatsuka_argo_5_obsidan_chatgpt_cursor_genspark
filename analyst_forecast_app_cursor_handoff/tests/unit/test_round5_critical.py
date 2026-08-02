"""Round5 critical regression tests (R5 migration, gate, cutoff, coverage, reject)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select

from analyst_forecast.application.ai_ingestion import AiIngestStatus, ingest_ai_output
from analyst_forecast.application.evaluation import evaluate_component
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.domain.market import MarketBar, MarketDataRequest, MarketSeries
from analyst_forecast.infrastructure.db.migration import upgrade_database
from analyst_forecast.infrastructure.db.models import (
    ForecastIssuanceRecord,
    TargetResolutionCandidateRecord,
)
from analyst_forecast.infrastructure.db.session import create_session_factory
from analyst_forecast.schemas.pipeline import P09Output
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


class FixedProvider:
    name = "fixture"

    def __init__(self, bars: tuple[MarketBar, ...]) -> None:
        self.bars = bars

    def fetch(self, request: MarketDataRequest) -> MarketSeries:
        return MarketSeries(
            provider=self.name,
            symbol=request.symbol,
            currency=request.currency,
            adjustment_type="split_adjusted_ohlc",
            frequency="1d",
            retrieved_at=datetime(2026, 7, 20, 13, tzinfo=UTC),
            bars=self.bars,
        )


def test_r5_p09_reject_disposition_required() -> None:
    """R5-043: reject without disposition rejected."""
    with pytest.raises(ValidationError, match="reject_disposition"):
        P09Output.model_validate(
            {
                "schema_version": "2.1.0",
                "run_id": "RUN-20260720-001",
                "source_id": "SRC-000001",
                "reviewed_artifact_id": "AIF-000001",
                "prompt_execution": _prompt("P09"),
                "input_hash": "a" * 64,
                "decision": "reject",
                "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                "findings": [],
                "corrected_payload": None,
            }
        )


def test_r5_p09_reject_terminal_field_forbidden_on_2_1() -> None:
    """R5-045: reject_terminal forbidden on 2.1.0."""
    with pytest.raises(ValidationError, match="reject_terminal"):
        P09Output.model_validate(
            {
                "schema_version": "2.1.0",
                "run_id": "RUN-20260720-001",
                "source_id": "SRC-000001",
                "reviewed_artifact_id": "AIF-000001",
                "prompt_execution": _prompt("P09"),
                "input_hash": "a" * 64,
                "decision": "reject",
                "reject_disposition": "retryable",
                "reject_reason": "retry",
                "reject_terminal": False,
                "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                "findings": [],
                "corrected_payload": None,
            }
        )


def test_r5_001_data_bearing_0007_upgrades_to_head(tmp_path: Path) -> None:
    """R5-001/003: child-row 0007 fixture reaches head with FK check empty."""
    database = tmp_path / "bearing.sqlite"
    upgrade_database(database, revision="0007")
    with sqlite3.connect(database) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        now = "2026-07-20T00:00:00+00:00"
        conn.execute(
            """
            INSERT INTO analysts (
                analyst_id, canonical_name, normalized_name, aliases, specialties,
                created_at, updated_at, version
            ) VALUES (?,?,?,?,?,?,?,?)
            """,
            ("A0001", "匿名アナリストA", "匿名アナリストa", "[]", "[]", now, now, 1),
        )
        conn.execute(
            """
            INSERT INTO runs (
                run_id, analyst_id, period_start, period_end, evaluation_as_of,
                selected_media, focus_targets, ai_environment, model_configuration,
                status, run_path, created_at, updated_at, version
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "RUN-20260720-001",
                "A0001",
                "2026-01-01",
                "2026-06-30",
                "2026-07-20",
                '["youtube"]',
                "[]",
                '["cursor"]',
                "{}",
                "not_started",
                "analysts/fixture/RUN-20260720-001",
                now,
                now,
                1,
            ),
        )
        conn.execute(
            """
            INSERT INTO sources (
                source_id, analyst_id, medium, retrieved_at, raw_file_path, raw_hash,
                acquisition_status, source_relation, created_at, updated_at, version
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "SRC-000001",
                "A0001",
                "youtube",
                now,
                "analysts/fixture/raw.txt",
                "b" * 64,
                "acquired",
                "original",
                now,
                now,
                1,
            ),
        )
        conn.execute(
            "INSERT INTO run_sources (run_id, source_id, observed_medium, linked_at) "
            "VALUES (?,?,?,?)",
            ("RUN-20260720-001", "SRC-000001", "youtube", now),
        )
        conn.execute(
            """
            INSERT INTO forecast_groups (
                forecast_group_id, analyst_id, central_thesis, first_issued_at,
                latest_issued_at, current_stance, reaffirmation_count, revision_count,
                withdrawal_status, created_at, updated_at, version
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "FCG-000001",
                "A0001",
                "thesis",
                now,
                now,
                "up",
                0,
                0,
                "active",
                now,
                now,
                1,
            ),
        )
        conn.execute(
            """
            INSERT INTO forecast_issuances (
                forecast_issuance_id, analyst_id, forecast_group_id, source_id, local_ref,
                made_at, publicly_available_at, forecast_type, commitment_strength,
                evidence_level, extraction_confidence, human_readable_summary,
                relation_to_previous, current_status, created_at, updated_at, version
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "FCI-000001",
                "A0001",
                "FCG-000001",
                "SRC-000001",
                "forecast-1",
                now,
                now,
                "directional",
                "explicit",
                "A",
                0.9,
                "summary",
                "initial",
                "active",
                now,
                now,
                1,
            ),
        )
        conn.execute(
            """
            INSERT INTO forecast_components (
                forecast_component_id, forecast_issuance_id, local_ref, sequence_number,
                prediction_form, direction, time_source, normalized_start, normalized_end,
                raw_target_label, target_resolution_status, importance,
                created_at, updated_at, version
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "FCC-000001",
                "FCI-000001",
                "c1",
                1,
                "period_direction",
                "up",
                "explicit",
                "2026-01-13",
                "2026-04-13",
                "日経平均",
                "pending",
                "normal",
                now,
                now,
                1,
            ),
        )
        conn.execute(
            """
            INSERT INTO forecast_evidence (
                forecast_evidence_id, forecast_issuance_id, source_id, quote,
                start_offset, end_offset, role, created_at, updated_at, version
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "FCE-000001",
                "FCI-000001",
                "SRC-000001",
                "quote",
                0,
                5,
                "prediction",
                now,
                now,
                1,
            ),
        )
        conn.commit()
        before_iss = conn.execute("SELECT COUNT(*) FROM forecast_issuances").fetchone()[0]
        before_comp = conn.execute("SELECT COUNT(*) FROM forecast_components").fetchone()[0]

    upgrade_database(database, backup_dir=tmp_path / "backups")
    with sqlite3.connect(database) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        version = conn.execute("SELECT version_num FROM alembic_version").fetchone()[0]
        assert version == "0010"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT COUNT(*) FROM forecast_issuances").fetchone()[0] == before_iss
        assert conn.execute("SELECT COUNT(*) FROM forecast_components").fetchone()[0] == before_comp
        cols = {row[1] for row in conn.execute("PRAGMA table_info(forecast_issuances)")}
        assert "lifecycle_status" in cols
        assert "made_at" in cols


def test_r5_034_single_symbol_one_day_unevaluable(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    """R5-034/035: multi-day forecast with 1 trading day is unevaluable."""
    component_id = import_locked_component(
        settings, run_result, source_result, tmp_path, label="r5cov"
    )
    bar = MarketBar(
        date=date(2026, 1, 13),
        open=Decimal("100"),
        high=Decimal("105"),
        low=Decimal("95"),
        close=Decimal("102"),
        adjusted_open=Decimal("100"),
        adjusted_close=Decimal("102"),
    )
    result = evaluate_component(
        settings,
        component_id=component_id,
        provider=FixedProvider((bar,)),
        as_of=date(2026, 7, 20),
        run_id=run_result.run_id,
    )
    assert result.evaluation_status == "unevaluable"
    assert result.direction_result is None
    assert result.actual_return is None
    assert "insufficient_trading_dates" in (result.unevaluable_reason or "")


def test_r5_019_superseded_component_p11_rejected(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    """R5-019/020: superseded component P11 → inactive_forecast_component."""
    from test_round4_critical import _ingest_p05, _ingest_p08

    p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="r5g")
    p08 = _ingest_p08(settings, run_result, source_result, tmp_path, p05, speaker, label="r5g")
    assert p08.status is AiIngestStatus.ACCEPTED
    component_id = p08.component_ids[0]
    sf = create_session_factory(settings.database_file)
    with sf.begin() as session:
        old = session.scalar(
            select(ForecastIssuanceRecord).where(
                ForecastIssuanceRecord.ai_artifact_id == p08.artifact_ids[0]
            )
        )
        assert old is not None
        old.lifecycle_status = "superseded"
        before = session.scalar(select(func.count()).select_from(TargetResolutionCandidateRecord))

    p11 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p11-inactive.json",
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
    assert p11.status in {AiIngestStatus.NEEDS_REVIEW, AiIngestStatus.REJECTED}, [
        f"{i.code}: {i.message}" for i in p11.issues
    ]
    assert any(i.code == "inactive_forecast_component" for i in p11.issues), [
        f"{i.code}: {i.message}" for i in p11.issues
    ]
    with sf() as session:
        after = session.scalar(select(func.count()).select_from(TargetResolutionCandidateRecord))
        assert after == before
