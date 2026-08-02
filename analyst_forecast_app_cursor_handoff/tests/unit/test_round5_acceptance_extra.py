"""Round5 extra acceptance coverage: R5-007/014/015/017/023/028/036."""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select
from test_round5_functional import (
    SymbolProvider,
    _bar,
    _forecast_block,
    _ingest_multi_p08,
    _ingest_p05,
    _mark_needs_review,
    _prompt,
    _series,
    _write,
)

from analyst_forecast.application.ai_ingestion import AiIngestStatus, ingest_ai_output
from analyst_forecast.application.evaluation import evaluate_component
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.domain.market import MarketDataRequest, MarketSeries
from analyst_forecast.infrastructure.db.migration import upgrade_database
from analyst_forecast.infrastructure.db.models import (
    EvaluationRecord,
    ForecastComponentRecord,
    ForecastCorrectionOperationRecord,
    ForecastEvidenceRecord,
    ForecastIssuanceRecord,
)
from analyst_forecast.infrastructure.db.session import create_session_factory
from helpers_pipeline_v2 import import_locked_component


def test_r5_014_015_add_and_remove_operations(
    settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
) -> None:
    """R5-014/015: add creates gen-1 lineage; remove withdraws but keeps history."""
    p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="main")
    p08 = _ingest_multi_p08(
        settings, run_result, source_result, tmp_path, p05, speaker, labels=("a", "b")
    )
    assert p08.status is AiIngestStatus.ACCEPTED, [i.message for i in p08.issues]
    _mark_needs_review(settings, p08.artifact_ids[0])

    sf = create_session_factory(settings.database_file)
    with sf() as session:
        old_b = session.scalar(
            select(ForecastIssuanceRecord).where(ForecastIssuanceRecord.local_ref == "forecast-b")
        )
        assert old_b is not None
        old_b_id = old_b.forecast_issuance_id
        before_iss = session.scalar(select(func.count()).select_from(ForecastIssuanceRecord)) or 0
        before_ev = session.scalar(select(func.count()).select_from(ForecastEvidenceRecord)) or 0

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
            _forecast_block(source_id=source_result.source_id, speaker=speaker, label="a-kept"),
            _forecast_block(source_id=source_result.source_id, speaker=speaker, label="c-added"),
        ],
    }
    p09 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p09-add-remove.json",
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
                        "corrected_forecast_ref": "forecast-a-kept",
                        "reason": "keep a via update",
                    },
                    {
                        "action": "remove",
                        "reviewed_forecast_ref": "forecast-b",
                        "corrected_forecast_ref": None,
                        "reason": "drop b",
                    },
                    {
                        "action": "add",
                        "reviewed_forecast_ref": None,
                        "corrected_forecast_ref": "forecast-c-added",
                        "reason": "introduce c",
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

        added = session.scalar(
            select(ForecastIssuanceRecord).where(
                ForecastIssuanceRecord.local_ref == "forecast-c-added"
            )
        )
        assert added is not None
        assert added.lifecycle_status == "active"
        assert added.generation == 1
        assert added.lineage_root_id == added.forecast_issuance_id
        assert added.supersedes_forecast_issuance_id is None

        kept = session.scalar(
            select(ForecastIssuanceRecord).where(
                ForecastIssuanceRecord.local_ref == "forecast-a-kept"
            )
        )
        assert kept is not None
        assert kept.lifecycle_status == "active"
        assert kept.generation >= 2

        actives = list(
            session.scalars(
                select(ForecastIssuanceRecord).where(
                    ForecastIssuanceRecord.lifecycle_status == "active"
                )
            )
        )
        assert len(actives) == 2
        assert {a.local_ref for a in actives} == {"forecast-a-kept", "forecast-c-added"}

        ops = list(session.scalars(select(ForecastCorrectionOperationRecord)))
        actions = {op.action for op in ops}
        assert "add" in actions
        assert "remove" in actions
        remove_ops = [op for op in ops if op.action == "remove"]
        assert len(remove_ops) == 1
        assert remove_ops[0].old_issuance_id == old_b_id
        assert remove_ops[0].new_issuance_id is None

        after_iss = session.scalar(select(func.count()).select_from(ForecastIssuanceRecord)) or 0
        after_ev = session.scalar(select(func.count()).select_from(ForecastEvidenceRecord)) or 0
        assert after_iss >= before_iss + 1  # at least the added issuance
        assert after_ev >= before_ev  # history retained


def test_r5_017_p09_correct_reingest_already_imported(
    settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
) -> None:
    """R5-017: same P09 correct file again → ALREADY_IMPORTED; counts unchanged."""
    p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="main")
    p08 = _ingest_multi_p08(
        settings, run_result, source_result, tmp_path, p05, speaker, labels=("x", "y")
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
            _forecast_block(source_id=source_result.source_id, speaker=speaker, label="x-new"),
            _forecast_block(source_id=source_result.source_id, speaker=speaker, label="y-new"),
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
                    "reviewed_forecast_ref": "forecast-x",
                    "corrected_forecast_ref": "forecast-x-new",
                    "reason": "fix x",
                },
                {
                    "action": "update",
                    "reviewed_forecast_ref": "forecast-y",
                    "corrected_forecast_ref": "forecast-y-new",
                    "reason": "fix y",
                },
            ],
        },
    )
    first = ingest_ai_output(settings, path)
    assert first.status is AiIngestStatus.ACCEPTED, [f"{i.code}: {i.message}" for i in first.issues]

    sf = create_session_factory(settings.database_file)
    with sf() as session:
        before = {
            "issuances": session.scalar(select(func.count()).select_from(ForecastIssuanceRecord)),
            "components": session.scalar(select(func.count()).select_from(ForecastComponentRecord)),
            "ops": session.scalar(
                select(func.count()).select_from(ForecastCorrectionOperationRecord)
            ),
            "actives": list(
                session.scalars(
                    select(ForecastIssuanceRecord).where(
                        ForecastIssuanceRecord.lifecycle_status == "active"
                    )
                )
            ),
        }
        active_ids = sorted(a.forecast_issuance_id for a in before["actives"])

    second = ingest_ai_output(settings, path)
    assert second.status is AiIngestStatus.ALREADY_IMPORTED

    with sf() as session:
        after_actives = list(
            session.scalars(
                select(ForecastIssuanceRecord).where(
                    ForecastIssuanceRecord.lifecycle_status == "active"
                )
            )
        )
        assert (
            session.scalar(select(func.count()).select_from(ForecastIssuanceRecord))
            == before["issuances"]
        )
        assert (
            session.scalar(select(func.count()).select_from(ForecastComponentRecord))
            == before["components"]
        )
        assert (
            session.scalar(select(func.count()).select_from(ForecastCorrectionOperationRecord))
            == before["ops"]
        )
        assert sorted(a.forecast_issuance_id for a in after_actives) == active_ids


def test_r5_028_cutoff_equal_accept_plus_one_us_reject(
    settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
) -> None:
    """R5-028: cutoff == corrected made_at ok; +1µs → p09_cutoff_exceeds_corrected_made_at."""
    from tests.unit.test_round4_acceptance import _ingest_p05 as r4_p05
    from tests.unit.test_round4_acceptance import _ingest_p08 as r4_p08
    from tests.unit.test_round4_acceptance import _mark_p08_needs_review, _p08_payload

    made_at = datetime(2026, 1, 10, 8, 0, 0, tzinfo=UTC)
    equal_cutoff = made_at.isoformat()
    plus_us = (made_at + timedelta(microseconds=1)).isoformat()

    p05, speaker = r4_p05(settings, run_result, source_result, tmp_path, label="eq")
    p08 = r4_p08(
        settings,
        run_result,
        source_result,
        tmp_path,
        p05,
        speaker,
        label="eq",
        made_at=None,
        made_at_source="unknown",
        publicly_available_at=None,
        knowledge_cutoff="2026-01-10T09:00:00+00:00",
        confidence=0.4,
    )
    _mark_p08_needs_review(settings, p08.artifact_ids[0])
    corrected = _p08_payload(
        run_id=run_result.run_id,
        source_id=source_result.source_id,
        upstream_artifact_id=p05.artifact_ids[0],
        input_hash=p05.output_hash,
        speaker=speaker,
        label="eq-known",
        made_at=equal_cutoff,
        knowledge_cutoff=equal_cutoff,
    )

    equal = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p09-eq.json",
            {
                "schema_version": "2.1.0",
                "run_id": run_result.run_id,
                "source_id": source_result.source_id,
                "reviewed_artifact_id": p08.artifact_ids[0],
                "prompt_execution": _prompt("P09"),
                "input_hash": p08.output_hash,
                "knowledge_cutoff": equal_cutoff,
                "decision": "correct",
                "findings": [],
                "corrected_payload": corrected,
            },
        ),
    )
    assert not any(i.code == "p09_cutoff_exceeds_corrected_made_at" for i in equal.issues), [
        f"{i.code}:{i.message}" for i in equal.issues
    ]

    # Fresh unknown P08 for the +1µs reject path (first ingest may have consumed review).
    p05b, speaker_b = r4_p05(settings, run_result, source_result, tmp_path, label="eq2")
    p08b = r4_p08(
        settings,
        run_result,
        source_result,
        tmp_path,
        p05b,
        speaker_b,
        label="eq2",
        made_at=None,
        made_at_source="unknown",
        publicly_available_at=None,
        knowledge_cutoff="2026-01-10T09:00:00+00:00",
        confidence=0.4,
    )
    _mark_p08_needs_review(settings, p08b.artifact_ids[0])
    corrected_b = _p08_payload(
        run_id=run_result.run_id,
        source_id=source_result.source_id,
        upstream_artifact_id=p05b.artifact_ids[0],
        input_hash=p05b.output_hash,
        speaker=speaker_b,
        label="eq2-known",
        made_at=equal_cutoff,
        knowledge_cutoff=equal_cutoff,
    )
    plus = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p09-plus-us.json",
            {
                "schema_version": "2.1.0",
                "run_id": run_result.run_id,
                "source_id": source_result.source_id,
                "reviewed_artifact_id": p08b.artifact_ids[0],
                "prompt_execution": _prompt("P09"),
                "input_hash": p08b.output_hash,
                "knowledge_cutoff": plus_us,
                "decision": "correct",
                "findings": [],
                "corrected_payload": corrected_b,
            },
        ),
    )
    assert plus.status is not AiIngestStatus.ACCEPTED
    assert any(i.code == "p09_cutoff_exceeds_corrected_made_at" for i in plus.issues), [
        f"{i.code}:{i.message}" for i in plus.issues
    ]


def test_r5_028_datetime_boundary_unit() -> None:
    """R5-028 schema-level: equal allowed, strict greater rejects."""
    made_at = datetime(2026, 1, 10, 8, 0, 0, tzinfo=UTC)
    equal = made_at
    plus = made_at + timedelta(microseconds=1)
    assert not (equal > made_at)
    assert plus > made_at


def test_r5_036_basket_one_common_date_persists_count(
    settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
) -> None:
    """R5-036: basket with 1 common date → unevaluable and common_date_count==1 in DB."""
    from tests.unit.test_round4_acceptance import _set_basket_instruments

    component_id = import_locked_component(
        settings, run_result, source_result, tmp_path, label="r5036"
    )
    _set_basket_instruments(settings, component_id)

    class BasketProvider:
        name = "basket-fixture"

        def fetch(self, request: MarketDataRequest) -> MarketSeries:
            series = {
                "AAA": _series("AAA", (_bar(date(2026, 1, 13), "100", "102"),)),
                "BBB": _series("BBB", (_bar(date(2026, 1, 13), "200", "205"),)),
            }.get(request.symbol)
            if series is None:
                from analyst_forecast.domain.market import MarketDataUnavailable

                raise MarketDataUnavailable(f"missing {request.symbol}")
            return series

    result = evaluate_component(
        settings,
        component_id=component_id,
        provider=BasketProvider(),
        as_of=date(2026, 4, 13),
        run_id=run_result.run_id,
    )
    assert result.evaluation_status == "unevaluable"
    assert result.unevaluable_reason is not None
    assert "insufficient_common_dates" in result.unevaluable_reason
    assert result.direction_result is None
    assert result.actual_return is None

    sf = create_session_factory(settings.database_file)
    with sf() as session:
        ev = session.scalar(
            select(EvaluationRecord).where(EvaluationRecord.forecast_component_id == component_id)
        )
        assert ev is not None
        assert ev.common_date_count == 1
        assert ev.coverage_audit is not None
        assert ev.coverage_audit.get("reason_code") == "insufficient_common_dates"
        assert ev.coverage_audit.get("common_date_count") == 1


def test_r5_007_multi_active_lineage_becomes_legacy_conflict(tmp_path: Path) -> None:
    """R5-007: after 0010, multi-active same lineage → legacy_conflict (not auto-active)."""
    database = tmp_path / "conflict.sqlite"
    upgrade_database(database, revision="0009")
    ts = "2026-07-20T00:00:00+00:00"
    hx = "c" * 64
    with sqlite3.connect(database) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(
            f"""
            INSERT INTO analysts (
                analyst_id, canonical_name, normalized_name, aliases, specialties,
                created_at, updated_at, version
            ) VALUES ('A0001', 'A', 'a', '[]', '[]', '{ts}', '{ts}', 1);
            INSERT INTO forecast_groups (
                forecast_group_id, analyst_id, central_thesis, first_issued_at,
                latest_issued_at, current_stance, reaffirmation_count, revision_count,
                withdrawal_status, created_at, updated_at, version
            ) VALUES (
                'FCG-000001', 'A0001', 't', '{ts}', '{ts}', 'up', 0, 0,
                'active', '{ts}', '{ts}', 1
            );
            INSERT INTO sources (
                source_id, analyst_id, medium, url, title, recorded_at, published_at,
                retrieved_at, evidence_level, raw_file_path, raw_hash, acquisition_status,
                source_relation, created_at, updated_at, version
            ) VALUES (
                'SRC-000001', 'A0001', 'youtube', 'https://example.invalid/x', 't',
                '{ts}', '{ts}', '{ts}', 'A', 'raw/x.txt', '{hx}', 'acquired',
                'original', '{ts}', '{ts}', 1
            );
            INSERT INTO forecast_issuances (
                forecast_issuance_id, analyst_id, forecast_group_id, source_id,
                local_ref, made_at, publicly_available_at, forecast_type,
                commitment_strength, evidence_level, extraction_confidence,
                human_readable_summary, relation_to_previous, current_status,
                lifecycle_status, generation, lineage_root_id,
                created_at, updated_at, version
            ) VALUES
            (
                'FCI-000001', 'A0001', 'FCG-000001', 'SRC-000001', 'f1',
                '{ts}', '{ts}', 'directional', 'explicit', 'A', 0.9, 's1', 'initial',
                'active', 'active', 1, 'ROOT-CONFLICT', '{ts}', '{ts}', 1
            ),
            (
                'FCI-000002', 'A0001', 'FCG-000001', 'SRC-000001', 'f2',
                '{ts}', '{ts}', 'directional', 'explicit', 'A', 0.9, 's2', 'initial',
                'active', 'active', 1, 'ROOT-CONFLICT', '{ts}', '{ts}', 1
            );
            """
        )
        conn.commit()
        before = conn.execute(
            "SELECT lifecycle_status FROM forecast_issuances "
            "WHERE lineage_root_id='ROOT-CONFLICT' ORDER BY forecast_issuance_id"
        ).fetchall()
        assert before == [("active",), ("active",)]

    upgrade_database(database, backup_dir=tmp_path / "backups")
    with sqlite3.connect(database) as conn:
        version = conn.execute("SELECT version_num FROM alembic_version").fetchone()[0]
        assert version == "0010"
        rows = conn.execute(
            "SELECT forecast_issuance_id, lifecycle_status, lifecycle_reason "
            "FROM forecast_issuances WHERE lineage_root_id='ROOT-CONFLICT' "
            "ORDER BY forecast_issuance_id"
        ).fetchall()
        assert len(rows) == 2
        assert all(r[1] == "legacy_conflict" for r in rows)
        assert all(r[2] == "round5_multi_active_lineage" for r in rows)
        active_count = conn.execute(
            "SELECT COUNT(*) FROM forecast_issuances "
            "WHERE lineage_root_id='ROOT-CONFLICT' AND lifecycle_status='active'"
        ).fetchone()[0]
        assert active_count == 0


def test_r5_023_evaluate_superseded_raises(
    settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
) -> None:
    """R5-023: evaluate superseded/inactive component → inactive_forecast_component."""
    component_id = import_locked_component(
        settings, run_result, source_result, tmp_path, label="r5023"
    )
    sf = create_session_factory(settings.database_file)
    with sf.begin() as session:
        comp = session.get(ForecastComponentRecord, component_id)
        assert comp is not None
        iss = session.get(ForecastIssuanceRecord, comp.forecast_issuance_id)
        assert iss is not None
        iss.lifecycle_status = "superseded"
        iss.lifecycle_reason = "fixture_superseded_for_r5023"

    provider = SymbolProvider(
        {
            "N225": _series(
                "N225",
                (
                    _bar(date(2026, 1, 13), "100", "100"),
                    _bar(date(2026, 4, 13), "110", "110"),
                ),
            )
        }
    )
    with pytest.raises(ValueError, match="inactive_forecast_component"):
        evaluate_component(
            settings,
            component_id=component_id,
            provider=provider,
            as_of=date(2026, 4, 13),
            run_id=run_result.run_id,
        )
