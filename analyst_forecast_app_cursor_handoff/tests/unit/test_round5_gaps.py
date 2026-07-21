"""Round5 remaining GAP acceptance: R5-012/022/025/029/031/033/041/050/052."""

from __future__ import annotations

import ast
import sqlite3
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy import func, select
from test_round5_functional import (
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
from analyst_forecast.application.workflow import refresh_workflow
from analyst_forecast.domain.knowledge_boundary import (
    KnowledgeBoundary,
    validate_cutoff_against_made_at,
    validate_knowledge_cutoff,
)
from analyst_forecast.domain.market import MarketDataRequest, MarketSeries
from analyst_forecast.infrastructure.db.migration import upgrade_database
from analyst_forecast.infrastructure.db.models import (
    AiArtifactRecord,
    ForecastComponentRecord,
    ForecastIssuanceRecord,
    SourceRecord,
    TargetResolutionCandidateRecord,
)
from analyst_forecast.infrastructure.db.session import create_session_factory
from helpers_pipeline_v2 import import_locked_component

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_r5_012_p09_correct_materialize_failure_rolls_back(
    settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
) -> None:
    """R5-012: forced failure mid P09 correct → old active + review rolled back."""
    from tests.unit.test_round4_acceptance import _ingest_p05 as r4_p05
    from tests.unit.test_round4_acceptance import _ingest_p08 as r4_p08
    from tests.unit.test_round4_acceptance import _mark_p08_needs_review, _p08_payload

    p05, speaker = r4_p05(settings, run_result, source_result, tmp_path, label="r5012")
    p08 = r4_p08(settings, run_result, source_result, tmp_path, p05, speaker, label="r5012")
    assert p08.status is AiIngestStatus.ACCEPTED
    _mark_p08_needs_review(settings, p08.artifact_ids[0])

    sf = create_session_factory(settings.database_file)
    with sf() as session:
        old = session.scalar(
            select(ForecastIssuanceRecord).where(
                ForecastIssuanceRecord.ai_artifact_id == p08.artifact_ids[0]
            )
        )
        assert old is not None
        old_id = old.forecast_issuance_id
        old_status = old.lifecycle_status
        before_iss = session.scalar(select(func.count()).select_from(ForecastIssuanceRecord)) or 0
        before_art = session.scalar(select(func.count()).select_from(AiArtifactRecord)) or 0

    corrected = _p08_payload(
        run_id=run_result.run_id,
        source_id=source_result.source_id,
        upstream_artifact_id=p05.artifact_ids[0],
        input_hash=p05.output_hash,
        speaker=speaker,
        label="r5012-new",
        segment_label="r5012",
    )
    path = _write(
        tmp_path,
        "p09-r5012.json",
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
    )

    import analyst_forecast.application.ai_pipeline as pipeline_mod

    real_next_id = pipeline_mod.next_id

    def _boom_after_partial(session: Any, prefix: str, *, width: int = 6, sequence_key: str) -> str:
        if sequence_key == "FORECAST_CORRECTION_OP":
            raise RuntimeError("forced failure after partial P09 materialize")
        return real_next_id(session, prefix, width=width, sequence_key=sequence_key)

    with (
        patch.object(pipeline_mod, "next_id", side_effect=_boom_after_partial),
        pytest.raises(RuntimeError, match=r"(DB取込みに失敗|forced failure)"),
    ):
        ingest_ai_output(settings, path)

    with sf() as session:
        restored = session.get(ForecastIssuanceRecord, old_id)
        assert restored is not None
        assert restored.lifecycle_status == old_status
        assert restored.lifecycle_status == "active"
        dual = session.scalar(
            select(func.count())
            .select_from(ForecastIssuanceRecord)
            .where(
                ForecastIssuanceRecord.lineage_root_id == restored.lineage_root_id,
                ForecastIssuanceRecord.lifecycle_status == "active",
            )
        )
        assert dual == 1
        assert (
            session.scalar(select(func.count()).select_from(ForecastIssuanceRecord)) == before_iss
        )
        assert session.scalar(select(func.count()).select_from(AiArtifactRecord)) == before_art
        p09_reviews = list(
            session.scalars(
                select(AiArtifactRecord).where(
                    AiArtifactRecord.prompt_id == "P09",
                    AiArtifactRecord.run_id == run_result.run_id,
                )
            )
        )
        assert p09_reviews == []


def test_r5_012_unique_index_prevents_dual_active(tmp_path: Path) -> None:
    """R5-012 fallback: unique active lineage index rejects dual actives."""
    database = tmp_path / "r5012-uniq.sqlite"
    upgrade_database(database)
    ts = "2026-07-20T00:00:00+00:00"
    hx = "a" * 64
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
            ) VALUES (
                'FCI-000001', 'A0001', 'FCG-000001', 'SRC-000001', 'f1',
                '{ts}', '{ts}', 'directional', 'explicit', 'A', 0.9, 's', 'initial',
                'active', 'active', 1, 'ROOT-R5012', '{ts}', '{ts}', 1
            );
            """
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                f"""
                INSERT INTO forecast_issuances (
                    forecast_issuance_id, analyst_id, forecast_group_id, source_id,
                    local_ref, made_at, publicly_available_at, forecast_type,
                    commitment_strength, evidence_level, extraction_confidence,
                    human_readable_summary, relation_to_previous, current_status,
                    lifecycle_status, generation, lineage_root_id,
                    created_at, updated_at, version
                ) VALUES (
                    'FCI-000002', 'A0001', 'FCG-000001', 'SRC-000001', 'f2',
                    '{ts}', '{ts}', 'directional', 'explicit', 'A', 0.9, 's', 'initial',
                    'active', 'active', 1, 'ROOT-R5012', '{ts}', '{ts}', 1
                )
                """
            )
        active = conn.execute(
            "SELECT COUNT(*) FROM forecast_issuances "
            "WHERE lineage_root_id='ROOT-R5012' AND lifecycle_status='active'"
        ).fetchone()[0]
        assert active == 1


def test_r5_022_stale_p13_after_supersede_rejected(
    settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
) -> None:
    """R5-022: after P12 accepted, supersede then P13 → inactive_forecast_component."""
    from tests.unit import test_ai_pipeline_v2 as pipe

    component_id = import_locked_component(
        settings, run_result, source_result, tmp_path, label="r5022"
    )
    sf = create_session_factory(settings.database_file)
    with sf() as session:
        p11 = session.scalar(
            select(AiArtifactRecord)
            .where(
                AiArtifactRecord.prompt_id == "P11",
                AiArtifactRecord.run_id == run_result.run_id,
            )
            .order_by(AiArtifactRecord.ai_artifact_id.desc())
        )
        p12 = session.scalar(
            select(AiArtifactRecord)
            .where(
                AiArtifactRecord.prompt_id == "P12",
                AiArtifactRecord.run_id == run_result.run_id,
            )
            .order_by(AiArtifactRecord.ai_artifact_id.desc())
        )
        assert p11 is not None and p12 is not None
        p11_id = p11.ai_artifact_id
        p12_id = p12.ai_artifact_id
        p12_hash = p12.output_hash
        before_cand = session.scalar(
            select(func.count()).select_from(TargetResolutionCandidateRecord)
        )

    with sf.begin() as session:
        comp = session.get(ForecastComponentRecord, component_id)
        assert comp is not None
        iss = session.get(ForecastIssuanceRecord, comp.forecast_issuance_id)
        assert iss is not None
        iss.lifecycle_status = "superseded"
        iss.lifecycle_reason = "fixture_superseded_for_r5022"

    p13 = ingest_ai_output(
        settings,
        pipe._write(
            tmp_path,
            "p13-stale.json",
            pipe._p13_payload(
                run_result.run_id,
                source_result.source_id,
                component_id,
                p11_id,
                p12_id,
                p12_hash,
            ),
        ),
    )
    assert p13.status is not AiIngestStatus.ACCEPTED
    assert any(i.code == "inactive_forecast_component" for i in p13.issues), [
        f"{i.code}: {i.message}" for i in p13.issues
    ]
    with sf() as session:
        after_cand = session.scalar(
            select(func.count()).select_from(TargetResolutionCandidateRecord)
        )
        assert after_cand == before_cand


def test_r5_025_next_actions_omits_superseded_component(
    settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
) -> None:
    """R5-025: after supersede, NEXT_ACTIONS must not list old component as pending P11."""
    from tests.unit.test_round4_acceptance import _ingest_p05 as r4_p05
    from tests.unit.test_round4_acceptance import _ingest_p08 as r4_p08
    from tests.unit.test_round4_acceptance import _mark_p08_needs_review, _p08_payload, _p09

    p05, speaker = r4_p05(settings, run_result, source_result, tmp_path, label="r5025")
    p08 = r4_p08(settings, run_result, source_result, tmp_path, p05, speaker, label="r5025")
    assert p08.status is AiIngestStatus.ACCEPTED
    old_component_id = p08.component_ids[0]

    refresh_workflow(settings, run_result.run_id)
    next_paths = list(settings.vault_root.rglob("NEXT_ACTIONS.md"))
    assert next_paths
    before_text = next_paths[0].read_text(encoding="utf-8")
    assert old_component_id in before_text

    sf = create_session_factory(settings.database_file)
    with sf() as session:
        old_iss = session.get(
            ForecastIssuanceRecord,
            session.get(ForecastComponentRecord, old_component_id).forecast_issuance_id,
        )
        assert old_iss is not None
        old_ref = old_iss.local_ref

    _mark_p08_needs_review(settings, p08.artifact_ids[0])
    corrected = _p08_payload(
        run_id=run_result.run_id,
        source_id=source_result.source_id,
        upstream_artifact_id=p05.artifact_ids[0],
        input_hash=p05.output_hash,
        speaker=speaker,
        label="r5025-new",
        segment_label="r5025",
    )
    p09 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p09-r5025.json",
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
                        "corrected_forecast_ref": "forecast-r5025-new",
                        "reason": "supersede for next_actions",
                    }
                ],
            },
        ),
    )
    assert p09.status is AiIngestStatus.ACCEPTED, [f"{i.code}: {i.message}" for i in p09.issues]

    refresh_workflow(settings, run_result.run_id)
    after_text = next_paths[0].read_text(encoding="utf-8")
    assert old_component_id not in after_text


def test_r5_029_multi_forecast_earliest_made_at_boundary(
    settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
) -> None:
    """R5-029: A=08:00 B=09:00, P09 cutoff=08:30 → p09_cutoff_exceeds_corrected_made_at."""
    p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="r5029")
    p08 = _ingest_multi_p08(
        settings,
        run_result,
        source_result,
        tmp_path,
        p05,
        speaker,
        labels=("a", "b"),
        segment_label="r5029",
    )
    assert p08.status is AiIngestStatus.ACCEPTED, [i.message for i in p08.issues]
    _mark_needs_review(settings, p08.artifact_ids[0])

    made_a = "2026-01-10T08:00:00+00:00"
    made_b = "2026-01-10T09:00:00+00:00"
    cutoff = "2026-01-10T08:30:00+00:00"
    corrected = {
        "schema_version": "2.1.0",
        "run_id": run_result.run_id,
        "source_id": source_result.source_id,
        "upstream_artifact_id": p05.artifact_ids[0],
        "upstream_prompt_id": "P05",
        "prompt_execution": _prompt("P08"),
        "input_hash": p05.output_hash,
        "knowledge_cutoff": made_a,
        "processing_status": "processed_with_forecasts",
        "forecasts": [
            _forecast_block(
                source_id=source_result.source_id,
                speaker=speaker,
                label="a-new",
                made_at=made_a,
                publicly_available_at="2026-01-10T10:00:00+00:00",
                segment_label="r5029",
            ),
            _forecast_block(
                source_id=source_result.source_id,
                speaker=speaker,
                label="b-new",
                made_at=made_b,
                publicly_available_at="2026-01-10T10:00:00+00:00",
                segment_label="r5029",
            ),
        ],
    }
    # Reverse array order — earliest made_at must still bound cutoff.
    corrected["forecasts"] = list(reversed(corrected["forecasts"]))
    p09 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p09-r5029.json",
            {
                "schema_version": "2.1.0",
                "run_id": run_result.run_id,
                "source_id": source_result.source_id,
                "reviewed_artifact_id": p08.artifact_ids[0],
                "prompt_execution": _prompt("P09"),
                "input_hash": p08.output_hash,
                "knowledge_cutoff": cutoff,
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
    assert p09.status is not AiIngestStatus.ACCEPTED
    assert any(i.code == "p09_cutoff_exceeds_corrected_made_at" for i in p09.issues), [
        f"{i.code}: {i.message}" for i in p09.issues
    ]


def test_r5_031_source_boundary_earliest_wins(
    settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
) -> None:
    """R5-031: source recorded_at earlier than made_at; cutoff past source boundary rejected."""
    from tests.unit.test_round4_acceptance import _ingest_p05 as r4_p05
    from tests.unit.test_round4_acceptance import _ingest_p08 as r4_p08
    from tests.unit.test_round4_acceptance import _mark_p08_needs_review, _p08_payload

    p05, speaker = r4_p05(settings, run_result, source_result, tmp_path, label="r5031")
    p08 = r4_p08(
        settings,
        run_result,
        source_result,
        tmp_path,
        p05,
        speaker,
        label="r5031",
        made_at=None,
        made_at_source="unknown",
        publicly_available_at=None,
        knowledge_cutoff="2026-01-10T09:00:00+00:00",
        confidence=0.4,
    )
    assert p08.status in {AiIngestStatus.ACCEPTED, AiIngestStatus.NEEDS_REVIEW}, [
        f"{i.code}: {i.message}" for i in p08.issues
    ]
    assert p08.artifact_ids
    _mark_p08_needs_review(settings, p08.artifact_ids[0])

    # Lower source boundary after P08 so made_at=09:00 is later than recorded_at=07:00.
    sf = create_session_factory(settings.database_file)
    with sf.begin() as session:
        source = session.get(SourceRecord, source_result.source_id)
        assert source is not None
        source.recorded_at = datetime(2026, 1, 10, 7, 0, tzinfo=UTC)
        source.published_at = datetime(2026, 1, 10, 7, 0, tzinfo=UTC)

    made_at = "2026-01-10T09:00:00+00:00"
    # Exceeds source boundary (07:00) but not corrected made_at (09:00).
    late_vs_source = "2026-01-10T08:00:00+00:00"
    corrected = _p08_payload(
        run_id=run_result.run_id,
        source_id=source_result.source_id,
        upstream_artifact_id=p05.artifact_ids[0],
        input_hash=p05.output_hash,
        speaker=speaker,
        label="r5031-known",
        segment_label="r5031",
        made_at=made_at,
        knowledge_cutoff="2026-01-10T07:00:00+00:00",
    )
    p09 = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p09-r5031.json",
            {
                "schema_version": "2.1.0",
                "run_id": run_result.run_id,
                "source_id": source_result.source_id,
                "reviewed_artifact_id": p08.artifact_ids[0],
                "prompt_execution": _prompt("P09"),
                "input_hash": p08.output_hash,
                "knowledge_cutoff": late_vs_source,
                "decision": "correct",
                "findings": [],
                "corrected_payload": corrected,
            },
        ),
    )
    assert p09.status is not AiIngestStatus.ACCEPTED
    assert any(i.code == "p09_cutoff_exceeds_source_boundary" for i in p09.issues), [
        f"{i.code}: {i.message}" for i in p09.issues
    ]

    # Also reject P08 ingest when cutoff exceeds the earlier source boundary.
    p08_late = ingest_ai_output(
        settings,
        _write(
            tmp_path,
            "p08-r5031-late.json",
            {
                "schema_version": "2.1.0",
                "run_id": run_result.run_id,
                "source_id": source_result.source_id,
                "upstream_artifact_id": p05.artifact_ids[0],
                "upstream_prompt_id": "P05",
                "prompt_execution": _prompt("P08"),
                "input_hash": p05.output_hash,
                "knowledge_cutoff": late_vs_source,
                "processing_status": "processed_with_forecasts",
                "forecasts": [
                    _forecast_block(
                        source_id=source_result.source_id,
                        speaker=speaker,
                        label="r5031-late",
                        made_at=made_at,
                        segment_label="r5031",
                    )
                ],
            },
        ),
    )
    assert p08_late.status is not AiIngestStatus.ACCEPTED
    assert any(i.code == "p08_cutoff_exceeds_boundary" for i in p08_late.issues), [
        f"{i.code}: {i.message}" for i in p08_late.issues
    ]


def test_r5_033_shared_cutoff_and_p09_market_prohibition() -> None:
    """R5-033: P09 prompt forbids 市場結果; shared cutoff validators exist."""
    prompt = (
        PROJECT_ROOT / "src" / "analyst_forecast" / "resources" / "prompts" / "P09.md.j2"
    ).read_text(encoding="utf-8")
    assert "市場結果" in prompt

    pipeline = (
        PROJECT_ROOT / "src" / "analyst_forecast" / "application" / "ai_pipeline.py"
    ).read_text(encoding="utf-8")
    assert "validate_knowledge_cutoff" in pipeline
    assert "p09_cutoff_exceeds_corrected_made_at" in pipeline
    assert "p09_cutoff_exceeds_source_boundary" in pipeline

    boundary_made = datetime(2026, 1, 10, 8, 0, tzinfo=UTC)
    assert (
        validate_cutoff_against_made_at(
            boundary_made,
            boundary_made,
            code="p09_cutoff_exceeds_corrected_made_at",
        )
        == []
    )
    bad = validate_cutoff_against_made_at(
        boundary_made + timedelta(microseconds=1),
        boundary_made,
        code="p09_cutoff_exceeds_corrected_made_at",
    )
    assert any(i.code == "p09_cutoff_exceeds_corrected_made_at" for i in bad)

    src_boundary = KnowledgeBoundary(
        boundary=datetime(2026, 1, 10, 7, 0, tzinfo=UTC),
        basis="recorded_at",
        note="test",
    )
    assert (
        validate_knowledge_cutoff(
            datetime(2026, 1, 10, 7, 0, tzinfo=UTC),
            src_boundary,
            code="p09_cutoff_exceeds_source_boundary",
        )
        == []
    )
    src_bad = validate_knowledge_cutoff(
        datetime(2026, 1, 10, 7, 0, 0, 1, tzinfo=UTC),
        src_boundary,
        code="p09_cutoff_exceeds_source_boundary",
    )
    assert any(i.code == "p09_cutoff_exceeds_source_boundary" for i in src_bad)


def test_r5_041_same_day_forecast_unevaluable(
    settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
) -> None:
    """R5-041: normalized_start==normalized_end → single_day_method_not_supported."""
    component_id = import_locked_component(
        settings, run_result, source_result, tmp_path, label="r5041"
    )
    day = date(2026, 1, 13)
    sf = create_session_factory(settings.database_file)
    with sf.begin() as session:
        comp = session.get(ForecastComponentRecord, component_id)
        assert comp is not None
        comp.normalized_start = day
        comp.normalized_end = day

    class SameDayProvider:
        name = "same-day-fixture"

        def fetch(self, request: MarketDataRequest) -> MarketSeries:
            return _series(
                request.symbol,
                (
                    _bar(day, "100", "102"),
                    _bar(date(2026, 1, 14), "110", "111"),
                ),
            )

    result = evaluate_component(
        settings,
        component_id=component_id,
        provider=SameDayProvider(),
        as_of=date(2026, 4, 13),
        run_id=run_result.run_id,
    )
    assert result.evaluation_status == "unevaluable"
    assert result.unevaluable_reason is not None
    assert "single_day_method_not_supported" in result.unevaluable_reason
    assert result.direction_result is None
    assert result.actual_return is None


def test_r5_050_round5_tests_have_no_skip_or_xfail_markers() -> None:
    """R5-050: pytest --collect-only for round5; no skip/xfail on test_round5*."""
    collect = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/test_round5_critical.py",
            "tests/unit/test_round5_functional.py",
            "tests/unit/test_round5_migration.py",
            "tests/unit/test_round5_acceptance_extra.py",
            "tests/unit/test_round5_gaps.py",
            "--collect-only",
            "-q",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert collect.returncode == 0, collect.stdout + collect.stderr
    out = (collect.stdout or "") + (collect.stderr or "")
    assert "test session starts" in out.lower() or "collected" in out.lower() or "test_" in out

    for path in sorted((PROJECT_ROOT / "tests" / "unit").glob("test_round5*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for dec in node.decorator_list:
                text = ast.unparse(dec)
                lowered = text.lower()
                assert "pytest.mark.skip" not in lowered, f"{path.name}:{node.name} {text}"
                assert "pytest.mark.xfail" not in lowered, f"{path.name}:{node.name} {text}"
                assert "pytest.skip(" not in lowered, f"{path.name}:{node.name} {text}"
                assert "pytest.xfail(" not in lowered, f"{path.name}:{node.name} {text}"
            # Also scan function bodies for pytest.skip / pytest.xfail calls.
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    call_txt = ast.unparse(child)
                    if call_txt.startswith("pytest.skip(") or call_txt.startswith("pytest.xfail("):
                        # Allow only inside R5-050 itself if scanning this file's helper — forbid.
                        raise AssertionError(f"{path.name} contains {call_txt}")

    # Body-level scan of round5 modules for skip/xfail (excluding this assertion's strings).
    for path in sorted((PROJECT_ROOT / "tests" / "unit").glob("test_round5*.py")):
        if path.name == "test_round5_gaps.py":
            # Skip self: this file mentions skip/xfail as strings under test.
            continue
        src = path.read_text(encoding="utf-8")
        assert "pytest.skip(" not in src
        assert "pytest.xfail(" not in src
        assert "pytest.mark.skip" not in src
        assert "pytest.mark.xfail" not in src

    wheel = (PROJECT_ROOT / "tests" / "unit" / "test_round4_wheel.py").read_text(encoding="utf-8")
    assert "pytest.xfail" not in wheel


def test_r5_052_chat_history_gitignore_and_secret_hygiene() -> None:
    """R5-052: CHAT_HISTORY.pdf exists; .gitignore protects .env/*.sqlite; secrets untracked."""
    chat = PROJECT_ROOT / "reference" / "CHAT_HISTORY.pdf"
    assert chat.is_file(), f"missing {chat}"
    assert chat.stat().st_size > 0

    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore
    assert "*.sqlite" in gitignore

    status = subprocess.run(
        ["git", "status", "--porcelain", "-u"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert status.returncode == 0, status.stderr
    tracked_forced: list[str] = []
    for ln in (status.stdout or "").splitlines():
        if not ln.strip():
            continue
        path = ln[3:].strip() if len(ln) > 3 else ln
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        lower = path.replace("\\", "/").lower()
        if not ln.startswith("??") and (
            lower.endswith(".env") or lower.endswith(".sqlite") or lower.endswith("/.env")
        ):
            tracked_forced.append(ln)
    assert tracked_forced == [], f"secrets appear force-tracked: {tracked_forced}"

    ls = subprocess.run(
        ["git", "ls-files", ".env", "*.sqlite"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    tracked = [ln for ln in (ls.stdout or "").splitlines() if ln.strip()]
    assert tracked == [], f"secret/db files are tracked: {tracked}"
