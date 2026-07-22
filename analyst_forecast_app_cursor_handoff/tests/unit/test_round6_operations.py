"""Round6 forecast_operations total-mapping contract tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from test_round5_functional import (
    _forecast_block,
    _ingest_multi_p08,
    _ingest_p05,
    _mark_needs_review,
    _prompt,
    _write,
)

from analyst_forecast.application.ai_ingestion import AiIngestStatus, ingest_ai_output
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.infrastructure.db.models import (
    ForecastComponentRecord,
    ForecastCorrectionOperationRecord,
    ForecastIssuanceRecord,
)
from analyst_forecast.infrastructure.db.session import create_session_factory


def _counts(settings: AppSettings) -> dict[str, int]:
    sf = create_session_factory(settings.database_file)
    with sf() as session:
        return {
            "issuances": session.scalar(select(func.count()).select_from(ForecastIssuanceRecord))
            or 0,
            "components": session.scalar(select(func.count()).select_from(ForecastComponentRecord))
            or 0,
            "operations": session.scalar(
                select(func.count()).select_from(ForecastCorrectionOperationRecord)
            )
            or 0,
        }


def _active_ids(settings: AppSettings) -> set[str]:
    sf = create_session_factory(settings.database_file)
    with sf() as session:
        rows = session.scalars(
            select(ForecastIssuanceRecord).where(
                ForecastIssuanceRecord.lifecycle_status == "active"
            )
        ).all()
        return {r.forecast_issuance_id for r in rows}


def _setup_ab(settings, run_result, source_result, tmp_path):
    p05, speaker = _ingest_p05(settings, run_result, source_result, tmp_path, label="main")
    p08 = _ingest_multi_p08(
        settings, run_result, source_result, tmp_path, p05, speaker, labels=("a", "b")
    )
    assert p08.status is AiIngestStatus.ACCEPTED, [i.message for i in p08.issues]
    _mark_needs_review(settings, p08.artifact_ids[0])
    return p05, speaker, p08


def _corrected_payload(run_result, source_result, p05, speaker, labels: tuple[str, ...]):
    return {
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
            _forecast_block(source_id=source_result.source_id, speaker=speaker, label=lab)
            for lab in labels
        ],
    }


def _p09(
    settings,
    tmp_path,
    *,
    run_result,
    source_result,
    p08,
    corrected,
    operations,
    name: str,
):
    return ingest_ai_output(
        settings,
        _write(
            tmp_path,
            name,
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
                "forecast_operations": operations,
            },
        ),
    )


def _assert_rejected_unchanged(settings, before_counts, before_active, result, *, code: str):
    assert result.status is not AiIngestStatus.ACCEPTED
    codes = {i.code for i in result.issues}
    assert code in codes or any(code in (i.message or "") for i in result.issues), (
        codes,
        [i.message for i in result.issues],
    )
    assert _counts(settings) == before_counts
    assert _active_ids(settings) == before_active


class TestR6OperationsNegative:
    def test_r6_005_many_to_one_rejected(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker, p08 = _setup_ab(settings, run_result, source_result, tmp_path)
        before_c, before_a = _counts(settings), _active_ids(settings)
        corrected = _corrected_payload(run_result, source_result, p05, speaker, ("x",))
        result = _p09(
            settings,
            tmp_path,
            run_result=run_result,
            source_result=source_result,
            p08=p08,
            corrected=corrected,
            operations=[
                {
                    "action": "update",
                    "reviewed_forecast_ref": "forecast-a",
                    "corrected_forecast_ref": "forecast-x",
                    "reason": "map a",
                },
                {
                    "action": "update",
                    "reviewed_forecast_ref": "forecast-b",
                    "corrected_forecast_ref": "forecast-x",
                    "reason": "map b",
                },
            ],
            name="p09-many-to-one.json",
        )
        _assert_rejected_unchanged(
            settings,
            before_c,
            before_a,
            result,
            code="duplicate_operation_corrected_ref",
        )

    def test_r6_008_009_undeclared_old_and_new(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker, p08 = _setup_ab(settings, run_result, source_result, tmp_path)
        before_c, before_a = _counts(settings), _active_ids(settings)
        corrected = _corrected_payload(run_result, source_result, p05, speaker, ("a2", "b2", "c"))
        result = _p09(
            settings,
            tmp_path,
            run_result=run_result,
            source_result=source_result,
            p08=p08,
            corrected=corrected,
            operations=[
                {
                    "action": "update",
                    "reviewed_forecast_ref": "forecast-a",
                    "corrected_forecast_ref": "forecast-a2",
                    "reason": "update a",
                },
                {
                    "action": "add",
                    "reviewed_forecast_ref": None,
                    "corrected_forecast_ref": "forecast-c",
                    "reason": "add c",
                },
            ],
            name="p09-undeclared.json",
        )
        codes = {i.code for i in result.issues}
        assert "incomplete_reviewed_forecast_coverage" in codes
        assert "incomplete_corrected_forecast_coverage" in codes
        assert _counts(settings) == before_c
        assert _active_ids(settings) == before_a

    def test_r6_007_unknown_refs(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker, p08 = _setup_ab(settings, run_result, source_result, tmp_path)
        before_c, before_a = _counts(settings), _active_ids(settings)
        corrected = _corrected_payload(run_result, source_result, p05, speaker, ("a2", "b2"))
        result = _p09(
            settings,
            tmp_path,
            run_result=run_result,
            source_result=source_result,
            p08=p08,
            corrected=corrected,
            operations=[
                {
                    "action": "update",
                    "reviewed_forecast_ref": "forecast-missing",
                    "corrected_forecast_ref": "forecast-a2",
                    "reason": "bad old",
                },
                {
                    "action": "update",
                    "reviewed_forecast_ref": "forecast-b",
                    "corrected_forecast_ref": "forecast-ghost",
                    "reason": "bad new",
                },
            ],
            name="p09-unknown.json",
        )
        codes = {i.code for i in result.issues}
        assert "unknown_reviewed_forecast_ref" in codes
        assert "unknown_corrected_forecast_ref" in codes
        assert _counts(settings) == before_c
        assert _active_ids(settings) == before_a

    def test_r6_005_one_to_many_rejected(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker, p08 = _setup_ab(settings, run_result, source_result, tmp_path)
        before_c, before_a = _counts(settings), _active_ids(settings)
        # Only one old (a) reviewed if we use single? Need A only reviewed with A→X, A→Y
        # Use multi A/B reviewed but ops that map A to two news — need corrected X,Y and leave B
        # Simpler: reviewed A only via single-label setup... Use A/B with ops update A→X, A→Y and
        # something for B — duplicate reviewed ref is the key.
        corrected = _corrected_payload(run_result, source_result, p05, speaker, ("x", "y"))
        result = _p09(
            settings,
            tmp_path,
            run_result=run_result,
            source_result=source_result,
            p08=p08,
            corrected=corrected,
            operations=[
                {
                    "action": "update",
                    "reviewed_forecast_ref": "forecast-a",
                    "corrected_forecast_ref": "forecast-x",
                    "reason": "map a-x",
                },
                {
                    "action": "update",
                    "reviewed_forecast_ref": "forecast-a",
                    "corrected_forecast_ref": "forecast-y",
                    "reason": "map a-y",
                },
            ],
            name="p09-one-to-many.json",
        )
        _assert_rejected_unchanged(
            settings,
            before_c,
            before_a,
            result,
            code="duplicate_operation_reviewed_ref",
        )

    def test_r6_008_undeclared_old_only(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker, p08 = _setup_ab(settings, run_result, source_result, tmp_path)
        before_c, before_a = _counts(settings), _active_ids(settings)
        corrected = _corrected_payload(run_result, source_result, p05, speaker, ("a2",))
        result = _p09(
            settings,
            tmp_path,
            run_result=run_result,
            source_result=source_result,
            p08=p08,
            corrected=corrected,
            operations=[
                {
                    "action": "update",
                    "reviewed_forecast_ref": "forecast-a",
                    "corrected_forecast_ref": "forecast-a2",
                    "reason": "only a",
                },
            ],
            name="p09-missing-old-b.json",
        )
        _assert_rejected_unchanged(
            settings,
            before_c,
            before_a,
            result,
            code="incomplete_reviewed_forecast_coverage",
        )

    def test_r6_009_undeclared_new_only(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker, p08 = _setup_ab(settings, run_result, source_result, tmp_path)
        before_c, before_a = _counts(settings), _active_ids(settings)
        corrected = _corrected_payload(run_result, source_result, p05, speaker, ("a2", "b2"))
        result = _p09(
            settings,
            tmp_path,
            run_result=run_result,
            source_result=source_result,
            p08=p08,
            corrected=corrected,
            operations=[
                {
                    "action": "update",
                    "reviewed_forecast_ref": "forecast-a",
                    "corrected_forecast_ref": "forecast-a2",
                    "reason": "a",
                },
                {
                    "action": "remove",
                    "reviewed_forecast_ref": "forecast-b",
                    "corrected_forecast_ref": None,
                    "reason": "drop b",
                },
            ],
            name="p09-missing-new-b2.json",
        )
        _assert_rejected_unchanged(
            settings,
            before_c,
            before_a,
            result,
            code="incomplete_corrected_forecast_coverage",
        )

    def test_r6_006_blank_reason_rejected(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker, p08 = _setup_ab(settings, run_result, source_result, tmp_path)
        before_c, before_a = _counts(settings), _active_ids(settings)
        corrected = _corrected_payload(run_result, source_result, p05, speaker, ("a2", "b2"))
        result = _p09(
            settings,
            tmp_path,
            run_result=run_result,
            source_result=source_result,
            p08=p08,
            corrected=corrected,
            operations=[
                {
                    "action": "update",
                    "reviewed_forecast_ref": "forecast-a",
                    "corrected_forecast_ref": "forecast-a2",
                    "reason": "   ",
                },
                {
                    "action": "update",
                    "reviewed_forecast_ref": "forecast-b",
                    "corrected_forecast_ref": "forecast-b2",
                    "reason": "ok",
                },
            ],
            name="p09-blank-reason.json",
        )
        assert result.status is not AiIngestStatus.ACCEPTED
        assert _counts(settings) == before_c
        assert _active_ids(settings) == before_a


class TestR6OperationsPositive:
    def test_r6_013_update_remove_add(
        self, settings: AppSettings, run_result: Any, source_result: Any, tmp_path: Path
    ):
        p05, speaker, p08 = _setup_ab(settings, run_result, source_result, tmp_path)
        corrected = _corrected_payload(run_result, source_result, p05, speaker, ("a2", "c"))
        result = _p09(
            settings,
            tmp_path,
            run_result=run_result,
            source_result=source_result,
            p08=p08,
            corrected=corrected,
            operations=[
                {
                    "action": "update",
                    "reviewed_forecast_ref": "forecast-a",
                    "corrected_forecast_ref": "forecast-a2",
                    "reason": "update a",
                },
                {
                    "action": "remove",
                    "reviewed_forecast_ref": "forecast-b",
                    "corrected_forecast_ref": None,
                    "reason": "remove b",
                },
                {
                    "action": "add",
                    "reviewed_forecast_ref": None,
                    "corrected_forecast_ref": "forecast-c",
                    "reason": "add c",
                },
            ],
            name="p09-mixed.json",
        )
        assert result.status is AiIngestStatus.ACCEPTED, [i.message for i in result.issues]
        sf = create_session_factory(settings.database_file)
        with sf() as session:
            actives = session.scalars(
                select(ForecastIssuanceRecord).where(
                    ForecastIssuanceRecord.lifecycle_status == "active"
                )
            ).all()
            assert {a.local_ref for a in actives} == {"forecast-a2", "forecast-c"}
            removed = session.scalars(
                select(ForecastIssuanceRecord).where(
                    ForecastIssuanceRecord.local_ref == "forecast-b"
                )
            ).all()
            assert removed and all(r.lifecycle_status == "withdrawn_by_correction" for r in removed)
