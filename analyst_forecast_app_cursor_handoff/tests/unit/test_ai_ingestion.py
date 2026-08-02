from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import func, select

from analyst_forecast.application import ai_ingestion as ai_ingestion_module
from analyst_forecast.application.ai_ingestion import AiIngestStatus, ingest_ai_output
from analyst_forecast.application.raw_sources import (
    RawSourceRequest,
    import_raw_source,
)
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.domain.models import Medium
from analyst_forecast.infrastructure.db.models import (
    AiImportRecord,
    ForecastComponentRecord,
    ForecastGroupRecord,
    ForecastIssuanceRecord,
)
from analyst_forecast.infrastructure.db.session import create_session_factory
from analyst_forecast.schemas.ai_output import ForecastExtractionOutput, schema_path
from conftest import make_ai_payload

FIXTURES = Path(__file__).parents[1] / "fixtures"


def write_payload(tmp_path: Path, payload: dict, name: str = "ai-output.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_fixed_json_schema_matches_pydantic_model() -> None:
    fixed_schema = json.loads(schema_path().read_text(encoding="utf-8"))
    generated = ForecastExtractionOutput.model_json_schema()

    assert fixed_schema["$id"] == "https://local.invalid/schemas/forecast-extraction-1.0.0.json"
    assert fixed_schema == generated


def test_anonymous_fixed_fixture_matches_schema_and_raw_quote() -> None:
    raw_text = (FIXTURES / "raw" / "anonymous_analyst_a.txt").read_text(encoding="utf-8")
    payload = json.loads(
        (FIXTURES / "ai" / "forecast_extraction_anonymous.json").read_text(encoding="utf-8")
    )

    parsed = ForecastExtractionOutput.model_validate(payload)
    evidence = parsed.forecasts[0].evidence[0]

    assert raw_text[evidence.start_offset : evidence.end_offset] == evidence.quote


def test_valid_ai_output_is_transactionally_imported_once(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
) -> None:
    payload = make_ai_payload(run_id=run_result.run_id, source_id=source_result.source_id)
    path = write_payload(tmp_path, payload)

    first = ingest_ai_output(settings, path)
    second = ingest_ai_output(settings, path)

    assert first.status is AiIngestStatus.ACCEPTED
    assert second.status is AiIngestStatus.ALREADY_IMPORTED
    assert first.output_hash == second.output_hash

    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(AiImportRecord)) == 1
        assert session.scalar(select(func.count()).select_from(ForecastGroupRecord)) == 1
        assert session.scalar(select(func.count()).select_from(ForecastIssuanceRecord)) == 1
        assert session.scalar(select(func.count()).select_from(ForecastComponentRecord)) == 1


def test_unknown_source_reference_is_rejected_before_formal_tables(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
) -> None:
    payload = make_ai_payload(run_id=run_result.run_id, source_id="SRC-999999")
    path = write_payload(tmp_path, payload)

    result = ingest_ai_output(settings, path)

    assert result.status is AiIngestStatus.REJECTED
    assert any("SOURCE ID" in issue.message for issue in result.issues)
    assert "次の操作" in result.guidance

    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ForecastIssuanceRecord)) == 0


def test_missing_quote_rejects_entire_batch(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
) -> None:
    payload = make_ai_payload(run_id=run_result.run_id, source_id=source_result.source_id)
    second = copy.deepcopy(payload["forecasts"][0])
    second["forecast_ref"] = "forecast-2"
    second["components"][0]["component_ref"] = "component-2"
    second["evidence"][0]["quote"] = "原文には存在しない引用"
    payload["forecasts"].append(second)

    result = ingest_ai_output(settings, write_payload(tmp_path, payload))

    assert result.status is AiIngestStatus.REJECTED
    assert any("引用" in issue.message for issue in result.issues)
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ForecastIssuanceRecord)) == 0


def test_wrong_quote_offsets_are_rejected(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
) -> None:
    payload = make_ai_payload(run_id=run_result.run_id, source_id=source_result.source_id)
    payload["forecasts"][0]["evidence"][0]["start_offset"] = 1

    result = ingest_ai_output(settings, write_payload(tmp_path, payload))

    assert result.status is AiIngestStatus.REJECTED
    assert any("位置" in issue.message for issue in result.issues)


def test_modified_raw_file_is_rejected_by_registered_hash(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
) -> None:
    raw_path = settings.vault_root / source_result.raw_file_path
    raw_path.write_text("改変された原文", encoding="utf-8")
    payload = make_ai_payload(run_id=run_result.run_id, source_id=source_result.source_id)

    result = ingest_ai_output(settings, write_payload(tmp_path, payload))

    assert result.status is AiIngestStatus.REJECTED
    assert any(issue.code == "raw_hash_mismatch" for issue in result.issues)


def test_low_confidence_output_waits_for_review(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
) -> None:
    payload = make_ai_payload(
        run_id=run_result.run_id,
        source_id=source_result.source_id,
        confidence=0.50,
    )

    result = ingest_ai_output(settings, write_payload(tmp_path, payload))

    assert result.status is AiIngestStatus.NEEDS_REVIEW
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ForecastIssuanceRecord)) == 0


def test_verified_target_mapping_requires_independent_ai_review(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
) -> None:
    payload = make_ai_payload(run_id=run_result.run_id, source_id=source_result.source_id)
    del payload["forecasts"][0]["components"][0]["target"]["review_result"]

    result = ingest_ai_output(settings, write_payload(tmp_path, payload))

    assert result.status is AiIngestStatus.REJECTED
    assert any("別AI" in issue.message for issue in result.issues)


def test_schema_1_0_0_inline_review_does_not_lock_mapping(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
) -> None:
    from analyst_forecast.infrastructure.db.models import TargetMappingRecord

    payload = make_ai_payload(run_id=run_result.run_id, source_id=source_result.source_id)
    result = ingest_ai_output(settings, write_payload(tmp_path, payload))

    assert result.status is AiIngestStatus.ACCEPTED
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        mapping = session.scalar(select(TargetMappingRecord))
        assert mapping is not None
        assert mapping.mapping_status == "legacy_inline_review"
        assert mapping.locked_at is None
        assert mapping.review_result is not None
        assert "[legacy_inline_review" in mapping.review_result
        component = session.get(ForecastComponentRecord, result.component_ids[0])
        assert component is not None
        assert component.target_resolution_status == "review_pending"


def test_restatements_are_separate_issuances_in_one_group(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
) -> None:
    payload = make_ai_payload(run_id=run_result.run_id, source_id=source_result.source_id)
    reaffirmation = copy.deepcopy(payload["forecasts"][0])
    reaffirmation["forecast_ref"] = "forecast-2"
    reaffirmation["relation_to_previous"] = "reaffirmation"
    reaffirmation["made_at"] = "2026-02-10T09:00:00+00:00"
    reaffirmation["publicly_available_at"] = "2026-02-10T10:00:00+00:00"
    reaffirmation["components"][0]["component_ref"] = "component-2"
    reaffirmation["components"][0]["normalized_start"] = "2026-02-12"
    payload["forecasts"].append(reaffirmation)

    result = ingest_ai_output(settings, write_payload(tmp_path, payload))

    assert result.status is AiIngestStatus.ACCEPTED
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ForecastGroupRecord)) == 1
        assert session.scalar(select(func.count()).select_from(ForecastIssuanceRecord)) == 2


def test_later_ai_output_can_reference_existing_forecast_group(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
) -> None:
    first_payload = make_ai_payload(run_id=run_result.run_id, source_id=source_result.source_id)
    first = ingest_ai_output(settings, write_payload(tmp_path, first_payload, "first.json"))
    assert first.status is AiIngestStatus.ACCEPTED

    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        group_id = session.scalar(select(ForecastGroupRecord.forecast_group_id))
    assert group_id is not None

    later_raw = "日経平均は今後上昇する。前回と見方は変わらない。"
    later_path = tmp_path / "later-source.txt"
    later_path.write_text(later_raw, encoding="utf-8")
    later_source = import_raw_source(
        settings,
        RawSourceRequest(
            run_id=run_result.run_id,
            input_path=later_path,
            medium=Medium.YOUTUBE,
            url="https://example.invalid/video/later",
            recorded_at=datetime(2026, 2, 10, 9, tzinfo=UTC),
            published_at=datetime(2026, 2, 10, 10, tzinfo=UTC),
            retrieved_at=datetime(2026, 7, 20, 14, tzinfo=UTC),
        ),
    )
    later_payload = make_ai_payload(
        run_id=run_result.run_id,
        source_id=later_source.source_id,
        raw_text=later_raw,
    )
    later_forecast = later_payload["forecasts"][0]
    later_forecast["existing_forecast_group_id"] = group_id
    later_forecast["relation_to_previous"] = "reaffirmation"
    later_forecast["made_at"] = "2026-02-10T09:00:00+00:00"
    later_forecast["publicly_available_at"] = "2026-02-10T10:00:00+00:00"
    later_forecast["components"][0]["normalized_start"] = "2026-02-12"

    later = ingest_ai_output(settings, write_payload(tmp_path, later_payload, "later.json"))

    assert later.status is AiIngestStatus.ACCEPTED
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ForecastGroupRecord)) == 1
        assert session.scalar(select(func.count()).select_from(ForecastIssuanceRecord)) == 2
        group = session.get(ForecastGroupRecord, group_id)
        assert group is not None
        assert group.reaffirmation_count == 1


def test_database_failure_rolls_back_batch_and_removes_accepted_copy(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = make_ai_payload(run_id=run_result.run_id, source_id=source_result.source_id)
    path = write_payload(tmp_path, payload)

    def fail_target_mapping(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("意図したtransaction失敗")

    monkeypatch.setattr(
        ai_ingestion_module,
        "_get_or_create_target_mapping",
        fail_target_mapping,
    )

    with pytest.raises(RuntimeError, match="ロールバック"):
        ingest_ai_output(settings, path)

    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(AiImportRecord)) == 0
        assert session.scalar(select(func.count()).select_from(ForecastIssuanceRecord)) == 0
    assert not list((run_result.run_path / "03_ai_outputs" / "accepted").glob("*.json"))
    assert list((run_result.run_path / "03_ai_outputs" / "rejected").glob("*.json"))
