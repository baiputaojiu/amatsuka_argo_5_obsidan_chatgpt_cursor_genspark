from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy import func, select

from analyst_forecast.application.raw_sources import (
    RawSourceRequest,
    can_reuse_processed_artifact,
    import_raw_source,
)
from analyst_forecast.application.runs import CreateRunRequest, create_run
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.application.workflow import refresh_workflow
from analyst_forecast.domain.models import Medium
from analyst_forecast.infrastructure.db.models import (
    RawArtifactRecord,
    RunSourceRecord,
    SourceRecord,
)
from analyst_forecast.infrastructure.db.session import create_session_factory
from conftest import RAW_TEXT


def test_raw_import_is_append_only_and_idempotent(
    settings: AppSettings,
    run_result,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "first-name.txt"
    original_bytes = RAW_TEXT.encode("utf-8")
    input_path.write_bytes(original_bytes)
    request = RawSourceRequest(
        run_id=run_result.run_id,
        input_path=input_path,
        medium=Medium.YOUTUBE,
        url="https://example.invalid/first",
        recorded_at=datetime(2026, 1, 10, 9, tzinfo=UTC),
        published_at=datetime(2026, 1, 10, 10, tzinfo=UTC),
        retrieved_at=datetime(2026, 7, 20, 12, tzinfo=UTC),
    )

    first = import_raw_source(settings, request)
    raw_path = settings.vault_root / first.raw_file_path
    second = import_raw_source(settings, request)

    assert first.duplicate is False
    assert second.duplicate is True
    assert second.source_id == first.source_id
    assert raw_path.read_bytes() == original_bytes
    assert list((settings.vault_root / "_system" / "backups" / "database").glob("*.sqlite"))

    input_path.write_text("入力側だけを変更", encoding="utf-8")
    assert raw_path.read_bytes() == original_bytes

    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(SourceRecord)) == 1
        assert session.scalar(select(func.count()).select_from(RunSourceRecord)) == 1
        assert session.scalar(select(func.count()).select_from(RawArtifactRecord)) == 1


def test_same_content_same_analyst_another_run_has_local_input(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
) -> None:
    other_run = create_run(
        settings,
        CreateRunRequest(
            canonical_name="匿名アナリストA",
            period_start=date(2026, 2, 1),
            period_end=date(2026, 6, 30),
            evaluation_as_of=date(2026, 7, 20),
            selected_media=[Medium.YOUTUBE],
        ),
        now=datetime(2026, 7, 21, tzinfo=UTC),
    )
    duplicate_path = tmp_path / "same-content.txt"
    duplicate_path.write_text(RAW_TEXT, encoding="utf-8")

    linked = import_raw_source(
        settings,
        RawSourceRequest(
            run_id=other_run.run_id,
            input_path=duplicate_path,
            medium=Medium.YOUTUBE,
            url="https://example.invalid/video/fixture",
            published_at=datetime(2026, 1, 10, 10, tzinfo=UTC),
            retrieved_at=datetime(2026, 7, 21, tzinfo=UTC),
        ),
    )

    assert linked.artifact_reused is True
    assert linked.source_id == source_result.source_id
    local = settings.vault_root / linked.raw_file_path
    assert local.is_file()
    assert local.read_text(encoding="utf-8") == RAW_TEXT

    state = refresh_workflow(settings, other_run.run_id)
    assert any(Path(path).exists() for path in state.recommended_action.inputs if path)


def test_same_bytes_different_analyst_keeps_separate_occurrence(
    settings: AppSettings,
    source_result,
    tmp_path: Path,
) -> None:
    other_run = create_run(
        settings,
        CreateRunRequest(
            canonical_name="匿名アナリストB",
            period_start=date(2026, 1, 1),
            period_end=date(2026, 6, 30),
            evaluation_as_of=date(2026, 7, 20),
            selected_media=[Medium.YOUTUBE],
        ),
        now=datetime(2026, 7, 22, tzinfo=UTC),
    )
    path = tmp_path / "other-analyst.txt"
    path.write_text(RAW_TEXT, encoding="utf-8")
    linked = import_raw_source(
        settings,
        RawSourceRequest(
            run_id=other_run.run_id,
            input_path=path,
            medium=Medium.YOUTUBE,
            url="https://example.invalid/video/fixture",
            published_at=datetime(2026, 1, 10, 10, tzinfo=UTC),
            retrieved_at=datetime(2026, 7, 22, tzinfo=UTC),
        ),
    )
    assert linked.source_id != source_result.source_id
    assert linked.artifact_reused is True
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        first = session.get(SourceRecord, source_result.source_id)
        second = session.get(SourceRecord, linked.source_id)
        assert first is not None and second is not None
        assert first.analyst_id != second.analyst_id
        assert first.raw_artifact_id == second.raw_artifact_id
        assert session.scalar(select(func.count()).select_from(RawArtifactRecord)) == 1


def test_same_bytes_different_url_keeps_both(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
) -> None:
    path = tmp_path / "repost.txt"
    path.write_text(RAW_TEXT, encoding="utf-8")
    second = import_raw_source(
        settings,
        RawSourceRequest(
            run_id=run_result.run_id,
            input_path=path,
            medium=Medium.YOUTUBE,
            url="https://example.invalid/repost",
            published_at=datetime(2026, 1, 11, tzinfo=UTC),
            retrieved_at=datetime(2026, 7, 21, tzinfo=UTC),
        ),
    )
    assert second.source_id != source_result.source_id
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        urls = {
            session.get(SourceRecord, source_result.source_id).url,  # type: ignore[union-attr]
            session.get(SourceRecord, second.source_id).url,  # type: ignore[union-attr]
        }
        assert urls == {
            "https://example.invalid/video/fixture",
            "https://example.invalid/repost",
        }


def test_same_text_different_medium_keeps_evidence(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
) -> None:
    path = tmp_path / "blog.txt"
    path.write_text(RAW_TEXT, encoding="utf-8")
    blog = import_raw_source(
        settings,
        RawSourceRequest(
            run_id=run_result.run_id,
            input_path=path,
            medium=Medium.BLOG,
            url="https://example.invalid/blog",
            published_at=datetime(2026, 1, 12, tzinfo=UTC),
            retrieved_at=datetime(2026, 7, 21, tzinfo=UTC),
        ),
    )
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        source = session.get(SourceRecord, blog.source_id)
        assert source is not None
        assert source.medium == "blog"
        assert (
            source.raw_artifact_id
            == session.get(SourceRecord, source_result.source_id).raw_artifact_id
        )  # type: ignore[union-attr]


def test_raw_tamper_is_rejected(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
) -> None:
    raw_path = settings.vault_root / source_result.raw_file_path
    raw_path.write_text("改変された原文", encoding="utf-8")
    input_path = tmp_path / "retry.txt"
    input_path.write_text(RAW_TEXT, encoding="utf-8")
    with pytest.raises(ValueError, match=r"変更|hash"):
        import_raw_source(
            settings,
            RawSourceRequest(
                run_id=run_result.run_id,
                input_path=input_path,
                medium=Medium.YOUTUBE,
                url="https://example.invalid/video/fixture",
                published_at=datetime(2026, 1, 10, 10, tzinfo=UTC),
                retrieved_at=datetime(2026, 7, 20, 12, 30, tzinfo=UTC),
            ),
        )


def test_processed_reuse_requires_matching_analyst(
    settings: AppSettings,
    source_result,
) -> None:
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        reused = can_reuse_processed_artifact(
            session,
            content_hash=source_result.raw_hash,
            prompt_id="P05",
            prompt_version="1.0.0",
            model="high-performance-fixture",
            analyst_id="A9999",
        )
        assert reused is None


def test_source_dates_and_origin_are_stored_separately(
    settings: AppSettings,
    source_result,
) -> None:
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        source = session.get(SourceRecord, source_result.source_id)
        assert source is not None
        assert source.url == "https://example.invalid/video/fixture"
        assert source.recorded_at != source.published_at
        assert source.retrieved_at > source.published_at
        assert source.raw_artifact_id is not None
