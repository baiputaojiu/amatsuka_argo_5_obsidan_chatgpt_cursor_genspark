from datetime import UTC, date, datetime
from pathlib import Path

from sqlalchemy import func, select

from analyst_forecast.application.raw_sources import RawSourceRequest, import_raw_source
from analyst_forecast.application.runs import CreateRunRequest, create_run
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.domain.models import Medium
from analyst_forecast.infrastructure.db.models import RunSourceRecord, SourceRecord
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


def test_same_content_in_another_run_links_existing_source(
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
            url="https://example.invalid/repost",
            published_at=datetime(2026, 1, 11, tzinfo=UTC),
            retrieved_at=datetime(2026, 7, 21, tzinfo=UTC),
        ),
    )

    assert linked.duplicate is True
    assert linked.source_id == source_result.source_id

    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(SourceRecord)) == 1
        assert session.scalar(select(func.count()).select_from(RunSourceRecord)) == 2


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
