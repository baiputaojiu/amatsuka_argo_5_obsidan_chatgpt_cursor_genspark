from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select

from analyst_forecast.application.settings import AppSettings
from analyst_forecast.domain.models import Medium
from analyst_forecast.infrastructure.db.backup import backup_database
from analyst_forecast.infrastructure.db.ids import next_id
from analyst_forecast.infrastructure.db.models import (
    RunRecord,
    RunSourceRecord,
    SourceRecord,
)
from analyst_forecast.infrastructure.db.session import create_session_factory


class RawSourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(pattern=r"^RUN-\d{8}-\d{3}$")
    input_path: Path
    medium: Medium
    url: str | None = None
    external_source_id: str | None = None
    title: str | None = None
    publisher_or_channel: str | None = None
    recorded_at: datetime | None = None
    published_at: datetime | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    evidence_level: str | None = None
    source_relation: str = "original"
    original_source_id: str | None = None

    @field_validator("recorded_at", "published_at", "retrieved_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("日時にはタイムゾーンが必要です")
        return value


@dataclass(frozen=True, slots=True)
class RawImportResult:
    source_id: str
    raw_hash: str
    raw_file_path: str
    duplicate: bool


def import_raw_source(
    settings: AppSettings,
    request: RawSourceRequest,
) -> RawImportResult:
    if not request.input_path.is_file():
        raise FileNotFoundError(
            f"原文ファイルがありません: {request.input_path}\n"
            "次の操作: 入力パスを確認して再実行してください。"
        )
    raw_bytes = request.input_path.read_bytes()
    if not raw_bytes:
        raise ValueError("空の原文は取り込めません。")
    try:
        raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(
            "原文をUTF-8として読めません。元ファイルを保持したまま、"
            "UTF-8版を別ファイルとして用意してください。"
        ) from error
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()

    backup_database(
        settings.database_file,
        backup_dir=settings.vault_root / "_system" / "backups" / "database",
        reason="raw_import",
    )
    session_factory = create_session_factory(settings.database_file)
    created_raw: Path | None = None
    created_metadata: Path | None = None
    try:
        with session_factory.begin() as session:
            run = session.get(RunRecord, request.run_id)
            if run is None:
                raise ValueError(
                    f"案件IDが存在しません: {request.run_id}\n"
                    "次の操作: request.yamlのrun_idを確認してください。"
                )
            existing = session.scalar(select(SourceRecord).where(SourceRecord.raw_hash == raw_hash))
            if existing is not None:
                link = session.get(
                    RunSourceRecord,
                    {"run_id": request.run_id, "source_id": existing.source_id},
                )
                if link is None:
                    session.add(
                        RunSourceRecord(
                            run_id=request.run_id,
                            source_id=existing.source_id,
                            observed_url=request.url,
                            observed_medium=request.medium.value,
                            observed_published_at=request.published_at,
                        )
                    )
                result = RawImportResult(
                    source_id=existing.source_id,
                    raw_hash=existing.raw_hash,
                    raw_file_path=existing.raw_file_path,
                    duplicate=True,
                )
            else:
                source_id = next_id(session, "SRC-", width=6, sequence_key="SOURCE")
                run_path = settings.vault_root / Path(run.run_path)
                source_date = (
                    request.published_at or request.recorded_at or request.retrieved_at
                ).date()
                extension = request.input_path.suffix.lower()
                if extension not in {".txt", ".md", ".json", ".vtt", ".srt"}:
                    extension = ".txt"
                filename = f"{source_id}__{source_date:%Y-%m-%d}__{request.medium.value}{extension}"
                raw_path = run_path / "02_sources" / request.medium.value / "raw" / filename
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                with raw_path.open("xb") as output:
                    output.write(raw_bytes)
                created_raw = raw_path
                relative_raw_path = raw_path.relative_to(settings.vault_root).as_posix()

                session.add(
                    SourceRecord(
                        source_id=source_id,
                        analyst_id=run.analyst_id,
                        medium=request.medium.value,
                        url=request.url,
                        external_source_id=request.external_source_id,
                        title=request.title,
                        publisher_or_channel=request.publisher_or_channel,
                        published_at=request.published_at,
                        recorded_at=request.recorded_at,
                        retrieved_at=request.retrieved_at,
                        evidence_level=request.evidence_level,
                        raw_file_path=relative_raw_path,
                        raw_hash=raw_hash,
                        acquisition_status="acquired",
                        source_relation=request.source_relation,
                        original_source_id=request.original_source_id,
                    )
                )
                session.flush()
                session.add(
                    RunSourceRecord(
                        run_id=request.run_id,
                        source_id=source_id,
                        observed_url=request.url,
                        observed_medium=request.medium.value,
                        observed_published_at=request.published_at,
                    )
                )
                metadata_path = (
                    run_path
                    / "02_sources"
                    / request.medium.value
                    / "metadata"
                    / f"{source_id}.yaml"
                )
                metadata = {
                    "schema_version": "1.0.0",
                    "source_id": source_id,
                    "run_id": request.run_id,
                    "medium": request.medium.value,
                    "url": request.url,
                    "external_source_id": request.external_source_id,
                    "title": request.title,
                    "publisher_or_channel": request.publisher_or_channel,
                    "recorded_at": _iso(request.recorded_at),
                    "published_at": _iso(request.published_at),
                    "retrieved_at": _iso(request.retrieved_at),
                    "evidence_level": request.evidence_level,
                    "source_relation": request.source_relation,
                    "original_source_id": request.original_source_id,
                    "raw_file_path": relative_raw_path,
                    "raw_hash": raw_hash,
                    "raw_immutable": True,
                }
                metadata_path.write_text(
                    yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False),
                    encoding="utf-8",
                    newline="\n",
                )
                created_metadata = metadata_path
                result = RawImportResult(
                    source_id=source_id,
                    raw_hash=raw_hash,
                    raw_file_path=relative_raw_path,
                    duplicate=False,
                )
    except Exception:
        if created_metadata is not None:
            created_metadata.unlink(missing_ok=True)
        if created_raw is not None:
            created_raw.unlink(missing_ok=True)
        raise

    from analyst_forecast.application.workflow import refresh_workflow

    refresh_workflow(settings, request.run_id)
    return result


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None
