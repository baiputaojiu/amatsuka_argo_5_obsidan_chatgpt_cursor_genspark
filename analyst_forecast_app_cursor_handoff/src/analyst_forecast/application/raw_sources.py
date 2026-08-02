from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from analyst_forecast.application.settings import AppSettings
from analyst_forecast.domain.models import Medium
from analyst_forecast.infrastructure.db.backup import backup_database
from analyst_forecast.infrastructure.db.ids import next_id
from analyst_forecast.infrastructure.db.models import (
    AiArtifactRecord,
    PromptExecutionRecord,
    RawArtifactRecord,
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
    artifact_reused: bool = False
    raw_artifact_id: str | None = None


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
    created_paths: list[Path] = []
    try:
        with session_factory.begin() as session:
            run = session.get(RunRecord, request.run_id)
            if run is None:
                raise ValueError(
                    f"案件IDが存在しません: {request.run_id}\n"
                    "次の操作: request.yamlのrun_idを確認してください。"
                )
            artifact, artifact_reused, created = _get_or_create_artifact(
                session,
                settings=settings,
                raw_bytes=raw_bytes,
                raw_hash=raw_hash,
                extension=_extension(request.input_path),
            )
            if created is not None:
                created_paths.append(created)

            occurrence = _find_matching_occurrence(
                session,
                artifact_id=artifact.raw_artifact_id,
                analyst_id=run.analyst_id,
                medium=request.medium.value,
                url=request.url,
            )
            occurrence_reused = occurrence is not None
            if occurrence is None:
                occurrence = _create_occurrence(
                    session,
                    run=run,
                    request=request,
                    artifact=artifact,
                    raw_hash=raw_hash,
                )

            link = session.get(
                RunSourceRecord,
                {"run_id": request.run_id, "source_id": occurrence.source_id},
            )
            existing_local: str | None = None
            if link is not None and link.local_input_path:
                local_path = settings.vault_root / Path(link.local_input_path)
                if local_path.is_file():
                    _assert_local_hash(local_path, raw_hash)
                    existing_local = link.local_input_path

            if existing_local is not None:
                result = RawImportResult(
                    source_id=occurrence.source_id,
                    raw_hash=raw_hash,
                    raw_file_path=existing_local,
                    duplicate=True,
                    artifact_reused=True,
                    raw_artifact_id=artifact.raw_artifact_id,
                )
            else:
                local_relative, local_created, manifest_relative = _materialize_run_input(
                    settings=settings,
                    run=run,
                    request=request,
                    source_id=occurrence.source_id,
                    artifact=artifact,
                    raw_bytes=raw_bytes,
                    raw_hash=raw_hash,
                )
                created_paths.extend(local_created)

                if link is None:
                    session.add(
                        RunSourceRecord(
                            run_id=request.run_id,
                            source_id=occurrence.source_id,
                            observed_url=request.url,
                            observed_medium=request.medium.value,
                            observed_published_at=request.published_at,
                            local_input_path=local_relative,
                            input_kind="copy",
                            artifact_manifest_path=manifest_relative,
                        )
                    )
                else:
                    link.observed_url = request.url or link.observed_url
                    link.observed_medium = request.medium.value
                    link.observed_published_at = request.published_at or link.observed_published_at
                    link.local_input_path = local_relative
                    link.input_kind = "copy"
                    link.artifact_manifest_path = manifest_relative

                if not occurrence_reused:
                    occurrence.raw_file_path = local_relative

                link = session.get(
                    RunSourceRecord,
                    {"run_id": request.run_id, "source_id": occurrence.source_id},
                )
                _try_reuse_preprocess_artifact(
                    session,
                    settings,
                    run=run,
                    link=link,
                    source=occurrence,
                    content_hash=raw_hash,
                    medium=request.medium.value,
                )

                result = RawImportResult(
                    source_id=occurrence.source_id,
                    raw_hash=raw_hash,
                    raw_file_path=local_relative,
                    duplicate=False,
                    artifact_reused=artifact_reused or occurrence_reused,
                    raw_artifact_id=artifact.raw_artifact_id,
                )
    except Exception:
        for path in reversed(created_paths):
            path.unlink(missing_ok=True)
        raise

    from analyst_forecast.application.workflow import refresh_workflow

    refresh_workflow(settings, request.run_id)
    return result


def can_reuse_processed_artifact(
    session: Session,
    *,
    content_hash: str,
    prompt_id: str,
    prompt_version: str,
    model: str,
    analyst_id: str,
    speaker_candidate: str | None = None,
) -> AiArtifactRecord | None:
    """同一artifact・処理version・話者条件が一致する場合のみ再利用可能。"""
    artifacts = list(
        session.scalars(
            select(AiArtifactRecord)
            .join(
                PromptExecutionRecord,
                PromptExecutionRecord.prompt_execution_id == AiArtifactRecord.prompt_execution_id,
            )
            .join(SourceRecord, SourceRecord.source_id == AiArtifactRecord.source_id)
            .where(
                AiArtifactRecord.prompt_id == prompt_id,
                AiArtifactRecord.input_hash == content_hash,
                PromptExecutionRecord.prompt_version == prompt_version,
                PromptExecutionRecord.model == model,
                SourceRecord.analyst_id == analyst_id,
                AiArtifactRecord.classification.in_(
                    ("accepted", "processed_with_forecasts", "processed_no_forecast")
                ),
            )
            .order_by(AiArtifactRecord.created_at.desc())
        )
    )
    for artifact in artifacts:
        if speaker_candidate is None:
            return artifact
        payload = artifact.payload or {}
        speakers = {
            str(item.get("speaker_candidate") or item.get("author_candidate"))
            for item in payload.get("segments", [])
            if isinstance(item, dict)
        }
        if speaker_candidate in speakers:
            return artifact
    return None


def _try_reuse_preprocess_artifact(
    session: Session,
    settings: AppSettings,
    *,
    run: RunRecord,
    link: RunSourceRecord | None,
    source: SourceRecord,
    content_hash: str,
    medium: str,
) -> AiArtifactRecord | None:
    """同一raw・同一analyst・同一処理条件ならP05/P07を別runへ関連付ける。"""
    if link is None:
        return None
    if link.latest_ai_artifact_id:
        existing = session.get(AiArtifactRecord, link.latest_ai_artifact_id)
        if existing is not None and existing.classification == "accepted":
            return existing
    model = settings.cursor_model or settings.chatgpt_model
    if not model:
        return None
    prompt_id = "P05" if medium == "youtube" else "P07"
    reused = can_reuse_processed_artifact(
        session,
        content_hash=content_hash,
        prompt_id=prompt_id,
        prompt_version="2.0.0",
        model=model,
        analyst_id=source.analyst_id,
    )
    if reused is None:
        return None
    # 別analystや別source混線防止: source_id一致または同一raw_hash occurrence
    if reused.source_id != source.source_id:
        reused_source = session.get(SourceRecord, reused.source_id)
        if reused_source is None or reused_source.raw_hash != source.raw_hash:
            return None
        if reused_source.analyst_id != source.analyst_id:
            return None
        if reused_source.medium != source.medium:
            return None
        from analyst_forecast.application.artifact_reuse import (
            ReuseError,
            reuse_artifact_for_source,
        )

        try:
            reuse_artifact_for_source(
                session,
                original_artifact_id=reused.ai_artifact_id,
                target_run_id=run.run_id,
                target_source_id=source.source_id,
            )
        except ReuseError:
            return None
    elif source.medium != medium:
        return None
    if reused.prompt_id != prompt_id:
        return None
    link.latest_ai_artifact_id = reused.ai_artifact_id
    link.processing_status = "accepted"
    run_path = settings.vault_root / Path(run.run_path)
    processed = (
        run_path / "02_sources" / medium / "processed" / f"{prompt_id}-{reused.ai_artifact_id}.json"
    )
    processed.parent.mkdir(parents=True, exist_ok=True)
    if not processed.exists():
        processed.write_text(
            json.dumps(reused.payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    manifest = {
        "schema_version": "1.0.0",
        "reuse": True,
        "ai_artifact_id": reused.ai_artifact_id,
        "origin_run_id": reused.run_id,
        "prompt_id": prompt_id,
        "input_hash": reused.input_hash,
        "output_hash": reused.output_hash,
    }
    manifest_path = processed.with_suffix(".reuse.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return reused


def resolve_source_raw_path(
    settings: AppSettings,
    session: Session,
    *,
    run_id: str,
    source: SourceRecord,
) -> Path | None:
    link = session.get(RunSourceRecord, {"run_id": run_id, "source_id": source.source_id})
    candidates: list[Path] = []
    if link is not None and link.local_input_path:
        candidates.append(settings.vault_root / Path(link.local_input_path))
    if source.raw_artifact_id:
        artifact = session.get(RawArtifactRecord, source.raw_artifact_id)
        if artifact is not None:
            candidates.append(settings.vault_root / Path(artifact.canonical_path))
    candidates.append(settings.vault_root / Path(source.raw_file_path))
    for path in candidates:
        if path.is_file():
            return path
    return None


def resolve_run_raw_path(
    settings: AppSettings,
    *,
    run_id: str,
    source_id: str,
) -> Path:
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        link = session.get(RunSourceRecord, {"run_id": run_id, "source_id": source_id})
        source = session.get(SourceRecord, source_id)
        if source is None:
            raise FileNotFoundError(f"SOURCEが存在しません: {source_id}")
        candidates: list[Path] = []
        if link is not None and link.local_input_path:
            candidates.append(settings.vault_root / Path(link.local_input_path))
        if link is not None and link.artifact_manifest_path:
            manifest = settings.vault_root / Path(link.artifact_manifest_path)
            if manifest.is_file():
                data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("canonical_path"):
                    candidates.append(settings.vault_root / Path(str(data["canonical_path"])))
        if source.raw_artifact_id:
            artifact = session.get(RawArtifactRecord, source.raw_artifact_id)
            if artifact is not None:
                candidates.append(settings.vault_root / Path(artifact.canonical_path))
        candidates.append(settings.vault_root / Path(source.raw_file_path))
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"登録済みraw原文が見つかりません: source_id={source_id}\n"
            "次の操作: source importで案件内入力を再作成してください。"
        )


def _get_or_create_artifact(
    session: Session,
    *,
    settings: AppSettings,
    raw_bytes: bytes,
    raw_hash: str,
    extension: str,
) -> tuple[RawArtifactRecord, bool, Path | None]:
    existing = session.scalar(
        select(RawArtifactRecord).where(RawArtifactRecord.content_hash == raw_hash)
    )
    if existing is not None:
        canonical = settings.vault_root / Path(existing.canonical_path)
        if canonical.is_file():
            current = hashlib.sha256(canonical.read_bytes()).hexdigest()
            if current != raw_hash:
                raise ValueError(
                    "raw artifactのhashが一致しません。上書きは禁止です。"
                    f" artifact={existing.raw_artifact_id}"
                )
        return existing, True, None

    artifact_id = next_id(session, "ART-", width=6, sequence_key="RAW_ARTIFACT")
    relative = f"_system/raw_artifacts/{raw_hash[:2]}/{raw_hash}{extension}"
    canonical = settings.vault_root / relative
    canonical.parent.mkdir(parents=True, exist_ok=True)
    with canonical.open("xb") as output:
        output.write(raw_bytes)
    artifact = RawArtifactRecord(
        raw_artifact_id=artifact_id,
        content_hash=raw_hash,
        canonical_path=relative,
        byte_size=len(raw_bytes),
        encoding="utf-8",
        first_seen_at=datetime.now(UTC),
    )
    session.add(artifact)
    session.flush()
    return artifact, False, canonical


def _find_matching_occurrence(
    session: Session,
    *,
    artifact_id: str,
    analyst_id: str,
    medium: str,
    url: str | None,
) -> SourceRecord | None:
    candidates = list(
        session.scalars(
            select(SourceRecord).where(
                SourceRecord.raw_artifact_id == artifact_id,
                SourceRecord.analyst_id == analyst_id,
                SourceRecord.medium == medium,
            )
        )
    )
    normalized_url = (url or "").strip()
    for item in candidates:
        if (item.url or "").strip() == normalized_url:
            return item
    return None


def _create_occurrence(
    session: Session,
    *,
    run: RunRecord,
    request: RawSourceRequest,
    artifact: RawArtifactRecord,
    raw_hash: str,
) -> SourceRecord:
    source_id = next_id(session, "SRC-", width=6, sequence_key="SOURCE")
    placeholder = artifact.canonical_path
    source = SourceRecord(
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
        raw_artifact_id=artifact.raw_artifact_id,
        raw_file_path=placeholder,
        raw_hash=raw_hash,
        acquisition_status="acquired",
        source_relation=request.source_relation,
        original_source_id=request.original_source_id,
    )
    session.add(source)
    session.flush()
    return source


def _materialize_run_input(
    *,
    settings: AppSettings,
    run: RunRecord,
    request: RawSourceRequest,
    source_id: str,
    artifact: RawArtifactRecord,
    raw_bytes: bytes,
    raw_hash: str,
) -> tuple[str, list[Path], str]:
    run_path = settings.vault_root / Path(run.run_path)
    source_date = (request.published_at or request.recorded_at or request.retrieved_at).date()
    extension = _extension(request.input_path)
    filename = f"{source_id}__{source_date:%Y-%m-%d}__{request.medium.value}{extension}"
    raw_path = run_path / "02_sources" / request.medium.value / "raw" / filename
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    if raw_path.exists():
        _assert_local_hash(raw_path, raw_hash)
    else:
        with raw_path.open("xb") as output:
            output.write(raw_bytes)
        created.append(raw_path)

    relative_raw = raw_path.relative_to(settings.vault_root).as_posix()
    metadata_path = (
        run_path / "02_sources" / request.medium.value / "metadata" / f"{source_id}.yaml"
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "1.1.0",
        "source_id": source_id,
        "run_id": request.run_id,
        "raw_artifact_id": artifact.raw_artifact_id,
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
        "raw_file_path": relative_raw,
        "raw_hash": raw_hash,
        "canonical_path": artifact.canonical_path,
        "raw_immutable": True,
        "input_kind": "copy",
    }
    if not metadata_path.exists():
        created.append(metadata_path)
    metadata_path.write_text(
        yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )

    manifest_path = (
        run_path
        / "02_sources"
        / request.medium.value
        / "metadata"
        / f"{source_id}.artifact_ref.json"
    )
    manifest = {
        "schema_version": "1.0.0",
        "source_id": source_id,
        "raw_artifact_id": artifact.raw_artifact_id,
        "content_hash": raw_hash,
        "canonical_path": artifact.canonical_path,
        "local_copy_path": relative_raw,
    }
    if not manifest_path.exists():
        created.append(manifest_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    relative_manifest = manifest_path.relative_to(settings.vault_root).as_posix()
    return relative_raw, created, relative_manifest


def _assert_local_hash(path: Path, expected_hash: str) -> None:
    current = hashlib.sha256(path.read_bytes()).hexdigest()
    if current != expected_hash:
        raise ValueError(
            f"raw原文が変更されています: {path}\n"
            "次の操作: 原文を復元するか、別ファイルとして取り込み直してください。"
        )


def _extension(path: Path) -> str:
    extension = path.suffix.lower()
    if extension not in {".txt", ".md", ".json", ".vtt", ".srt"}:
        return ".txt"
    return extension


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None
