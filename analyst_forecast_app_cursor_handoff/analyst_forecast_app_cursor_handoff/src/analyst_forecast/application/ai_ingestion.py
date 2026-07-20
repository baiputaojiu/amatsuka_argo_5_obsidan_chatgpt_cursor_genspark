from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from pydantic import ValidationError
from sqlalchemy import select

from analyst_forecast.application.settings import AppSettings
from analyst_forecast.infrastructure.db.backup import backup_database
from analyst_forecast.infrastructure.db.ids import next_id
from analyst_forecast.infrastructure.db.models import (
    AiImportRecord,
    ForecastComponentRecord,
    ForecastEvidenceRecord,
    ForecastGroupRecord,
    ForecastIssuanceRecord,
    PromptExecutionRecord,
    RunRecord,
    RunSourceRecord,
    SourceRecord,
    TargetMappingRecord,
    TargetRecord,
)
from analyst_forecast.infrastructure.db.session import create_session_factory
from analyst_forecast.schemas.ai_output import (
    ForecastExtractionOutput,
    TargetMappingOutput,
    schema_path,
)


class AiIngestStatus(StrEnum):
    ACCEPTED = "accepted"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"
    ALREADY_IMPORTED = "already_imported"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    message: str
    path: str = "$"


@dataclass(frozen=True, slots=True)
class AiIngestResult:
    status: AiIngestStatus
    output_hash: str
    issues: tuple[ValidationIssue, ...] = ()
    guidance: str = ""
    forecast_issuance_ids: tuple[str, ...] = ()
    component_ids: tuple[str, ...] = ()


def ingest_ai_output(settings: AppSettings, input_path: Path) -> AiIngestResult:
    if not input_path.is_file():
        raise FileNotFoundError(
            f"AI出力ファイルがありません: {input_path}\n"
            "次の操作: inboxへJSONを保存して再実行してください。"
        )
    raw_bytes = input_path.read_bytes()
    output_hash = hashlib.sha256(raw_bytes).hexdigest()
    issues: list[ValidationIssue] = []
    untyped_payload: dict[str, Any] | None = None

    try:
        loaded = json.loads(raw_bytes.decode("utf-8-sig"))
        if not isinstance(loaded, dict):
            issues.append(
                ValidationIssue("json_root", "JSONの最上位はオブジェクトである必要があります。")
            )
        else:
            untyped_payload = loaded
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        issues.append(
            ValidationIssue(
                "invalid_json",
                f"UTF-8のJSONとして読めません。詳細: {error}",
            )
        )

    if untyped_payload is not None:
        fixed_schema = json.loads(schema_path().read_text(encoding="utf-8"))
        schema_errors = sorted(
            Draft202012Validator(fixed_schema).iter_errors(untyped_payload),
            key=lambda item: list(item.absolute_path),
        )
        for schema_error in schema_errors:
            path = "$" + "".join(f"[{part!r}]" for part in schema_error.absolute_path)
            issues.append(
                ValidationIssue(
                    "json_schema",
                    f"固定JSON Schemaに適合しません。詳細: {schema_error.message}",
                    path,
                )
            )

    payload: ForecastExtractionOutput | None = None
    if untyped_payload is not None and not issues:
        try:
            payload = ForecastExtractionOutput.model_validate(untyped_payload)
        except ValidationError as error:
            for detail in error.errors(include_url=False):
                location = "$" + "".join(f"[{part!r}]" for part in detail["loc"])
                issues.append(
                    ValidationIssue(
                        "pydantic",
                        f"AI出力の値が不正です。詳細: {detail['msg']}",
                        location,
                    )
                )

    run_id = (
        payload.run_id if payload is not None else str((untyped_payload or {}).get("run_id", ""))
    )
    if payload is not None:
        issues.extend(_validate_references_and_quotes(settings, payload))

    if issues:
        _classify_and_audit(
            settings,
            run_id=run_id,
            input_path=input_path,
            raw_bytes=raw_bytes,
            output_hash=output_hash,
            classification=AiIngestStatus.REJECTED,
            issues=issues,
        )
        return AiIngestResult(
            status=AiIngestStatus.REJECTED,
            output_hash=output_hash,
            issues=tuple(issues),
            guidance=(
                "次の操作: エラー箇所を原文とSchemaに照合し、別名の修正版JSONを作成してください。"
            ),
        )

    assert payload is not None
    review_issues = _review_reasons(settings, payload)
    if review_issues:
        _classify_and_audit(
            settings,
            run_id=payload.run_id,
            input_path=input_path,
            raw_bytes=raw_bytes,
            output_hash=output_hash,
            classification=AiIngestStatus.NEEDS_REVIEW,
            issues=review_issues,
        )
        return AiIngestResult(
            status=AiIngestStatus.NEEDS_REVIEW,
            output_hash=output_hash,
            issues=tuple(review_issues),
            guidance=(
                "次の操作: 低確信度または未検証マッピングだけを別の高性能AIでレビューしてください。"
            ),
        )

    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        existing = session.scalar(
            select(AiImportRecord).where(AiImportRecord.output_hash == output_hash)
        )
        if existing is not None:
            existing_component_ids = tuple(
                session.scalars(
                    select(ForecastComponentRecord.forecast_component_id)
                    .join(
                        ForecastIssuanceRecord,
                        ForecastIssuanceRecord.forecast_issuance_id
                        == ForecastComponentRecord.forecast_issuance_id,
                    )
                    .where(ForecastIssuanceRecord.ai_import_id == existing.ai_import_id)
                )
            )
            existing_issuance_ids = tuple(
                session.scalars(
                    select(ForecastIssuanceRecord.forecast_issuance_id).where(
                        ForecastIssuanceRecord.ai_import_id == existing.ai_import_id
                    )
                )
            )
            return AiIngestResult(
                status=AiIngestStatus.ALREADY_IMPORTED,
                output_hash=output_hash,
                guidance="同じAI出力は既に取り込み済みです。次の操作: 状態表示を確認してください。",
                forecast_issuance_ids=existing_issuance_ids,
                component_ids=existing_component_ids,
            )

    backup_database(
        settings.database_file,
        backup_dir=settings.vault_root / "_system" / "backups" / "database",
        reason="ai_import",
    )
    accepted_path = _classify_and_audit(
        settings,
        run_id=payload.run_id,
        input_path=input_path,
        raw_bytes=raw_bytes,
        output_hash=output_hash,
        classification=AiIngestStatus.ACCEPTED,
        issues=[],
    )
    try:
        issuance_ids, component_ids = _insert_payload(
            settings,
            payload=payload,
            output_hash=output_hash,
            classified_path=accepted_path,
        )
    except Exception as error:
        if accepted_path is not None:
            accepted_path.unlink(missing_ok=True)
        transaction_issue = ValidationIssue(
            "database_transaction",
            "SQLite transactionが失敗したため、正式テーブルへの取込みを全て取り消しました。",
        )
        _classify_and_audit(
            settings,
            run_id=payload.run_id,
            input_path=input_path,
            raw_bytes=raw_bytes,
            output_hash=output_hash,
            classification=AiIngestStatus.REJECTED,
            issues=[transaction_issue],
        )
        raise RuntimeError(
            "AI出力のDB取込みに失敗し、transactionをロールバックしました。"
            "次の操作: rejectedの検証記録とエラーログを確認してください。"
        ) from error

    from analyst_forecast.application.workflow import refresh_workflow

    refresh_workflow(settings, payload.run_id)
    return AiIngestResult(
        status=AiIngestStatus.ACCEPTED,
        output_hash=output_hash,
        guidance="検証と取込みが完了しました。次の操作: 市場評価の案内を確認してください。",
        forecast_issuance_ids=tuple(issuance_ids),
        component_ids=tuple(component_ids),
    )


def _validate_references_and_quotes(
    settings: AppSettings,
    payload: ForecastExtractionOutput,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        run = session.get(RunRecord, payload.run_id)
        if run is None:
            return [
                ValidationIssue(
                    "unknown_run",
                    f"案件IDが存在しません: {payload.run_id}",
                    "$['run_id']",
                )
            ]
        source = session.get(SourceRecord, payload.source_id)
        if source is None:
            return [
                ValidationIssue(
                    "unknown_source",
                    f"SOURCE IDが存在しません: {payload.source_id}",
                    "$['source_id']",
                )
            ]
        link = session.get(
            RunSourceRecord,
            {"run_id": payload.run_id, "source_id": payload.source_id},
        )
        if link is None:
            issues.append(
                ValidationIssue(
                    "source_not_in_run",
                    "SOURCE IDは存在しますが、この案件へ関連付けられていません。",
                    "$['source_id']",
                )
            )
        raw_path = settings.vault_root / Path(source.raw_file_path)
        if not raw_path.is_file():
            issues.append(
                ValidationIssue(
                    "raw_missing",
                    f"登録済みraw原文が見つかりません: {source.raw_file_path}",
                )
            )
            return issues
        raw_bytes = raw_path.read_bytes()
        current_hash = hashlib.sha256(raw_bytes).hexdigest()
        if current_hash != source.raw_hash:
            issues.append(
                ValidationIssue(
                    "raw_hash_mismatch",
                    "raw原文のSHA-256が登録時から変化しています。改変された原文は使用できません。",
                )
            )
            return issues
        raw_text = raw_bytes.decode("utf-8-sig")

        for forecast_index, forecast in enumerate(payload.forecasts):
            if forecast.existing_forecast_group_id is not None:
                existing_group = session.get(
                    ForecastGroupRecord, forecast.existing_forecast_group_id
                )
                group_path = f"$['forecasts'][{forecast_index}]['existing_forecast_group_id']"
                if existing_group is None:
                    issues.append(
                        ValidationIssue(
                            "unknown_forecast_group",
                            (
                                "参照された予想グループIDが存在しません: "
                                f"{forecast.existing_forecast_group_id}"
                            ),
                            group_path,
                        )
                    )
                elif existing_group.analyst_id != run.analyst_id:
                    issues.append(
                        ValidationIssue(
                            "forecast_group_analyst_mismatch",
                            "予想グループがこの案件の分析対象者に属していません。",
                            group_path,
                        )
                    )
            for evidence_index, evidence in enumerate(forecast.evidence):
                path = f"$['forecasts'][{forecast_index}]['evidence'][{evidence_index}]"
                if evidence.source_id != payload.source_id:
                    issues.append(
                        ValidationIssue(
                            "source_reference_mismatch",
                            "引用のSOURCE IDが出力最上位のSOURCE IDと一致しません。",
                            f"{path}['source_id']",
                        )
                    )
                    continue
                if evidence.quote not in raw_text:
                    issues.append(
                        ValidationIssue(
                            "quote_missing",
                            "原文に存在しない引用です。",
                            f"{path}['quote']",
                        )
                    )
                    continue
                if raw_text[evidence.start_offset : evidence.end_offset] != evidence.quote:
                    issues.append(
                        ValidationIssue(
                            "quote_offset_mismatch",
                            "引用の文字位置がraw原文と一致しません。",
                            path,
                        )
                    )
    return issues


def _review_reasons(
    settings: AppSettings,
    payload: ForecastExtractionOutput,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for index, forecast in enumerate(payload.forecasts):
        if forecast.extraction_confidence < settings.confidence_review_threshold:
            issues.append(
                ValidationIssue(
                    "low_confidence",
                    (
                        f"抽出確信度{forecast.extraction_confidence:.2f}が"
                        f"設定閾値{settings.confidence_review_threshold:.2f}未満です。"
                    ),
                    f"$['forecasts'][{index}]['extraction_confidence']",
                )
            )
        for component_index, component in enumerate(forecast.components):
            if component.target.mapping_status not in {"verified", "corrected"}:
                issues.append(
                    ValidationIssue(
                        "mapping_not_verified",
                        "評価対象マッピングが検証済みではありません。",
                        (
                            f"$['forecasts'][{index}]['components'][{component_index}]"
                            "['target']['mapping_status']"
                        ),
                    )
                )
    return issues


def _insert_payload(
    settings: AppSettings,
    *,
    payload: ForecastExtractionOutput,
    output_hash: str,
    classified_path: Path | None,
) -> tuple[list[str], list[str]]:
    session_factory = create_session_factory(settings.database_file)
    issuance_ids: list[str] = []
    component_ids: list[str] = []
    with session_factory.begin() as session:
        run = session.get(RunRecord, payload.run_id)
        if run is None:
            raise ValueError(f"案件IDが取込み直前に消失しました: {payload.run_id}")
        ai_import_id = next_id(session, "AII-", width=6, sequence_key="AI_IMPORT")
        relative_classified = (
            classified_path.relative_to(settings.vault_root).as_posix()
            if classified_path is not None
            else ""
        )
        session.add(
            AiImportRecord(
                ai_import_id=ai_import_id,
                run_id=payload.run_id,
                source_id=payload.source_id,
                output_hash=output_hash,
                schema_version=payload.schema_version,
                classified_file_path=relative_classified,
                classification="accepted",
                validation_status="valid",
            )
        )
        session.add(
            PromptExecutionRecord(
                prompt_execution_id=next_id(
                    session, "PEX-", width=6, sequence_key="PROMPT_EXECUTION"
                ),
                ai_import_id=ai_import_id,
                run_id=payload.run_id,
                prompt_id=payload.prompt_execution.prompt_id,
                prompt_version=payload.prompt_execution.prompt_version,
                environment=payload.prompt_execution.environment,
                model=payload.prompt_execution.model,
                input_files=[payload.source_id],
                output_file=relative_classified,
                validation_status="valid",
            )
        )

        groups: dict[str, ForecastGroupRecord] = {}
        for group_ref in dict.fromkeys(
            forecast.forecast_group_ref for forecast in payload.forecasts
        ):
            members = [
                forecast
                for forecast in payload.forecasts
                if forecast.forecast_group_ref == group_ref
            ]
            first_member = min(members, key=lambda item: item.made_at)
            latest_member = max(members, key=lambda item: item.made_at)
            reaffirmations = sum(
                member.relation_to_previous == "reaffirmation" for member in members
            )
            revisions = sum(
                member.relation_to_previous
                in {
                    "strengthened",
                    "weakened",
                    "numeric_revision",
                    "timing_revision",
                    "condition_added",
                    "reversal",
                }
                for member in members
            )
            existing_group_id = first_member.existing_forecast_group_id
            if existing_group_id is not None:
                group = session.get(ForecastGroupRecord, existing_group_id)
                if group is None or group.analyst_id != run.analyst_id:
                    raise ValueError("検証済み予想グループ参照が取込み直前に不整合となりました")
                if _as_utc(first_member.made_at) < _as_utc(group.first_issued_at):
                    group.first_issued_at = first_member.made_at
                if _as_utc(latest_member.made_at) >= _as_utc(group.latest_issued_at):
                    group.latest_issued_at = latest_member.made_at
                    group.current_stance = latest_member.components[0].direction.value
                    if latest_member.relation_to_previous == "withdrawal":
                        group.withdrawal_status = "withdrawn"
                group.reaffirmation_count += reaffirmations
                group.revision_count += revisions
            else:
                group = ForecastGroupRecord(
                    forecast_group_id=next_id(
                        session, "FCG-", width=6, sequence_key="FORECAST_GROUP"
                    ),
                    analyst_id=run.analyst_id,
                    central_thesis=first_member.human_readable_summary,
                    first_issued_at=first_member.made_at,
                    latest_issued_at=latest_member.made_at,
                    current_stance=latest_member.components[0].direction.value,
                    reaffirmation_count=reaffirmations,
                    revision_count=revisions,
                    withdrawal_status=(
                        "withdrawn"
                        if latest_member.relation_to_previous == "withdrawal"
                        else "active"
                    ),
                )
                session.add(group)
            groups[group_ref] = group
        session.flush()

        for forecast in payload.forecasts:
            issuance_id = next_id(session, "FCI-", width=6, sequence_key="FORECAST_ISSUANCE")
            issuance_ids.append(issuance_id)
            current_status = _initial_status(
                run.evaluation_as_of,
                forecast.components[0].normalized_start,
            )
            session.add(
                ForecastIssuanceRecord(
                    forecast_issuance_id=issuance_id,
                    analyst_id=run.analyst_id,
                    forecast_group_id=groups[forecast.forecast_group_ref].forecast_group_id,
                    ai_import_id=ai_import_id,
                    source_id=payload.source_id,
                    local_ref=forecast.forecast_ref,
                    made_at=forecast.made_at,
                    publicly_available_at=forecast.publicly_available_at,
                    forecast_type=forecast.forecast_type,
                    commitment_strength=forecast.commitment_strength,
                    evidence_level=forecast.evidence_level,
                    extraction_confidence=forecast.extraction_confidence,
                    human_readable_summary=forecast.human_readable_summary,
                    relation_to_previous=forecast.relation_to_previous,
                    current_status=current_status,
                )
            )
            for evidence in forecast.evidence:
                session.add(
                    ForecastEvidenceRecord(
                        forecast_evidence_id=next_id(
                            session,
                            "EVD-",
                            width=6,
                            sequence_key="FORECAST_EVIDENCE",
                        ),
                        forecast_issuance_id=issuance_id,
                        source_id=evidence.source_id,
                        quote=evidence.quote,
                        start_offset=evidence.start_offset,
                        end_offset=evidence.end_offset,
                        role=evidence.role,
                    )
                )

            local_component_ids = {
                component.component_ref: next_id(
                    session, "FCC-", width=6, sequence_key="FORECAST_COMPONENT"
                )
                for component in forecast.components
            }
            for component in forecast.components:
                target, mapping = _get_or_create_target_mapping(
                    session,
                    component.target,
                )
                component_id = local_component_ids[component.component_ref]
                component_ids.append(component_id)
                session.add(
                    ForecastComponentRecord(
                        forecast_component_id=component_id,
                        forecast_issuance_id=issuance_id,
                        parent_component_id=(
                            local_component_ids.get(component.parent_component_ref)
                            if component.parent_component_ref
                            else None
                        ),
                        local_ref=component.component_ref,
                        sequence_number=component.sequence_number,
                        prediction_form=component.prediction_form,
                        direction=component.direction.value,
                        time_expression_raw=component.time_expression_raw,
                        time_source=component.time_source.value,
                        normalized_start=component.normalized_start,
                        normalized_end=component.normalized_end,
                        time_precision=component.time_precision,
                        magnitude_value=component.magnitude_value,
                        magnitude_unit=component.magnitude_unit,
                        magnitude_operator=component.magnitude_operator,
                        scenario_probability=component.scenario_probability,
                        target_id=target.target_id,
                        target_mapping_id=mapping.target_mapping_id,
                    )
                )
    return issuance_ids, component_ids


def _get_or_create_target_mapping(
    session: Any,
    target_output: TargetMappingOutput,
) -> tuple[TargetRecord, TargetMappingRecord]:
    target = session.scalar(
        select(TargetRecord).where(
            TargetRecord.canonical_name == target_output.canonical_name,
            TargetRecord.ticker == target_output.symbol,
            TargetRecord.currency == target_output.currency,
        )
    )
    if target is None:
        target = TargetRecord(
            target_id=next_id(session, "TGT-", width=6, sequence_key="TARGET"),
            raw_label=target_output.raw_label,
            canonical_name=target_output.canonical_name,
            target_type=target_output.target_type,
            ticker=target_output.symbol,
            exchange=target_output.exchange,
            currency=target_output.currency,
        )
        session.add(target)
        session.flush()

    mapping_data = target_output.model_dump(mode="json")
    mapping_hash = hashlib.sha256(
        json.dumps(mapping_data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    mapping = session.scalar(
        select(TargetMappingRecord).where(TargetMappingRecord.mapping_hash == mapping_hash)
    )
    if mapping is None:
        mapping = TargetMappingRecord(
            target_mapping_id=next_id(session, "MAP-", width=6, sequence_key="TARGET_MAPPING"),
            target_id=target.target_id,
            mapping_method=target_output.mapping_method,
            evaluation_instruments=[target_output.symbol],
            weights=[1.0],
            knowledge_cutoff=target_output.knowledge_cutoff,
            source_evidence=target_output.source_evidence,
            proposal_model=target_output.proposal_model,
            review_result=target_output.review_result,
            mapping_status=target_output.mapping_status.value,
            mapping_hash=mapping_hash,
            locked_at=datetime.now(UTC),
        )
        session.add(mapping)
        session.flush()
    return target, mapping


def _initial_status(evaluation_as_of: Any, normalized_start: Any) -> str:
    if normalized_start is not None and evaluation_as_of < normalized_start:
        return "not_started"
    return "active_indeterminate"


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _classify_and_audit(
    settings: AppSettings,
    *,
    run_id: str,
    input_path: Path,
    raw_bytes: bytes,
    output_hash: str,
    classification: AiIngestStatus,
    issues: list[ValidationIssue],
) -> Path | None:
    classified = _classification_destination(
        settings,
        run_id=run_id,
        input_path=input_path,
        output_hash=output_hash,
        classification=classification,
    )
    if classified is None:
        return None
    run_path = _run_path(settings, run_id)
    assert run_path is not None
    if not classified.exists():
        classified.parent.mkdir(parents=True, exist_ok=True)
        with classified.open("xb") as output:
            output.write(raw_bytes)

    report_path = (
        run_path / "05_audit" / "processing_logs" / f"AI-{output_hash[:12]}__validation.json"
    )
    report = {
        "schema_version": "1.0.0",
        "output_hash": output_hash,
        "classification": classification.value,
        "validated_at": datetime.now(UTC).isoformat(),
        "issues": [
            {"code": issue.code, "path": issue.path, "message": issue.message} for issue in issues
        ],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return classified


def _classification_destination(
    settings: AppSettings,
    *,
    run_id: str,
    input_path: Path,
    output_hash: str,
    classification: AiIngestStatus,
) -> Path | None:
    run_path = _run_path(settings, run_id)
    if run_path is None:
        return None
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", input_path.stem)[:60] or "ai-output"
    return (
        run_path / "03_ai_outputs" / classification.value / f"{output_hash[:12]}__{safe_name}.json"
    )


def _run_path(settings: AppSettings, run_id: str) -> Path | None:
    if not re.fullmatch(r"RUN-\d{8}-\d{3}", run_id):
        return None
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        run = session.get(RunRecord, run_id)
        return settings.vault_root / Path(run.run_path) if run is not None else None
