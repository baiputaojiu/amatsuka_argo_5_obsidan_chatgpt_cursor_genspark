from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from analyst_forecast.application.ai_ingestion import (
    AiIngestResult,
    AiIngestStatus,
    ValidationIssue,
)
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.infrastructure.db.backup import backup_database
from analyst_forecast.infrastructure.db.ids import next_id
from analyst_forecast.infrastructure.db.models import (
    AiArtifactRecord,
    ForecastComponentRecord,
    ForecastEvidenceRecord,
    ForecastGroupRecord,
    ForecastIssuanceRecord,
    PromptExecutionRecord,
    RunRecord,
    RunSourceRecord,
    SegmentRecord,
    SourceRecord,
    TargetMappingRecord,
    TargetRecord,
    TargetResolutionAdjudicationRecord,
    TargetResolutionCandidateRecord,
    TargetResolutionReviewRecord,
)
from analyst_forecast.infrastructure.db.session import create_session_factory
from analyst_forecast.schemas.pipeline import (
    PIPELINE_MODELS,
    CandidateReview,
    P05Output,
    P06Output,
    P07Output,
    P08Output,
    P09Output,
    P11Output,
    P12Output,
    P13Output,
    PipelineOutput,
    TargetResolutionCandidate,
    pipeline_schema_path,
)


def ingest_pipeline_output(
    settings: AppSettings,
    *,
    input_path: Path,
    raw_bytes: bytes,
    untyped_payload: dict[str, Any],
    output_hash: str,
) -> AiIngestResult:
    prompt_id = _prompt_id(untyped_payload)
    run_id = str(untyped_payload.get("run_id", ""))
    issues: list[ValidationIssue] = []
    model_class = PIPELINE_MODELS.get(prompt_id)
    if model_class is None:
        issues.append(
            ValidationIssue(
                "unsupported_prompt",
                f"Schema 2.0.0で未対応のprompt_idです: {prompt_id or '未指定'}",
                "$['prompt_execution']['prompt_id']",
            )
        )
        return _reject(
            settings,
            input_path=input_path,
            raw_bytes=raw_bytes,
            output_hash=output_hash,
            run_id=run_id,
            issues=issues,
        )

    fixed_schema = json.loads(pipeline_schema_path(prompt_id).read_text(encoding="utf-8"))
    for schema_error in sorted(
        Draft202012Validator(fixed_schema).iter_errors(untyped_payload),
        key=lambda item: list(item.absolute_path),
    ):
        issue_path = "$" + "".join(f"[{part!r}]" for part in schema_error.absolute_path)
        issues.append(
            ValidationIssue(
                "json_schema",
                f"{prompt_id}の固定Schemaに適合しません。詳細: {schema_error.message}",
                issue_path,
            )
        )

    payload: PipelineOutput | None = None
    if not issues:
        try:
            payload = model_class.model_validate(untyped_payload)  # type: ignore[assignment]
        except ValidationError as error:
            issues.extend(_pydantic_issues(error))
    if payload is not None:
        issues.extend(_validate_references(settings, payload))
    if issues:
        return _reject(
            settings,
            input_path=input_path,
            raw_bytes=raw_bytes,
            output_hash=output_hash,
            run_id=run_id,
            issues=issues,
        )
    assert payload is not None

    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        existing = session.scalar(
            select(AiArtifactRecord).where(AiArtifactRecord.output_hash == output_hash)
        )
        if existing is not None:
            component_ids = tuple(
                session.scalars(
                    select(ForecastComponentRecord.forecast_component_id)
                    .join(
                        ForecastIssuanceRecord,
                        ForecastIssuanceRecord.forecast_issuance_id
                        == ForecastComponentRecord.forecast_issuance_id,
                    )
                    .where(ForecastIssuanceRecord.ai_artifact_id == existing.ai_artifact_id)
                )
            )
            return AiIngestResult(
                status=AiIngestStatus.ALREADY_IMPORTED,
                output_hash=output_hash,
                artifact_ids=(existing.ai_artifact_id,),
                component_ids=component_ids,
                guidance="同じAI成果物は既に記録済みです。",
            )

    review_issues = _review_issues(settings, payload)
    classification = AiIngestStatus.NEEDS_REVIEW if review_issues else AiIngestStatus.ACCEPTED
    classified_path = _store_classified(
        settings,
        run_id=payload.run_id,
        input_path=input_path,
        raw_bytes=raw_bytes,
        output_hash=output_hash,
        classification=classification,
        issues=review_issues,
    )
    backup_database(
        settings.database_file,
        backup_dir=settings.vault_root / "_system" / "backups" / "database",
        reason=f"{prompt_id.lower()}_import",
    )
    try:
        artifact_id, created_issuances, created_components = _insert_pipeline_payload(
            settings,
            payload=payload,
            output_hash=output_hash,
            classified_path=classified_path,
            classification=classification,
        )
    except Exception as error:
        if classified_path is not None:
            classified_path.unlink(missing_ok=True)
        transaction_issue = ValidationIssue(
            "database_transaction",
            "AI成果物のDB transactionをロールバックしました。",
        )
        _store_classified(
            settings,
            run_id=payload.run_id,
            input_path=input_path,
            raw_bytes=raw_bytes,
            output_hash=output_hash,
            classification=AiIngestStatus.REJECTED,
            issues=[transaction_issue],
        )
        raise RuntimeError(
            "AI成果物のDB取込みに失敗しました。次の操作: rejectedの監査記録を確認してください。"
        ) from error

    from analyst_forecast.application.workflow import refresh_workflow

    refresh_workflow(settings, payload.run_id)
    return AiIngestResult(
        status=classification,
        output_hash=output_hash,
        issues=tuple(review_issues),
        artifact_ids=(artifact_id,),
        forecast_issuance_ids=tuple(created_issuances),
        component_ids=tuple(created_components),
        guidance=(
            "AI成果物を記録しました。別AIレビューを実行してください。"
            if classification is AiIngestStatus.NEEDS_REVIEW
            else f"{prompt_id}成果物を検証・記録しました。"
        ),
    )


def _prompt_id(payload: dict[str, Any]) -> str:
    execution = payload.get("prompt_execution")
    return str(execution.get("prompt_id", "")) if isinstance(execution, dict) else ""


def _pydantic_issues(error: ValidationError) -> list[ValidationIssue]:
    return [
        ValidationIssue(
            "pydantic",
            f"AI成果物の値が不正です。詳細: {detail['msg']}",
            "$" + "".join(f"[{part!r}]" for part in detail["loc"]),
        )
        for detail in error.errors(include_url=False)
    ]


def _validate_references(
    settings: AppSettings,
    payload: PipelineOutput,
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
                    "SOURCE IDがこの案件に関連付けられていません。",
                    "$['source_id']",
                )
            )
            return issues

        if isinstance(payload, P05Output):
            issues.extend(_validate_p05(settings, session, source, payload))
        elif isinstance(payload, P07Output):
            issues.extend(_validate_p07(settings, session, source, payload))
        elif isinstance(payload, P06Output):
            issues.extend(
                _validate_review_artifact(session, payload, allowed_prompts=("P05", "P07"))
            )
        elif isinstance(payload, P08Output):
            issues.extend(_validate_p08(settings, session, source, payload))
        elif isinstance(payload, P09Output):
            issues.extend(_validate_review_artifact(session, payload, allowed_prompts=("P08",)))
        elif isinstance(payload, P11Output):
            issues.extend(_validate_p11(session, run, payload))
        elif isinstance(payload, P12Output):
            issues.extend(_validate_p12(session, run, payload))
        elif isinstance(payload, P13Output):
            issues.extend(_validate_p13(session, run, payload))
    return issues


def _validate_text_preprocess(
    settings: AppSettings,
    session: Session,
    source: SourceRecord,
    *,
    run_id: str,
    source_id: str,
    input_hash: str,
    segments: list[Any],
    prompt_label: str,
) -> list[ValidationIssue]:
    from analyst_forecast.application.raw_sources import resolve_source_raw_path

    raw_path = resolve_source_raw_path(settings, session, run_id=run_id, source=source)
    if raw_path is None:
        return [ValidationIssue("raw_missing", "登録済みraw原文がありません。")]
    raw_bytes = raw_path.read_bytes()
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    if raw_hash != source.raw_hash or input_hash != source.raw_hash:
        return [
            ValidationIssue(
                "input_hash_mismatch",
                f"{prompt_label} input_hashまたはraw SHA-256が登録値と一致しません。",
                "$['input_hash']",
            )
        ]
    raw_text = raw_bytes.decode("utf-8-sig")
    issues: list[ValidationIssue] = []
    for index, segment in enumerate(segments):
        if raw_text[segment.raw_start_offset : segment.raw_end_offset] != segment.raw_text:
            issues.append(
                ValidationIssue(
                    "segment_offset_mismatch",
                    f"{prompt_label} segmentのraw offsetとraw_textが一致しません。",
                    f"$['segments'][{index}]",
                )
            )
    return issues


def _validate_p05(
    settings: AppSettings,
    session: Session,
    source: SourceRecord,
    payload: P05Output,
) -> list[ValidationIssue]:
    if source.medium != "youtube":
        return [
            ValidationIssue(
                "medium_mismatch",
                "P05はYouTube媒体専用です。blog/X/webはP07を使ってください。",
            )
        ]
    issues = _validate_text_preprocess(
        settings,
        session,
        source,
        run_id=payload.run_id,
        source_id=payload.source_id,
        input_hash=payload.input_hash,
        segments=list(payload.segments),
        prompt_label="P05",
    )
    return issues


def _validate_p07(
    settings: AppSettings,
    session: Session,
    source: SourceRecord,
    payload: P07Output,
) -> list[ValidationIssue]:
    if source.medium == "youtube":
        return [
            ValidationIssue(
                "medium_mismatch",
                "P07は非YouTube媒体専用です。YouTubeはP05を使ってください。",
            )
        ]
    return _validate_text_preprocess(
        settings,
        session,
        source,
        run_id=payload.run_id,
        source_id=payload.source_id,
        input_hash=payload.input_hash,
        segments=list(payload.segments),
        prompt_label="P07",
    )


def _validate_review_artifact(
    session: Session,
    payload: P06Output | P09Output,
    *,
    allowed_prompts: tuple[str, ...],
) -> list[ValidationIssue]:
    reviewed = session.get(AiArtifactRecord, payload.reviewed_artifact_id)
    if (
        reviewed is None
        or reviewed.prompt_id not in allowed_prompts
        or reviewed.run_id != payload.run_id
        or reviewed.source_id != payload.source_id
    ):
        return [
            ValidationIssue(
                "invalid_reviewed_artifact",
                "レビュー対象artifactが存在しないか、案件・SOURCE・promptが一致しません。",
                "$['reviewed_artifact_id']",
            )
        ]
    if payload.input_hash != reviewed.output_hash:
        return [
            ValidationIssue(
                "input_hash_mismatch",
                "レビューinput_hashが対象artifactのoutput hashと一致しません。",
                "$['input_hash']",
            )
        ]
    if reviewed.classification not in {"needs_review", "accepted"}:
        return [
            ValidationIssue(
                "reviewed_not_reviewable",
                "レビュー対象のclassificationが不正です。",
                "$['reviewed_artifact_id']",
            )
        ]
    if payload.decision == "correct" and payload.corrected_payload is not None:
        model_class = PIPELINE_MODELS.get(reviewed.prompt_id)
        if model_class is None:
            return [
                ValidationIssue("unsupported_corrected_prompt", "修正payloadのpromptが未対応です。")
            ]
        try:
            model_class.model_validate(payload.corrected_payload)
        except ValidationError as error:
            return _pydantic_issues(error)
    return []


def _validate_p08(
    settings: AppSettings,
    session: Session,
    source: SourceRecord,
    payload: P08Output,
) -> list[ValidationIssue]:
    upstream_id = payload.upstream_artifact_id or payload.p05_artifact_id
    upstream_prompt = payload.upstream_prompt_id or ("P05" if payload.p05_artifact_id else None)
    if upstream_id is None or upstream_prompt is None:
        return [
            ValidationIssue(
                "missing_upstream",
                "P08にはupstream_artifact_id（または互換のp05_artifact_id）が必要です。",
            )
        ]
    upstream = session.get(AiArtifactRecord, upstream_id)
    link = session.get(
        RunSourceRecord,
        {"run_id": payload.run_id, "source_id": payload.source_id},
    )
    run_ok = False
    if upstream is not None and (
        upstream.run_id == payload.run_id
        or (
            link is not None
            and link.latest_ai_artifact_id == upstream.ai_artifact_id
            and upstream.source_id == payload.source_id
        )
    ):
        run_ok = True
    if (
        upstream is None
        or upstream.prompt_id != upstream_prompt
        or upstream.prompt_id not in {"P05", "P07"}
        or not run_ok
        or upstream.source_id != payload.source_id
    ):
        return [
            ValidationIssue(
                "invalid_upstream_reference",
                "P08が参照する上流成果物が存在しないか、案件・SOURCE・promptが一致しません。",
                "$['upstream_artifact_id']",
            )
        ]
    if source.medium == "youtube" and upstream.prompt_id != "P05":
        return [
            ValidationIssue(
                "medium_upstream_mismatch",
                "YouTube媒体のP08は受理済みP05だけを上流にできます。",
            )
        ]
    if source.medium != "youtube" and upstream.prompt_id != "P07":
        return [
            ValidationIssue(
                "medium_upstream_mismatch",
                "非YouTube媒体のP08は受理済みP07だけを上流にできます。",
            )
        ]
    if upstream.classification != "accepted":
        return [
            ValidationIssue(
                "upstream_not_accepted",
                "上流成果物のAIレビューが未解決です。",
                "$['upstream_artifact_id']",
            )
        ]
    if payload.input_hash != upstream.output_hash:
        return [
            ValidationIssue(
                "input_hash_mismatch",
                "P08 input_hashが上流artifactのoutput hashと一致しません。",
                "$['input_hash']",
            )
        ]
    from analyst_forecast.application.raw_sources import resolve_source_raw_path

    raw_path = resolve_source_raw_path(settings, session, run_id=payload.run_id, source=source)
    if raw_path is None:
        return [ValidationIssue("raw_missing", "登録済みraw原文がありません。")]
    raw_bytes = raw_path.read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != source.raw_hash:
        return [ValidationIssue("raw_hash_mismatch", "raw原文が変更されています。")]
    raw_text = raw_bytes.decode("utf-8-sig")
    segments = list(
        session.scalars(
            select(SegmentRecord).where(SegmentRecord.ai_artifact_id == upstream.ai_artifact_id)
        )
    )
    segment_by_ref = {item.local_ref: item for item in segments}
    issues: list[ValidationIssue] = []
    for forecast_index, forecast in enumerate(payload.forecasts):
        if forecast.made_at > forecast.publicly_available_at:
            issues.append(
                ValidationIssue(
                    "made_at_after_public",
                    "made_atはpublicly_available_at以前にしてください。",
                    f"$['forecasts'][{forecast_index}]",
                )
            )
        if forecast.upstream_segment_refs:
            for ref in forecast.upstream_segment_refs:
                segment = segment_by_ref.get(ref)
                if segment is None:
                    issues.append(
                        ValidationIssue(
                            "unknown_segment_ref",
                            f"上流segment参照が存在しません: {ref}",
                            f"$['forecasts'][{forecast_index}]['upstream_segment_refs']",
                        )
                    )
                    continue
                for evidence_index, evidence in enumerate(forecast.evidence):
                    issue_path = f"$['forecasts'][{forecast_index}]['evidence'][{evidence_index}]"
                    if not (
                        segment.raw_start_offset
                        <= evidence.start_offset
                        < evidence.end_offset
                        <= segment.raw_end_offset
                    ):
                        issues.append(
                            ValidationIssue(
                                "evidence_outside_segment",
                                "引用offsetが申告segmentの範囲外です。",
                                issue_path,
                            )
                        )
        for evidence_index, evidence in enumerate(forecast.evidence):
            issue_path = f"$['forecasts'][{forecast_index}]['evidence'][{evidence_index}]"
            if evidence.source_id != payload.source_id:
                issues.append(
                    ValidationIssue(
                        "source_reference_mismatch",
                        "引用のSOURCE IDがP08のSOURCE IDと一致しません。",
                        issue_path,
                    )
                )
            elif raw_text[evidence.start_offset : evidence.end_offset] != evidence.quote:
                issues.append(
                    ValidationIssue(
                        "quote_offset_mismatch",
                        "引用とraw offsetが一致しません。",
                        issue_path,
                    )
                )
    return issues


def _validate_p11(
    session: Session,
    run: RunRecord,
    payload: P11Output,
) -> list[ValidationIssue]:
    context = _component_context(session, payload.forecast_component_id)
    if context is None:
        return [
            ValidationIssue(
                "unknown_component",
                "P11の構成予想IDが存在しません。",
                "$['forecast_component_id']",
            )
        ]
    component, issuance, artifact = context
    if (
        artifact is None
        or artifact.prompt_id != "P08"
        or artifact.run_id != payload.run_id
        or issuance.source_id != payload.source_id
        or issuance.analyst_id != run.analyst_id
    ):
        return [
            ValidationIssue(
                "component_context_mismatch",
                "P11の案件、SOURCE、P08構成予想の参照が一致しません。",
            )
        ]
    issues: list[ValidationIssue] = []
    if payload.input_hash != artifact.output_hash:
        issues.append(
            ValidationIssue(
                "input_hash_mismatch",
                "P11 input_hashがP08 output hashと一致しません。",
                "$['input_hash']",
            )
        )
    made_at = _as_utc(issuance.made_at)
    if _as_utc(payload.knowledge_cutoff) > made_at:
        issues.append(
            ValidationIssue(
                "future_knowledge_cutoff",
                "knowledge_cutoffが発言日時より後です。",
                "$['knowledge_cutoff']",
            )
        )
    for index, candidate in enumerate(payload.candidates):
        if _as_utc(candidate.knowledge_cutoff) > made_at:
            issues.append(
                ValidationIssue(
                    "future_candidate_cutoff",
                    "candidateのknowledge_cutoffが発言日時より後です。",
                    f"$['candidates'][{index}]['knowledge_cutoff']",
                )
            )
        if candidate.existed_at > issuance.made_at.date():
            issues.append(
                ValidationIssue(
                    "candidate_not_existing_at_statement",
                    "candidateが発言日時点に存在しません。",
                    f"$['candidates'][{index}]['existed_at']",
                )
            )
    if component.target_resolution_status == "locked":
        issues.append(
            ValidationIssue(
                "mapping_already_locked",
                "固定済みmappingをP11で上書きできません。",
            )
        )
    return issues


def _validate_p12(
    session: Session,
    run: RunRecord,
    payload: P12Output,
) -> list[ValidationIssue]:
    proposal = session.get(AiArtifactRecord, payload.proposal_artifact_id)
    if proposal is None or proposal.prompt_id != "P11":
        return [
            ValidationIssue(
                "invalid_p11_reference",
                "P12は存在するP11成果物を参照する必要があります。",
                "$['proposal_artifact_id']",
            )
        ]
    if (
        proposal.run_id != payload.run_id
        or proposal.source_id != payload.source_id
        or payload.input_hash != proposal.output_hash
    ):
        return [
            ValidationIssue(
                "p11_context_mismatch",
                "P12の案件、SOURCE、input hashがP11成果物と一致しません。",
            )
        ]
    context = _component_context(session, payload.forecast_component_id)
    if context is None:
        return [ValidationIssue("unknown_component", "P12の構成予想IDが存在しません。")]
    _, issuance, _ = context
    issues: list[ValidationIssue] = []
    proposal_component_ids = set(
        session.scalars(
            select(TargetResolutionCandidateRecord.forecast_component_id).where(
                TargetResolutionCandidateRecord.proposal_artifact_id == proposal.ai_artifact_id
            )
        )
    )
    if proposal.payload.get("forecast_component_id") not in {
        None,
        payload.forecast_component_id,
    }:
        issues.append(
            ValidationIssue(
                "component_mismatch",
                "P12が参照するP11は同じforecast_component_idでなければなりません。",
            )
        )
    if proposal_component_ids and payload.forecast_component_id not in proposal_component_ids:
        issues.append(
            ValidationIssue(
                "component_mismatch",
                "P12が別componentのP11を参照しています。",
            )
        )
    if issuance.analyst_id != run.analyst_id or issuance.source_id != payload.source_id:
        issues.append(
            ValidationIssue(
                "component_context_mismatch",
                "P12の構成予想が案件・SOURCEと一致しません。",
            )
        )
    if _as_utc(payload.knowledge_cutoff) > _as_utc(issuance.made_at):
        issues.append(
            ValidationIssue(
                "future_knowledge_cutoff",
                "knowledge_cutoffが発言日時より後です。",
            )
        )
    candidate_refs = set(
        session.scalars(
            select(TargetResolutionCandidateRecord.candidate_ref).where(
                TargetResolutionCandidateRecord.proposal_artifact_id == proposal.ai_artifact_id
            )
        )
    )
    for review in payload.reviews:
        if review.candidate_ref not in candidate_refs:
            issues.append(
                ValidationIssue(
                    "unknown_candidate_reference",
                    f"P12が未知のcandidateを参照しています: {review.candidate_ref}",
                )
            )
    return issues


def _validate_p13(
    session: Session,
    run: RunRecord,
    payload: P13Output,
) -> list[ValidationIssue]:
    proposal = session.get(AiArtifactRecord, payload.proposal_artifact_id)
    review = session.get(AiArtifactRecord, payload.review_artifact_id)
    if proposal is None or proposal.prompt_id != "P11":
        return [ValidationIssue("invalid_p11_reference", "P13のP11参照が不正です。")]
    if review is None or review.prompt_id != "P12":
        return [ValidationIssue("invalid_p12_reference", "P13のP12参照が不正です。")]
    if (
        proposal.run_id != payload.run_id
        or review.run_id != payload.run_id
        or proposal.source_id != payload.source_id
        or review.source_id != payload.source_id
        or payload.input_hash != review.output_hash
    ):
        return [ValidationIssue("adjudication_context_mismatch", "P13の参照関係が不正です。")]
    context = _component_context(session, payload.forecast_component_id)
    if context is None:
        return [ValidationIssue("unknown_component", "P13の構成予想IDが存在しません。")]
    _, issuance, _ = context
    review_payload = review.payload or {}
    if review_payload.get("proposal_artifact_id") not in {None, payload.proposal_artifact_id}:
        return [
            ValidationIssue(
                "p11_p12_pair_mismatch",
                "P13が参照するP12のproposal IDがP13のproposal IDと一致しません。",
            )
        ]
    if review_payload.get("forecast_component_id") not in {None, payload.forecast_component_id}:
        return [
            ValidationIssue(
                "component_mismatch",
                "P13のP12参照が別componentです。",
            )
        ]
    if proposal.payload.get("forecast_component_id") not in {None, payload.forecast_component_id}:
        return [
            ValidationIssue(
                "component_mismatch",
                "P13のP11参照が別componentです。",
            )
        ]
    if issuance.analyst_id != run.analyst_id:
        return [ValidationIssue("component_context_mismatch", "P13の案件参照が不正です。")]
    if _as_utc(payload.knowledge_cutoff) > _as_utc(issuance.made_at):
        return [
            ValidationIssue(
                "future_knowledge_cutoff",
                "knowledge_cutoffが発言日時より後です。",
            )
        ]
    if payload.selected_candidate_ref is not None:
        candidate = session.scalar(
            select(TargetResolutionCandidateRecord).where(
                TargetResolutionCandidateRecord.proposal_artifact_id
                == payload.proposal_artifact_id,
                TargetResolutionCandidateRecord.candidate_ref == payload.selected_candidate_ref,
            )
        )
        if candidate is None:
            return [
                ValidationIssue(
                    "unknown_candidate_reference",
                    "P13が未知のcandidateを参照しています。",
                )
            ]
    return []


def _review_issues(
    settings: AppSettings,
    payload: PipelineOutput,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if isinstance(payload, P05Output):
        for index, segment in enumerate(payload.segments):
            if (
                segment.speaker_status == "unknown"
                or segment.speaker_confidence < settings.confidence_review_threshold
                or segment.review_status == "needs_review"
            ):
                issues.append(
                    ValidationIssue(
                        "speaker_review_required",
                        "話者がunknownまたは低確信度のため別AIレビューが必要です。",
                        f"$['segments'][{index}]",
                    )
                )
            if segment.importance == "high":
                issues.append(
                    ValidationIssue(
                        "high_importance",
                        f"高重要度の理由: {segment.high_importance_reason}",
                        f"$['segments'][{index}]",
                    )
                )
    elif isinstance(payload, P07Output):
        for index, text_segment in enumerate(payload.segments):
            if (
                text_segment.author_status == "unknown"
                or text_segment.author_confidence < settings.confidence_review_threshold
                or text_segment.review_status == "needs_review"
            ):
                issues.append(
                    ValidationIssue(
                        "author_review_required",
                        "著者がunknownまたは低確信度のため別AIレビューが必要です。",
                        f"$['segments'][{index}]",
                    )
                )
            if text_segment.importance == "high":
                issues.append(
                    ValidationIssue(
                        "high_importance",
                        f"高重要度の理由: {text_segment.high_importance_reason}",
                        f"$['segments'][{index}]",
                    )
                )
    elif isinstance(payload, P08Output):
        for index, forecast in enumerate(payload.forecasts):
            if forecast.extraction_confidence < settings.confidence_review_threshold:
                issues.append(
                    ValidationIssue(
                        "low_confidence",
                        "予想抽出が低確信度のため別AIレビューが必要です。",
                        f"$['forecasts'][{index}]",
                    )
                )
            if forecast.importance == "high":
                issues.append(
                    ValidationIssue(
                        "high_importance",
                        f"高重要度の理由: {forecast.high_importance_reason}",
                        f"$['forecasts'][{index}]",
                    )
                )
            if forecast.speaker_attribution_status == "uncertain":
                issues.append(
                    ValidationIssue(
                        "attribution_uncertain",
                        "話者帰属がuncertainのため別AIレビューが必要です。",
                        f"$['forecasts'][{index}]",
                    )
                )
    return issues


def _insert_pipeline_payload(
    settings: AppSettings,
    *,
    payload: PipelineOutput,
    output_hash: str,
    classified_path: Path | None,
    classification: AiIngestStatus,
) -> tuple[str, list[str], list[str]]:
    session_factory = create_session_factory(settings.database_file)
    issuance_ids: list[str] = []
    component_ids: list[str] = []
    with session_factory.begin() as session:
        execution_id = next_id(session, "PEX-", width=6, sequence_key="PROMPT_EXECUTION")
        relative_path = (
            classified_path.relative_to(settings.vault_root).as_posix()
            if classified_path is not None
            else ""
        )
        session.add(
            PromptExecutionRecord(
                prompt_execution_id=execution_id,
                ai_import_id=None,
                run_id=payload.run_id,
                prompt_id=payload.prompt_execution.prompt_id,
                prompt_version=payload.prompt_execution.prompt_version,
                environment=payload.prompt_execution.environment,
                model=payload.prompt_execution.model,
                input_files=[payload.source_id, payload.input_hash],
                output_file=relative_path,
                executed_at=payload.prompt_execution.executed_at,
                validation_status=classification.value,
            )
        )
        session.flush()
        artifact_id = next_id(session, "AIF-", width=6, sequence_key="AI_ARTIFACT")
        artifact = AiArtifactRecord(
            ai_artifact_id=artifact_id,
            run_id=payload.run_id,
            source_id=payload.source_id,
            prompt_execution_id=execution_id,
            prompt_id=payload.prompt_execution.prompt_id,
            schema_version=payload.schema_version,
            input_hash=payload.input_hash,
            output_hash=output_hash,
            classified_file_path=relative_path,
            classification=classification.value,
            resolution_status=_initial_resolution_status(payload, classification),
            confidence=_payload_confidence(payload),
            importance=_payload_importance(payload),
            high_importance_reason=_payload_importance_reason(payload),
            knowledge_cutoff=getattr(payload, "knowledge_cutoff", None),
            payload=payload.model_dump(mode="json"),
        )
        session.add(artifact)
        session.flush()

        if isinstance(payload, P05Output):
            _insert_p05(session, settings, payload, artifact)
        elif isinstance(payload, P07Output):
            _insert_p07(session, settings, payload, artifact)
        elif isinstance(payload, P06Output | P09Output):
            issuance_ids, component_ids = _apply_review_decision(
                session,
                settings,
                payload=payload,
                review_artifact=artifact,
                classification=classification,
            )
        elif isinstance(payload, P08Output):
            if classification is AiIngestStatus.ACCEPTED:
                issuance_ids, component_ids = _insert_p08(session, payload, artifact)
            _update_run_source_after_p08(
                session,
                payload,
                artifact,
                classification,
            )
        elif isinstance(payload, P11Output):
            _insert_p11(session, payload, artifact)
        elif isinstance(payload, P12Output):
            _insert_p12(session, payload, artifact)
        elif isinstance(payload, P13Output):
            _insert_p13(session, payload, artifact)
    return artifact_id, issuance_ids, component_ids


def _insert_p05(
    session: Session,
    settings: AppSettings,
    payload: P05Output,
    artifact: AiArtifactRecord,
) -> None:
    for segment in payload.segments:
        session.add(
            SegmentRecord(
                segment_id=next_id(session, "SEG-", width=6, sequence_key="SEGMENT"),
                ai_artifact_id=artifact.ai_artifact_id,
                source_id=payload.source_id,
                local_ref=segment.segment_ref,
                sequence_number=segment.sequence_number,
                raw_start_offset=segment.raw_start_offset,
                raw_end_offset=segment.raw_end_offset,
                raw_text=segment.raw_text,
                normalized_text=segment.normalized_text,
                speaker_status=segment.speaker_status,
                speaker_candidate=segment.speaker_candidate,
                speaker_confidence=segment.speaker_confidence,
                attribution_basis=segment.attribution_basis,
                review_status=segment.review_status,
                importance=segment.importance,
                high_importance_reason=segment.high_importance_reason,
            )
        )
    _update_preprocess_link(session, settings, payload, artifact, prompt_label="P05")


def _insert_p07(
    session: Session,
    settings: AppSettings,
    payload: P07Output,
    artifact: AiArtifactRecord,
) -> None:
    for segment in payload.segments:
        session.add(
            SegmentRecord(
                segment_id=next_id(session, "SEG-", width=6, sequence_key="SEGMENT"),
                ai_artifact_id=artifact.ai_artifact_id,
                source_id=payload.source_id,
                local_ref=segment.segment_ref,
                sequence_number=segment.sequence_number,
                raw_start_offset=segment.raw_start_offset,
                raw_end_offset=segment.raw_end_offset,
                raw_text=segment.raw_text,
                normalized_text=segment.normalized_text,
                speaker_status=segment.author_status,
                speaker_candidate=segment.author_candidate,
                speaker_confidence=segment.author_confidence,
                attribution_basis=f"{segment.statement_kind}: {segment.attribution_basis}",
                review_status=segment.review_status,
                importance=segment.importance,
                high_importance_reason=segment.high_importance_reason,
            )
        )
    _update_preprocess_link(session, settings, payload, artifact, prompt_label="P07")


def _update_preprocess_link(
    session: Session,
    settings: AppSettings,
    payload: P05Output | P07Output,
    artifact: AiArtifactRecord,
    *,
    prompt_label: str,
) -> None:
    link = session.get(
        RunSourceRecord,
        {"run_id": payload.run_id, "source_id": payload.source_id},
    )
    if link is not None:
        link.processing_status = artifact.classification
        link.latest_ai_artifact_id = artifact.ai_artifact_id
    source = session.get(SourceRecord, payload.source_id)
    run = session.get(RunRecord, payload.run_id)
    if source is not None and run is not None:
        run_path = settings.vault_root / Path(run.run_path)
        processed = (
            run_path
            / "02_sources"
            / source.medium
            / "processed"
            / f"{prompt_label}-{artifact.ai_artifact_id}.json"
        )
        _atomic_json(processed, artifact.payload)


def _insert_p08(
    session: Session,
    payload: P08Output,
    artifact: AiArtifactRecord,
) -> tuple[list[str], list[str]]:
    run = session.get(RunRecord, payload.run_id)
    if run is None:
        raise ValueError("P08取込み中に案件が消失しました")
    formal_forecasts = [forecast for forecast in payload.forecasts if _is_formal_forecast(forecast)]
    issuance_ids: list[str] = []
    component_ids: list[str] = []
    groups: dict[str, ForecastGroupRecord] = {}
    for group_ref in dict.fromkeys(forecast.forecast_group_ref for forecast in formal_forecasts):
        members = [
            forecast for forecast in formal_forecasts if forecast.forecast_group_ref == group_ref
        ]
        first = min(members, key=lambda item: item.made_at)
        latest = max(members, key=lambda item: item.made_at)
        existing_id = first.existing_forecast_group_id
        if existing_id is not None:
            group = session.get(ForecastGroupRecord, existing_id)
            if group is None or group.analyst_id != run.analyst_id:
                raise ValueError("既存予想グループ参照が不正です")
            group.latest_issued_at = latest.made_at
            group.current_stance = latest.components[0].direction.value
        else:
            group = ForecastGroupRecord(
                forecast_group_id=next_id(session, "FCG-", width=6, sequence_key="FORECAST_GROUP"),
                analyst_id=run.analyst_id,
                central_thesis=first.human_readable_summary,
                first_issued_at=first.made_at,
                latest_issued_at=latest.made_at,
                current_stance=latest.components[0].direction.value,
                reaffirmation_count=sum(
                    item.relation_to_previous == "reaffirmation" for item in members
                ),
                revision_count=sum(
                    item.relation_to_previous not in {"initial", "reaffirmation", "withdrawal"}
                    for item in members
                ),
                withdrawal_status=(
                    "withdrawn" if latest.relation_to_previous == "withdrawal" else "active"
                ),
            )
            session.add(group)
        groups[group_ref] = group
    session.flush()

    for forecast in formal_forecasts:
        issuance_id = next_id(session, "FCI-", width=6, sequence_key="FORECAST_ISSUANCE")
        issuance_ids.append(issuance_id)
        issuance = ForecastIssuanceRecord(
            forecast_issuance_id=issuance_id,
            analyst_id=run.analyst_id,
            forecast_group_id=groups[forecast.forecast_group_ref].forecast_group_id,
            ai_import_id=None,
            ai_artifact_id=artifact.ai_artifact_id,
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
            current_status=_forecast_initial_status(
                run.evaluation_as_of,
                forecast.components[0].normalized_start,
            ),
        )
        session.add(issuance)
        session.flush()
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
        local_ids = {
            component.component_ref: next_id(
                session, "FCC-", width=6, sequence_key="FORECAST_COMPONENT"
            )
            for component in forecast.components
        }
        for component in forecast.components:
            component_id = local_ids[component.component_ref]
            component_ids.append(component_id)
            session.add(
                ForecastComponentRecord(
                    forecast_component_id=component_id,
                    forecast_issuance_id=issuance_id,
                    parent_component_id=(
                        local_ids.get(component.parent_component_ref)
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
                    raw_target_label=component.raw_target_label,
                    target_resolution_status="pending",
                    importance=forecast.importance,
                    high_importance_reason=forecast.high_importance_reason,
                    target_id=None,
                    target_mapping_id=None,
                )
            )
    return issuance_ids, component_ids


def _is_formal_forecast(forecast: Any) -> bool:
    if forecast.statement_kind == "third_party_summary":
        return False
    return forecast.speaker_attribution_status not in {"not_target", "uncertain"}


def _apply_review_decision(
    session: Session,
    settings: AppSettings,
    *,
    payload: P06Output | P09Output,
    review_artifact: AiArtifactRecord,
    classification: AiIngestStatus,
) -> tuple[list[str], list[str]]:
    del classification  # review自身は通常accepted分類
    reviewed = session.get(AiArtifactRecord, payload.reviewed_artifact_id)
    if reviewed is None:
        raise ValueError("レビュー対象artifactが消失しました")
    # 冪等: 既に同じreviewで解決済みなら二重materializeしない
    if reviewed.resolved_by_artifact_id == review_artifact.ai_artifact_id:
        existing = list(
            session.scalars(
                select(ForecastIssuanceRecord).where(
                    ForecastIssuanceRecord.ai_artifact_id == reviewed.ai_artifact_id
                )
            )
        )
        if not existing and reviewed.supersedes_artifact_id:
            existing = list(
                session.scalars(
                    select(ForecastIssuanceRecord).where(
                        ForecastIssuanceRecord.ai_artifact_id == reviewed.supersedes_artifact_id
                    )
                )
            )
        resolved_component_ids = [
            component.forecast_component_id
            for issuance in existing
            for component in session.scalars(
                select(ForecastComponentRecord).where(
                    ForecastComponentRecord.forecast_issuance_id == issuance.forecast_issuance_id
                )
            )
        ]
        return [item.forecast_issuance_id for item in existing], resolved_component_ids

    issuance_ids: list[str] = []
    component_ids: list[str] = []
    if payload.decision == "accept":
        reviewed.resolution_status = "resolved"
        reviewed.resolved_by_artifact_id = review_artifact.ai_artifact_id
        reviewed.classification = "accepted"
        review_artifact.resolution_status = "resolved"
        if reviewed.prompt_id == "P08":
            p08 = P08Output.model_validate(reviewed.payload)
            issuance_ids, component_ids = _insert_p08(session, p08, reviewed)
            _update_run_source_after_p08(session, p08, reviewed, AiIngestStatus.ACCEPTED)
        else:
            link = session.get(
                RunSourceRecord,
                {"run_id": payload.run_id, "source_id": payload.source_id},
            )
            if link is not None:
                link.processing_status = "accepted"
                link.latest_ai_artifact_id = reviewed.ai_artifact_id
    elif payload.decision == "correct":
        assert payload.corrected_payload is not None
        corrected_hash = hashlib.sha256(
            json.dumps(payload.corrected_payload, ensure_ascii=False, sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()
        existing_corrected = session.scalar(
            select(AiArtifactRecord).where(AiArtifactRecord.output_hash == corrected_hash)
        )
        if existing_corrected is not None:
            corrected = existing_corrected
        else:
            corrected_id = next_id(session, "AIF-", width=6, sequence_key="AI_ARTIFACT")
            execution_id = next_id(session, "PEX-", width=6, sequence_key="PROMPT_EXECUTION")
            session.add(
                PromptExecutionRecord(
                    prompt_execution_id=execution_id,
                    ai_import_id=None,
                    run_id=payload.run_id,
                    prompt_id=reviewed.prompt_id,
                    prompt_version=payload.prompt_execution.prompt_version,
                    environment=payload.prompt_execution.environment,
                    model=payload.prompt_execution.model,
                    input_files=[payload.source_id, corrected_hash],
                    output_file=review_artifact.classified_file_path,
                    executed_at=payload.prompt_execution.executed_at,
                    validation_status="accepted",
                )
            )
            session.flush()
            corrected = AiArtifactRecord(
                ai_artifact_id=corrected_id,
                run_id=payload.run_id,
                source_id=payload.source_id,
                prompt_execution_id=execution_id,
                prompt_id=reviewed.prompt_id,
                schema_version=str(payload.corrected_payload.get("schema_version", "2.0.0")),
                input_hash=str(payload.corrected_payload.get("input_hash", reviewed.input_hash)),
                output_hash=corrected_hash,
                classified_file_path=review_artifact.classified_file_path,
                classification="accepted",
                resolution_status="accepted",
                confidence=None,
                importance="normal",
                knowledge_cutoff=payload.knowledge_cutoff,
                supersedes_artifact_id=reviewed.ai_artifact_id,
                resolved_by_artifact_id=review_artifact.ai_artifact_id,
                payload=dict(payload.corrected_payload),
            )
            session.add(corrected)
            session.flush()
            model_class = PIPELINE_MODELS[reviewed.prompt_id]
            corrected_model = model_class.model_validate(payload.corrected_payload)
            if isinstance(corrected_model, P05Output):
                _insert_p05(session, settings, corrected_model, corrected)
            elif isinstance(corrected_model, P07Output):
                _insert_p07(session, settings, corrected_model, corrected)
            elif isinstance(corrected_model, P08Output):
                issuance_ids, component_ids = _insert_p08(session, corrected_model, corrected)
                _update_run_source_after_p08(
                    session, corrected_model, corrected, AiIngestStatus.ACCEPTED
                )
        reviewed.classification = "superseded"
        reviewed.resolution_status = "superseded"
        reviewed.resolved_by_artifact_id = review_artifact.ai_artifact_id
        reviewed.supersedes_artifact_id = None
        review_artifact.resolution_status = "resolved"
        link = session.get(
            RunSourceRecord,
            {"run_id": payload.run_id, "source_id": payload.source_id},
        )
        if link is not None and reviewed.prompt_id in {"P05", "P07"}:
            link.processing_status = "accepted"
            link.latest_ai_artifact_id = corrected.ai_artifact_id
    elif payload.decision == "reject":
        reviewed.classification = "rejected"
        reviewed.resolution_status = "rejected"
        reviewed.resolved_by_artifact_id = review_artifact.ai_artifact_id
        review_artifact.resolution_status = "resolved"
    else:
        reviewed.resolution_status = "unresolved"
        reviewed.resolved_by_artifact_id = review_artifact.ai_artifact_id
        review_artifact.resolution_status = "unresolved"
    return issuance_ids, component_ids


def _update_run_source_after_p08(
    session: Session,
    payload: P08Output,
    artifact: AiArtifactRecord,
    classification: AiIngestStatus,
) -> None:
    link = session.get(
        RunSourceRecord,
        {"run_id": payload.run_id, "source_id": payload.source_id},
    )
    if link is not None:
        link.processing_status = (
            payload.processing_status
            if classification is AiIngestStatus.ACCEPTED
            else "needs_review"
        )
        link.latest_ai_artifact_id = artifact.ai_artifact_id


def _insert_p11(
    session: Session,
    payload: P11Output,
    artifact: AiArtifactRecord,
) -> None:
    for candidate in payload.candidates:
        session.add(
            TargetResolutionCandidateRecord(
                target_resolution_candidate_id=next_id(
                    session,
                    "TRC-",
                    width=6,
                    sequence_key="TARGET_RESOLUTION_CANDIDATE",
                ),
                proposal_artifact_id=artifact.ai_artifact_id,
                forecast_component_id=payload.forecast_component_id,
                candidate_ref=candidate.candidate_ref,
                rank=candidate.rank,
                canonical_name=candidate.canonical_name,
                target_type=candidate.target_type,
                mapping_method=candidate.mapping_method,
                instruments=[
                    instrument.model_dump(mode="json") for instrument in candidate.instruments
                ],
                existed_at=candidate.existed_at,
                knowledge_cutoff=candidate.knowledge_cutoff,
                source_evidence=candidate.source_evidence,
                confidence=candidate.confidence,
                candidate_status="proposed",
            )
        )
    component = session.get(ForecastComponentRecord, payload.forecast_component_id)
    if component is not None:
        # P11受理後は独立P12待ち。proposedのままにするとworkflowがP11を繰り返す。
        component.target_resolution_status = "awaiting_review"


def _insert_p12(
    session: Session,
    payload: P12Output,
    artifact: AiArtifactRecord,
) -> None:
    for review in payload.reviews:
        session.add(
            TargetResolutionReviewRecord(
                target_resolution_review_id=next_id(
                    session,
                    "TRV-",
                    width=6,
                    sequence_key="TARGET_RESOLUTION_REVIEW",
                ),
                review_artifact_id=artifact.ai_artifact_id,
                proposal_artifact_id=payload.proposal_artifact_id,
                forecast_component_id=payload.forecast_component_id,
                candidate_ref=review.candidate_ref,
                decision=review.decision,
                confidence=review.confidence,
                rationale=review.rationale,
                corrected_candidate=(
                    review.corrected_candidate.model_dump(mode="json")
                    if review.corrected_candidate is not None
                    else None
                ),
            )
        )
    proposal = session.get(AiArtifactRecord, payload.proposal_artifact_id)
    if proposal is not None:
        proposal.resolved_by_artifact_id = artifact.ai_artifact_id
        proposal.resolution_status = (
            "resolved"
            if payload.resolution_status in {"agreed", "unresolved"}
            else "awaiting_adjudication"
        )
    component = session.get(ForecastComponentRecord, payload.forecast_component_id)
    if component is None:
        raise ValueError("P12のcomponentが消失しました")
    if payload.resolution_status == "agreed":
        assert payload.recommended_candidate_ref is not None
        origin = payload.recommended_candidate_origin or "p11_proposal"
        candidate = session.scalar(
            select(TargetResolutionCandidateRecord).where(
                TargetResolutionCandidateRecord.proposal_artifact_id
                == payload.proposal_artifact_id,
                TargetResolutionCandidateRecord.candidate_ref == payload.recommended_candidate_ref,
            )
        )
        if candidate is None:
            raise ValueError("P12推奨candidateが消失しました")
        corrected: TargetResolutionCandidate | None = None
        if origin == "p12_correction":
            matched_review = next(
                (
                    item
                    for item in payload.reviews
                    if item.candidate_ref == payload.recommended_candidate_ref
                    and item.decision == "correct"
                    and item.corrected_candidate is not None
                ),
                None,
            )
            if matched_review is None or matched_review.corrected_candidate is None:
                raise ValueError("P12修正候補が見つかりません")
            corrected = matched_review.corrected_candidate
        _lock_candidate_mapping(
            session,
            component=component,
            candidate=candidate,
            proposal_artifact_id=payload.proposal_artifact_id,
            review_artifact_id=artifact.ai_artifact_id,
            adjudication_artifact_id=None,
            review_result=_review_summary(payload.reviews),
            candidate_origin=origin,
            corrected_candidate=corrected,
        )
        artifact.resolution_status = "resolved"
    elif payload.resolution_status == "unresolved":
        _lock_unresolvable_mapping(
            session,
            component=component,
            knowledge_cutoff=payload.knowledge_cutoff,
            reason=payload.unevaluable_reason or "対象解決不能",
            proposal_artifact_id=payload.proposal_artifact_id,
            review_artifact_id=artifact.ai_artifact_id,
            adjudication_artifact_id=None,
        )
        artifact.resolution_status = "resolved"
    else:
        component.target_resolution_status = "awaiting_adjudication"
        artifact.resolution_status = "awaiting_adjudication"


def _insert_p13(
    session: Session,
    payload: P13Output,
    artifact: AiArtifactRecord,
) -> None:
    session.add(
        TargetResolutionAdjudicationRecord(
            target_resolution_adjudication_id=next_id(
                session,
                "TRA-",
                width=6,
                sequence_key="TARGET_RESOLUTION_ADJUDICATION",
            ),
            adjudication_artifact_id=artifact.ai_artifact_id,
            proposal_artifact_id=payload.proposal_artifact_id,
            review_artifact_id=payload.review_artifact_id,
            forecast_component_id=payload.forecast_component_id,
            final_status=payload.final_status,
            selected_candidate_ref=payload.selected_candidate_ref,
            rationale=payload.rationale,
        )
    )
    component = session.get(ForecastComponentRecord, payload.forecast_component_id)
    if component is None:
        raise ValueError("P13のcomponentが消失しました")
    if payload.final_status == "verified":
        assert payload.selected_candidate_ref is not None
        origin = payload.selected_candidate_origin or "p11_proposal"
        candidate = session.scalar(
            select(TargetResolutionCandidateRecord).where(
                TargetResolutionCandidateRecord.proposal_artifact_id
                == payload.proposal_artifact_id,
                TargetResolutionCandidateRecord.candidate_ref == payload.selected_candidate_ref,
            )
        )
        if candidate is None:
            raise ValueError("P13選択candidateが消失しました")
        corrected: TargetResolutionCandidate | None = None
        if origin == "p12_correction":
            review_row = session.scalar(
                select(TargetResolutionReviewRecord).where(
                    TargetResolutionReviewRecord.review_artifact_id == payload.review_artifact_id,
                    TargetResolutionReviewRecord.candidate_ref == payload.selected_candidate_ref,
                    TargetResolutionReviewRecord.decision == "correct",
                )
            )
            if review_row is None or not review_row.corrected_candidate:
                raise ValueError("P13が参照するP12修正候補がありません")
            corrected = TargetResolutionCandidate.model_validate(review_row.corrected_candidate)
        _lock_candidate_mapping(
            session,
            component=component,
            candidate=candidate,
            proposal_artifact_id=payload.proposal_artifact_id,
            review_artifact_id=payload.review_artifact_id,
            adjudication_artifact_id=artifact.ai_artifact_id,
            review_result=payload.rationale,
            candidate_origin=origin,
            corrected_candidate=corrected,
        )
    else:
        _lock_unresolvable_mapping(
            session,
            component=component,
            knowledge_cutoff=payload.knowledge_cutoff,
            reason=payload.unevaluable_reason or "裁定でも対象解決不能",
            proposal_artifact_id=payload.proposal_artifact_id,
            review_artifact_id=payload.review_artifact_id,
            adjudication_artifact_id=artifact.ai_artifact_id,
        )
    review = session.get(AiArtifactRecord, payload.review_artifact_id)
    proposal = session.get(AiArtifactRecord, payload.proposal_artifact_id)
    if review is not None:
        review.resolved_by_artifact_id = artifact.ai_artifact_id
        review.resolution_status = "resolved"
    if proposal is not None:
        proposal.resolved_by_artifact_id = artifact.ai_artifact_id
        proposal.resolution_status = "resolved"
    artifact.resolution_status = "resolved"


def _lock_candidate_mapping(
    session: Session,
    *,
    component: ForecastComponentRecord,
    candidate: TargetResolutionCandidateRecord,
    proposal_artifact_id: str,
    review_artifact_id: str,
    adjudication_artifact_id: str | None,
    review_result: str,
    candidate_origin: str = "p11_proposal",
    corrected_candidate: TargetResolutionCandidate | None = None,
) -> None:
    if corrected_candidate is not None:
        instruments = [
            instrument.model_dump(mode="json") for instrument in corrected_candidate.instruments
        ]
        canonical_name = corrected_candidate.canonical_name
        target_type: str = corrected_candidate.target_type
        mapping_method: str = corrected_candidate.mapping_method
        knowledge_cutoff = corrected_candidate.knowledge_cutoff
        source_evidence = corrected_candidate.source_evidence
        mapping_status = "corrected"
    else:
        instruments = list(candidate.instruments)
        canonical_name = candidate.canonical_name
        target_type = candidate.target_type
        mapping_method = candidate.mapping_method
        knowledge_cutoff = candidate.knowledge_cutoff
        source_evidence = candidate.source_evidence
        mapping_status = "verified"
    primary = instruments[0]
    target = session.scalar(
        select(TargetRecord).where(
            TargetRecord.canonical_name == canonical_name,
            TargetRecord.ticker == primary["symbol"],
            TargetRecord.currency == primary["currency"],
        )
    )
    if target is None:
        target = TargetRecord(
            target_id=next_id(session, "TGT-", width=6, sequence_key="TARGET"),
            raw_label=component.raw_target_label or canonical_name,
            canonical_name=canonical_name,
            target_type=target_type,
            ticker=primary["symbol"],
            exchange=primary.get("exchange"),
            currency=primary["currency"],
        )
        session.add(target)
        session.flush()
    mapping_data = {
        "candidate_id": candidate.target_resolution_candidate_id,
        "candidate_origin": candidate_origin,
        "instruments": instruments,
        "original_instruments": candidate.instruments,
        "knowledge_cutoff": knowledge_cutoff.isoformat(),
        "proposal_artifact_id": proposal_artifact_id,
        "review_artifact_id": review_artifact_id,
        "adjudication_artifact_id": adjudication_artifact_id,
    }
    mapping_hash = hashlib.sha256(
        json.dumps(mapping_data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    mapping = TargetMappingRecord(
        target_mapping_id=next_id(session, "MAP-", width=6, sequence_key="TARGET_MAPPING"),
        target_id=target.target_id,
        mapping_method=mapping_method,
        evaluation_instruments=instruments,
        weights=[float(instrument["weight"]) for instrument in instruments],
        knowledge_cutoff=knowledge_cutoff,
        source_evidence=source_evidence,
        proposal_model=_artifact_model(session, proposal_artifact_id),
        review_result=review_result,
        proposal_artifact_id=proposal_artifact_id,
        review_artifact_id=review_artifact_id,
        adjudication_artifact_id=adjudication_artifact_id,
        mapping_status=mapping_status,
        mapping_hash=mapping_hash,
        locked_at=datetime.now(UTC),
    )
    session.add(mapping)
    session.flush()
    candidate.candidate_status = mapping_status
    component.target_id = target.target_id
    component.target_mapping_id = mapping.target_mapping_id
    component.target_resolution_status = "locked"


def _lock_unresolvable_mapping(
    session: Session,
    *,
    component: ForecastComponentRecord,
    knowledge_cutoff: datetime,
    reason: str,
    proposal_artifact_id: str,
    review_artifact_id: str,
    adjudication_artifact_id: str | None,
) -> None:
    target = TargetRecord(
        target_id=next_id(session, "TGT-", width=6, sequence_key="TARGET"),
        raw_label=component.raw_target_label or "対象解決不能",
        canonical_name=component.raw_target_label or "対象解決不能",
        target_type="theme",
        ticker=None,
        exchange=None,
        currency=None,
    )
    session.add(target)
    session.flush()
    mapping_hash = hashlib.sha256(
        json.dumps(
            {
                "component_id": component.forecast_component_id,
                "status": "unresolvable",
                "reason": reason,
                "proposal_artifact_id": proposal_artifact_id,
                "review_artifact_id": review_artifact_id,
                "adjudication_artifact_id": adjudication_artifact_id,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    mapping = TargetMappingRecord(
        target_mapping_id=next_id(session, "MAP-", width=6, sequence_key="TARGET_MAPPING"),
        target_id=target.target_id,
        mapping_method="unresolvable",
        evaluation_instruments=[],
        weights=None,
        knowledge_cutoff=knowledge_cutoff,
        source_evidence=reason,
        proposal_model=_artifact_model(session, proposal_artifact_id),
        review_result=reason,
        proposal_artifact_id=proposal_artifact_id,
        review_artifact_id=review_artifact_id,
        adjudication_artifact_id=adjudication_artifact_id,
        mapping_status="unresolvable",
        mapping_hash=mapping_hash,
        locked_at=datetime.now(UTC),
        unevaluable_reason=reason,
    )
    session.add(mapping)
    session.flush()
    component.target_id = target.target_id
    component.target_mapping_id = mapping.target_mapping_id
    component.target_resolution_status = "unresolvable"


def _artifact_model(session: Session, artifact_id: str) -> str | None:
    artifact = session.get(AiArtifactRecord, artifact_id)
    if artifact is None:
        return None
    execution = session.get(PromptExecutionRecord, artifact.prompt_execution_id)
    return execution.model if execution is not None else None


def _component_context(
    session: Session,
    component_id: str,
) -> (
    tuple[
        ForecastComponentRecord,
        ForecastIssuanceRecord,
        AiArtifactRecord | None,
    ]
    | None
):
    component = session.get(ForecastComponentRecord, component_id)
    if component is None:
        return None
    issuance = session.get(ForecastIssuanceRecord, component.forecast_issuance_id)
    if issuance is None:
        return None
    artifact = (
        session.get(AiArtifactRecord, issuance.ai_artifact_id)
        if issuance.ai_artifact_id is not None
        else None
    )
    return component, issuance, artifact


def _initial_resolution_status(
    payload: PipelineOutput,
    classification: AiIngestStatus,
) -> str:
    if classification is AiIngestStatus.NEEDS_REVIEW:
        return "needs_review"
    if isinstance(payload, (P06Output, P09Output)):
        return payload.decision if payload.decision != "accept" else "resolved"
    if isinstance(payload, P11Output):
        return "awaiting_review"
    if isinstance(payload, P12Output) and payload.resolution_status == "disagreed":
        return "awaiting_adjudication"
    return "accepted"


def _payload_confidence(payload: PipelineOutput) -> float | None:
    if isinstance(payload, P05Output):
        return min(segment.speaker_confidence for segment in payload.segments)
    if isinstance(payload, P07Output):
        return min(segment.author_confidence for segment in payload.segments)
    if isinstance(payload, P08Output) and payload.forecasts:
        return min(forecast.extraction_confidence for forecast in payload.forecasts)
    if isinstance(payload, (P06Output, P09Output)) and payload.findings:
        return None
    if isinstance(payload, P11Output) and payload.candidates:
        return max(candidate.confidence for candidate in payload.candidates)
    if isinstance(payload, P12Output) and payload.reviews:
        return min(review.confidence for review in payload.reviews)
    return None


def _payload_importance(payload: PipelineOutput) -> str:
    items: list[Any]
    if isinstance(payload, (P05Output, P07Output)):
        items = list(payload.segments)
    elif isinstance(payload, P08Output):
        items = list(payload.forecasts)
    else:
        return "normal"
    return "high" if any(item.importance == "high" for item in items) else "normal"


def _payload_importance_reason(payload: PipelineOutput) -> str | None:
    items: list[Any]
    if isinstance(payload, (P05Output, P07Output)):
        items = list(payload.segments)
    elif isinstance(payload, P08Output):
        items = list(payload.forecasts)
    else:
        return None
    reasons = [
        item.high_importance_reason
        for item in items
        if item.importance == "high" and item.high_importance_reason
    ]
    return " / ".join(reasons) if reasons else None


def _forecast_initial_status(
    evaluation_as_of: Any,
    normalized_start: Any,
) -> str:
    if normalized_start is not None and evaluation_as_of < normalized_start:
        return "not_started"
    return "active_indeterminate"


def _review_summary(reviews: list[CandidateReview]) -> str:
    return " / ".join(
        f"{review.candidate_ref}:{review.decision}:{review.rationale}" for review in reviews
    )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _reject(
    settings: AppSettings,
    *,
    input_path: Path,
    raw_bytes: bytes,
    output_hash: str,
    run_id: str,
    issues: list[ValidationIssue],
) -> AiIngestResult:
    _store_classified(
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
        guidance="次の操作: prompt別Schemaと参照関係を確認し、別名の修正版を作成してください。",
    )


def _store_classified(
    settings: AppSettings,
    *,
    run_id: str,
    input_path: Path,
    raw_bytes: bytes,
    output_hash: str,
    classification: AiIngestStatus,
    issues: list[ValidationIssue],
) -> Path | None:
    run_path = _run_path(settings, run_id)
    if run_path is None:
        return None
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", input_path.stem)[:60] or "ai-output"
    classified = (
        run_path / "03_ai_outputs" / classification.value / f"{output_hash[:12]}__{safe_name}.json"
    )
    classified.parent.mkdir(parents=True, exist_ok=True)
    if not classified.exists():
        with classified.open("xb") as output:
            output.write(raw_bytes)
    report = {
        "schema_version": "2.0.0",
        "output_hash": output_hash,
        "classification": classification.value,
        "validated_at": datetime.now(UTC).isoformat(),
        "issues": [
            {"code": issue.code, "path": issue.path, "message": issue.message} for issue in issues
        ],
    }
    _atomic_json(
        run_path / "05_audit" / "processing_logs" / f"AI-{output_hash[:12]}__validation.json",
        report,
    )
    return classified


def _run_path(settings: AppSettings, run_id: str) -> Path | None:
    if not re.fullmatch(r"RUN-\d{8}-\d{3}", run_id):
        return None
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        run = session.get(RunRecord, run_id)
        return settings.vault_root / Path(run.run_path) if run is not None else None


def _atomic_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)
