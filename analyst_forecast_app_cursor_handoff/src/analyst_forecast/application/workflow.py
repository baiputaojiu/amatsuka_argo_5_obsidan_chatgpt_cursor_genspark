from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from analyst_forecast import __version__
from analyst_forecast.application.results import generate_run_results
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.domain.models import ComponentResolutionStage
from analyst_forecast.domain.resolution import classify_component_resolution_stage
from analyst_forecast.infrastructure.db.ids import next_id
from analyst_forecast.infrastructure.db.models import (
    AiArtifactRecord,
    AiImportRecord,
    EvaluationRecord,
    ForecastComponentRecord,
    ForecastIssuanceRecord,
    RunRecord,
    RunSourceRecord,
    SourceRecord,
    TargetMappingRecord,
    TargetRecord,
    TargetResolutionCandidateRecord,
    TargetResolutionReviewRecord,
    WorkflowTaskRecord,
)
from analyst_forecast.infrastructure.db.session import create_session_factory


class WorkflowAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    title: str
    reason: str
    executor: str
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    command_or_prompt: str | None = None


class WorkflowState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.1.0"
    app_version: str = __version__
    run_id: str
    stage: str
    updated_at: datetime
    counts: dict[str, int]
    recommended_action: WorkflowAction
    alternatives: list[WorkflowAction] = Field(default_factory=list, max_length=2)
    blockers: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    component_summaries: list[dict[str, str | None]] = Field(default_factory=list)


def refresh_workflow(settings: AppSettings, run_id: str) -> WorkflowState:
    session_factory = create_session_factory(settings.database_file)
    with session_factory.begin() as session:
        run = session.get(RunRecord, run_id)
        if run is None:
            raise ValueError(f"案件IDが存在しません: {run_id}")
        run_path = settings.vault_root / Path(run.run_path)
        context = _load_run_context(session, settings, run)
        stage, action, alternatives = _choose_action(settings, run, context)
        _sync_workflow_tasks(session, run_id, stage, action, alternatives, context)
        counts = context.counts
        issues = _issues_for(settings, counts)
        blockers: list[str] = []
        run.status = stage
        component_summaries = context.component_summaries

    state = WorkflowState(
        run_id=run_id,
        stage=stage,
        updated_at=datetime.now(UTC),
        counts=counts,
        recommended_action=action,
        alternatives=alternatives,
        blockers=blockers,
        issues=issues,
        component_summaries=component_summaries,
    )
    _write_state_files(run_path, state)
    generate_run_results(settings, run_id)
    return state


def assert_component_belongs_to_run(
    settings: AppSettings,
    *,
    run_id: str,
    component_id: str,
) -> None:
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        if not _component_in_run(session, run_id=run_id, component_id=component_id):
            raise ValueError(
                f"構成予想 {component_id} は案件 {run_id} に属していません。\n"
                "次の操作: status で正しいcomponent IDを確認してください。"
            )


class _RunContext:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.source_links: list[RunSourceRecord] = []
        self.existing_inputs: list[str] = []
        self.issuances: list[ForecastIssuanceRecord] = []
        self.components: list[ForecastComponentRecord] = []
        self.evaluations: list[EvaluationRecord] = []
        self.unevaluable_open: int = 0
        self.unevaluated_components: list[ForecastComponentRecord] = []
        self.needs_review_unresolved: int = 0
        self.processed_no_forecast: int = 0
        self.pending_p05: list[str] = []
        self.pending_p07: list[str] = []
        self.pending_p08: list[str] = []
        self.pending_p06: list[str] = []
        self.pending_p09: list[str] = []
        self.pending_p11: list[str] = []
        self.pending_p12: list[str] = []
        self.pending_p13: list[str] = []
        self.unresolvable_components: list[str] = []
        self.component_summaries: list[dict[str, str | None]] = []
        self.evaluation_as_of: date | None = None
        # 次処理に必要な artifact ID（component_id → ids）
        self.resolution_refs: dict[str, dict[str, str | None]] = {}


def _preprocess_prompt_for_medium(medium: str) -> str:
    return "P05" if medium == "youtube" else "P07"


def _load_run_context(
    session: Session,
    settings: AppSettings,
    run: RunRecord,
) -> _RunContext:
    context = _RunContext()
    context.evaluation_as_of = run.evaluation_as_of
    context.source_links = list(
        session.scalars(select(RunSourceRecord).where(RunSourceRecord.run_id == run.run_id))
    )
    for link in context.source_links:
        if link.local_input_path:
            path = settings.vault_root / Path(link.local_input_path)
            if path.is_file():
                context.existing_inputs.append(str(path))
            else:
                source = session.get(SourceRecord, link.source_id)
                if source is not None:
                    fallback = settings.vault_root / Path(source.raw_file_path)
                    if fallback.is_file():
                        context.existing_inputs.append(str(fallback))
        if link.processing_status in {
            "processed_no_forecast",
            "processed_no_formal_forecast",
        }:
            context.processed_no_forecast += 1
        if (
            link.processing_status in {"raw_imported", "p05_pending", "p07_pending", "needs_review"}
            and link.local_input_path
        ):
            source = session.get(SourceRecord, link.source_id)
            medium = source.medium if source is not None else link.observed_medium
            pending_path = str(settings.vault_root / Path(link.local_input_path))
            if _preprocess_prompt_for_medium(medium) == "P05":
                context.pending_p05.append(pending_path)
            else:
                context.pending_p07.append(pending_path)

    artifacts = list(
        session.scalars(select(AiArtifactRecord).where(AiArtifactRecord.run_id == run.run_id))
    )
    unresolved_review = [
        item
        for item in artifacts
        if item.classification == "needs_review"
        and item.resolution_status not in {
            "resolved", "superseded", "accepted", "rejected", "unresolved",
        }
    ]
    for item in unresolved_review:
        if item.prompt_id in {"P05", "P07"}:
            context.pending_p06.append(item.ai_artifact_id)
        elif item.prompt_id == "P08":
            source_link = next(
                (lnk for lnk in context.source_links if lnk.source_id == item.source_id),
                None,
            )
            if source_link is not None and source_link.processing_status in {
                "p08_review_unresolved_terminal",
                "p08_rejected_terminal",
            }:
                pass
            else:
                context.pending_p09.append(item.ai_artifact_id)
    legacy_imports = list(
        session.scalars(select(AiImportRecord).where(AiImportRecord.run_id == run.run_id))
    )
    unresolved_legacy = [
        item
        for item in legacy_imports
        if item.classification == "needs_review"
        and not any(
            other.classification == "accepted"
            and other.source_id == item.source_id
            and other.ai_import_id != item.ai_import_id
            for other in legacy_imports
        )
    ]
    context.needs_review_unresolved = len(unresolved_review) + len(unresolved_legacy)

    context.issuances = list(
        session.scalars(
            select(ForecastIssuanceRecord)
            .outerjoin(
                AiImportRecord,
                AiImportRecord.ai_import_id == ForecastIssuanceRecord.ai_import_id,
            )
            .outerjoin(
                AiArtifactRecord,
                AiArtifactRecord.ai_artifact_id == ForecastIssuanceRecord.ai_artifact_id,
            )
            .where(
                (AiImportRecord.run_id == run.run_id) | (AiArtifactRecord.run_id == run.run_id),
                ForecastIssuanceRecord.lifecycle_status == "active",
                ForecastIssuanceRecord.made_at.is_not(None),
            )
        )
    )
    issuance_ids = [item.forecast_issuance_id for item in context.issuances]
    context.components = (
        list(
            session.scalars(
                select(ForecastComponentRecord).where(
                    ForecastComponentRecord.forecast_issuance_id.in_(issuance_ids)
                )
            )
        )
        if issuance_ids
        else []
    )
    component_ids = [item.forecast_component_id for item in context.components]
    context.evaluations = (
        list(
            session.scalars(
                select(EvaluationRecord).where(
                    EvaluationRecord.forecast_component_id.in_(component_ids)
                )
            )
        )
        if component_ids
        else []
    )

    for component in context.components:
        mapping = (
            session.get(TargetMappingRecord, component.target_mapping_id)
            if component.target_mapping_id
            else None
        )
        target = session.get(TargetRecord, component.target_id) if component.target_id else None
        latest = _latest_evaluation(context.evaluations, component.forecast_component_id)
        stage = classify_component_resolution_stage(component, mapping)
        cid = component.forecast_component_id

        if stage is ComponentResolutionStage.NEED_P11:
            context.pending_p11.append(cid)
        elif stage is ComponentResolutionStage.NEED_P12:
            context.pending_p12.append(cid)
            context.resolution_refs[cid] = _resolution_refs_for_component(session, cid)
        elif stage is ComponentResolutionStage.NEED_P13:
            context.pending_p13.append(cid)
            context.resolution_refs[cid] = _resolution_refs_for_component(session, cid)
        elif stage is ComponentResolutionStage.UNRESOLVABLE:
            context.unresolvable_components.append(cid)
        elif stage is ComponentResolutionStage.READY_FOR_EVALUATION:
            if latest is None:
                context.unevaluated_components.append(component)
            elif latest.evaluation_status == "unevaluable":
                context.unevaluable_open += 1

        context.component_summaries.append(
            {
                "forecast_component_id": cid,
                "forecast_issuance_id": component.forecast_issuance_id,
                "raw_target_label": component.raw_target_label,
                "symbol": target.ticker if target else None,
                "direction": component.direction,
                "mapping_status": mapping.mapping_status if mapping else None,
                "target_resolution_status": component.target_resolution_status,
                "resolution_stage": stage.value,
                "latest_evaluation_status": latest.evaluation_status if latest else None,
            }
        )

    from analyst_forecast.application.artifact_reuse import is_artifact_applicable_for_source

    accepted_preprocess_sources = {
        item.source_id
        for item in artifacts
        if item.prompt_id in {"P05", "P07"}
        and item.classification == "accepted"
        and item.source_id
        and is_artifact_applicable_for_source(
            session, artifact_id=item.ai_artifact_id, source_id=item.source_id
        )
    }
    # 再利用適用済みの上流artifactも受理扱い（P08と同じ applicability 判定）
    for link in context.source_links:
        if link.latest_ai_artifact_id:
            reused = session.get(AiArtifactRecord, link.latest_ai_artifact_id)
            if (
                reused is not None
                and reused.prompt_id in {"P05", "P07"}
                and reused.classification == "accepted"
                and is_artifact_applicable_for_source(
                    session,
                    artifact_id=reused.ai_artifact_id,
                    source_id=link.source_id,
                )
            ):
                accepted_preprocess_sources.add(link.source_id)
    for link in context.source_links:
        source = session.get(SourceRecord, link.source_id)
        medium = source.medium if source is not None else link.observed_medium
        preprocess_prompt = _preprocess_prompt_for_medium(medium)
        if (
            link.source_id not in accepted_preprocess_sources
            and link.processing_status
            not in {
                "processed_no_forecast",
                "processed_no_formal_forecast",
                "accepted",
            }
            and link.local_input_path
        ):
            pending_path = str(settings.vault_root / Path(link.local_input_path))
            if preprocess_prompt == "P05":
                if pending_path not in context.pending_p05:
                    context.pending_p05.append(pending_path)
            elif pending_path not in context.pending_p07:
                context.pending_p07.append(pending_path)
        terminal_p08_statuses = {
            "accepted",
            "processed_with_forecasts",
            "processed_no_forecast",
            "processed_no_formal_forecast",
            "p08_rejected_terminal",
            "p08_review_unresolved_terminal",
        }
        has_p08 = any(
            item.prompt_id == "P08"
            and item.source_id == link.source_id
            and (
                item.classification in terminal_p08_statuses
                or link.processing_status in terminal_p08_statuses
            )
            for item in artifacts
        ) or link.processing_status in {
            "processed_no_forecast",
            "processed_no_formal_forecast",
            "processed_with_forecasts",
        }
        pending_p08 = (
            link.source_id in accepted_preprocess_sources
            and link.local_input_path
            and (
                (
                    not has_p08
                    and link.processing_status
                    not in {"processed_no_forecast", "processed_no_formal_forecast"}
                )
                or link.processing_status == "p08_reextract_required"
            )
        )
        if pending_p08 and link.local_input_path is not None:
            context.pending_p08.append(str(settings.vault_root / Path(link.local_input_path)))

    rejected = sum(1 for item in artifacts if item.classification == "rejected")
    context.counts = {
        "sources": len(context.source_links),
        "existing_inputs": len(context.existing_inputs),
        "forecast_issuances": len(context.issuances),
        "forecast_components": len(context.components),
        "evaluations": len(context.evaluations),
        "unevaluated_components": len(context.unevaluated_components),
        "needs_review": context.needs_review_unresolved,
        "rejected": rejected,
        "unevaluable": context.unevaluable_open,
        "processed_no_forecast": context.processed_no_forecast,
        "pending_p05": len(context.pending_p05),
        "pending_p07": len(context.pending_p07),
    }
    return context


def _latest_evaluation(
    evaluations: list[EvaluationRecord],
    component_id: str,
) -> EvaluationRecord | None:
    matches = [item for item in evaluations if item.forecast_component_id == component_id]
    if not matches:
        return None
    return sorted(
        matches,
        key=lambda item: (item.evaluation_as_of, item.created_at),
        reverse=True,
    )[0]


def _resolution_refs_for_component(
    session: Session,
    component_id: str,
) -> dict[str, str | None]:
    candidate = session.scalar(
        select(TargetResolutionCandidateRecord)
        .where(TargetResolutionCandidateRecord.forecast_component_id == component_id)
        .order_by(TargetResolutionCandidateRecord.created_at.desc())
    )
    review = session.scalar(
        select(TargetResolutionReviewRecord)
        .where(TargetResolutionReviewRecord.forecast_component_id == component_id)
        .order_by(TargetResolutionReviewRecord.created_at.desc())
    )
    return {
        "proposal_artifact_id": candidate.proposal_artifact_id if candidate is not None else None,
        "review_artifact_id": review.review_artifact_id if review is not None else None,
    }


def _choose_action(
    settings: AppSettings,
    run: RunRecord,
    context: _RunContext,
) -> tuple[str, WorkflowAction, list[WorkflowAction]]:
    run_path = settings.vault_root / Path(run.run_path)
    if context.counts["sources"] == 0 or context.counts["existing_inputs"] == 0:
        return (
            "awaiting_raw_source",
            WorkflowAction(
                action_id="IMPORT_RAW",
                title="raw原文を取り込む",
                reason="評価へ進むための変更禁止原文がまだ登録されていません。",
                executor="[USER]",
                inputs=["UTF-8原文ファイル", "媒体・URL・発言日時・公開日時"],
                outputs=["02_sources/<medium>/raw/", "SOURCE登録", "SHA-256"],
                command_or_prompt=f"analyst-forecast source import {run.run_id} <原文ファイル>",
            ),
            [
                WorkflowAction(
                    action_id="REVIEW_REQUEST",
                    title="案件条件を確認する",
                    reason="対象期間や媒体を先に確認したい場合の代替です。",
                    executor="[USER]",
                    inputs=[str(run_path / "request.yaml")],
                    outputs=["確認済みの案件条件"],
                )
            ],
        )

    if context.needs_review_unresolved > 0:
        review_prompt = "P06" if context.pending_p06 else "P09"
        return (
            "awaiting_ai_review",
            WorkflowAction(
                action_id="REVIEW_AI_OUTPUT",
                title="低確信度のAI出力をレビューする",
                reason=(
                    f"未解決のneeds_reviewが{context.needs_review_unresolved}件あります。"
                    f"{review_prompt}の別AIレビューを取り込んでください。"
                ),
                executor="[AI Cursor]",
                inputs=["03_ai_outputs/needs_review/", *context.existing_inputs[:3]],
                outputs=["修正版JSONまたは問題点"],
                command_or_prompt=(
                    f"{run_path / '01_prompts'} の{review_prompt}を実行し、"
                    "analyst-forecast ai ingest <出力JSON> で取り込んでください。"
                ),
            ),
            [],
        )

    if context.pending_p05 or context.pending_p07:
        if context.pending_p05 and not context.pending_p07:
            prompt_id = "P05"
            inputs = context.pending_p05[:5]
        elif context.pending_p07 and not context.pending_p05:
            prompt_id = "P07"
            inputs = context.pending_p07[:5]
        else:
            prompt_id = "P05/P07"
            inputs = (context.pending_p05 + context.pending_p07)[:5]
        return (
            "awaiting_source_preprocess",
            WorkflowAction(
                action_id="RUN_PREPROCESS",
                title="媒体別の原文前処理を実行する",
                reason=f"{prompt_id}の前処理が未完了のSOURCEがあります。",
                executor="[AI Cursor]",
                inputs=inputs or context.existing_inputs[:5],
                outputs=["02_sources/*/processed/"],
                command_or_prompt=(
                    f"{run_path / '01_prompts'} の{prompt_id}を実行し、"
                    "analyst-forecast ai ingest <出力JSON> で取り込んでください。"
                ),
            ),
            [],
        )

    # source単位: 他sourceにissuanceがあっても、pending P08がある限りEXTRACTを優先
    if context.pending_p08:
        pending_ids = sorted(
            {
                link.source_id
                for link in context.source_links
                if link.local_input_path
                and any(
                    str(settings.vault_root / Path(link.local_input_path)) == path
                    for path in context.pending_p08
                )
            }
        )
        return (
            "awaiting_forecast_extraction",
            WorkflowAction(
                action_id="EXTRACT_FORECASTS",
                title="AIで予想抽出を実行する",
                reason=(
                    f"前処理受理済みだがP08未完了のSOURCEが{len(context.pending_p08)}件あります。"
                    + (f" pending_source_ids={pending_ids}" if pending_ids else "")
                ),
                executor="[AI Cursor]",
                inputs=context.pending_p08[:5] or context.existing_inputs[:5],
                outputs=["03_ai_outputs/inbox/forecast_extraction.json"],
                command_or_prompt=(
                    f"{run_path / '01_prompts'} のP08を実行し、"
                    "analyst-forecast ai ingest <出力JSON> で取り込んでください。"
                ),
            ),
            [],
        )

    if (
        context.counts["forecast_issuances"] == 0
        and context.processed_no_forecast > 0
        and not context.pending_p08
    ):
        return (
            "processed_no_forecast",
            WorkflowAction(
                action_id="REVIEW_NO_FORECAST",
                title="予想なし処理結果を確認する",
                reason="処理済みですが予想0件のため、同じP08を繰り返し要求しません。",
                executor="[USER]",
                inputs=context.existing_inputs[:5],
                outputs=["調査網羅性の確認"],
                command_or_prompt="追加原文がある場合のみ source import してください。",
            ),
            [
                WorkflowAction(
                    action_id="ADD_ANOTHER_SOURCE",
                    title="同じ案件へ原文を追加する",
                    reason="別情報源を追加する場合の代替です。",
                    executor="[USER]",
                    inputs=["追加raw原文"],
                    outputs=["追加SOURCEと差分処理"],
                )
            ],
        )

    source_terminal_statuses = {
        "p08_rejected_terminal",
        "p08_review_unresolved_terminal",
        "processed_no_forecast",
        "processed_no_formal_forecast",
    }
    if (
        context.counts["forecast_issuances"] == 0
        and context.source_links
        and all(link.processing_status in source_terminal_statuses for link in context.source_links)
        and not context.pending_p08
        and not context.pending_p05
        and not context.pending_p07
    ):
        return (
            "complete_no_active_forecast",
            WorkflowAction(
                action_id="COMPLETE_NO_ACTIVE_FORECAST",
                title="アクティブ予想なしで完了する",
                reason=(
                    "全SOURCEが終端状態であり、active予想表明が0件です。"
                    "これ以上の自動P08/P09反復は行いません。"
                ),
                executor="[USER]",
                inputs=context.existing_inputs[:5],
                outputs=["完了確認", "追加原文の検討"],
                command_or_prompt="追加原文がある場合のみ source import してください。",
            ),
            [
                WorkflowAction(
                    action_id="ADD_ANOTHER_SOURCE",
                    title="同じ案件へ原文を追加する",
                    reason="別情報源を追加する場合の代替です。",
                    executor="[USER]",
                    inputs=["追加raw原文"],
                    outputs=["追加SOURCEと差分処理"],
                )
            ],
        )

    if context.counts["forecast_issuances"] == 0:
        return (
            "awaiting_forecast_extraction",
            WorkflowAction(
                action_id="EXTRACT_FORECASTS",
                title="AIで予想抽出を実行する",
                reason="raw原文は登録済みですが、検証済みの予想表明がありません。",
                executor="[AI Cursor]",
                inputs=(
                    context.existing_inputs[:5]
                    or [str(path) for path in sorted((run_path / "01_prompts").glob("P0*"))]
                ),
                outputs=["03_ai_outputs/inbox/forecast_extraction.json"],
                command_or_prompt=(
                    f"{run_path / '01_prompts'} のP05（必要時）とP08を順に実行してください。"
                ),
            ),
            [
                WorkflowAction(
                    action_id="USE_CHATGPT_PROMPT",
                    title="ChatGPT版プロンプトを使う",
                    reason="Cursorで処理しない場合の同一Schema代替です。",
                    executor="[AI ChatGPT]",
                    inputs=context.existing_inputs[:3],
                    outputs=["forecast_extraction.json"],
                )
            ],
        )

    if context.pending_p11:
        component_id = context.pending_p11[0]
        return (
            "awaiting_target_resolution",
            WorkflowAction(
                action_id="RUN_P11",
                title="P11対象解決提案を実行する",
                reason=f"構成予想 {component_id} の対象解決が未完了です。",
                executor="[AI Cursor]",
                inputs=[
                    f"run_id={run.run_id}",
                    f"component_id={component_id}",
                    *context.existing_inputs[:2],
                ],
                outputs=["03_ai_outputs/inbox/p11_*.json"],
                command_or_prompt=(
                    f"P11を component {component_id} に対して実行し、"
                    "analyst-forecast ai ingest で取り込んでください。"
                ),
            ),
            [],
        )

    if context.pending_p12:
        component_id = context.pending_p12[0]
        refs = context.resolution_refs.get(component_id, {})
        proposal_id = refs.get("proposal_artifact_id") or "（P11成果物IDをstatusで確認）"
        return (
            "awaiting_target_review",
            WorkflowAction(
                action_id="RUN_P12",
                title="P12対象レビューを実行する",
                reason=f"構成予想 {component_id} の独立レビューが未完了です。",
                executor="[AI Cursor]",
                inputs=[
                    f"run_id={run.run_id}",
                    f"component_id={component_id}",
                    f"proposal_artifact_id={proposal_id}",
                ],
                outputs=["03_ai_outputs/inbox/p12_*.json"],
                command_or_prompt=(
                    f"P11とは別実行でP12を component {component_id} "
                    f"(proposal={proposal_id}) に行い、ai ingestしてください。"
                ),
            ),
            [],
        )

    if context.pending_p13:
        component_id = context.pending_p13[0]
        refs = context.resolution_refs.get(component_id, {})
        proposal_id = refs.get("proposal_artifact_id") or "（P11成果物IDをstatusで確認）"
        review_id = refs.get("review_artifact_id") or "（P12成果物IDをstatusで確認）"
        return (
            "awaiting_target_adjudication",
            WorkflowAction(
                action_id="RUN_P13",
                title="P13対象解決の裁定を実行する",
                reason=(
                    f"構成予想 {component_id} でP11とP12が不一致のため、"
                    "市場評価の前にP13裁定が必要です。"
                ),
                executor="[AI Cursor]",
                inputs=[
                    f"run_id={run.run_id}",
                    f"component_id={component_id}",
                    f"proposal_artifact_id={proposal_id}",
                    f"review_artifact_id={review_id}",
                ],
                outputs=["03_ai_outputs/inbox/p13_*.json"],
                command_or_prompt=(
                    f"P13を component {component_id} "
                    f"(proposal={proposal_id}, review={review_id}) に実行し、"
                    "ai ingestしてください。市場評価はまだ行わないでください。"
                ),
            ),
            [],
        )

    if context.unevaluated_components:
        component = context.unevaluated_components[0]
        as_of = run.evaluation_as_of.isoformat()
        command = (
            f"analyst-forecast market evaluate {run.run_id} "
            f"{component.forecast_component_id} --as-of {as_of}"
        )
        return (
            "ready_for_market_evaluation",
            WorkflowAction(
                action_id="EVALUATE_MARKET",
                title="市場データで最小方向評価を実行する",
                reason=(f"未評価の構成予想が{len(context.unevaluated_components)}件あります。"),
                executor="[PYTHON]",
                inputs=[
                    component.forecast_component_id,
                    "固定済みtarget mapping",
                    "市場provider",
                ],
                outputs=["evaluation", "04_results/", "市場データcache"],
                command_or_prompt=command,
            ),
            [
                WorkflowAction(
                    action_id="IMPORT_MARKET_CSV",
                    title="市場データCSVを用意する",
                    reason="自動providerで取得できない場合の再現可能な代替です。",
                    executor="[USER]",
                    inputs=["日付・OHLC・調整済み始値/終値を含むCSV"],
                    outputs=["検証可能なCSV入力"],
                    command_or_prompt=(
                        f"analyst-forecast market evaluate {run.run_id} "
                        f"{component.forecast_component_id} --as-of {as_of} "
                        "--provider csv --csv-path <CSV>"
                    ),
                )
            ],
        )

    if context.unresolvable_components and not context.unevaluated_components:
        return (
            "target_unresolvable",
            WorkflowAction(
                action_id="REVIEW_UNRESOLVABLE",
                title="対象解決不能の結果を確認する",
                reason=(
                    "対象がunresolvableとして確定したため、推測値での市場取得は行いません。"
                    f"対象component: {', '.join(context.unresolvable_components[:5])}"
                ),
                executor="[USER]",
                inputs=context.unresolvable_components[:5],
                outputs=["評価不能としての確認", "04_results/"],
                command_or_prompt="同じP11/P12を無限反復せず、結果確認または別原文追加へ進んでください。",
            ),
            [],
        )

    if context.unevaluable_open > 0:
        return (
            "market_data_unavailable",
            WorkflowAction(
                action_id="SUPPLY_MARKET_CSV",
                title="取得不能対象の市場CSVを確認する",
                reason="推測値を使わず評価不能として保存された項目があります。",
                executor="[USER]",
                inputs=["OPEN_ISSUES.md", "信頼できる市場CSV"],
                outputs=["再評価可能なCSVまたは評価不能の確認"],
            ),
            [],
        )

    return (
        "vertical_mvp_review",
        WorkflowAction(
            action_id="REVIEW_RESULTS",
            title="縦断MVPの結果を確認する",
            reason="原文、AI検証、DB取込み、最小方向評価が完了しました。",
            executor="[USER]",
            inputs=[
                str(run_path / "04_results" / "forecasts" / "all_forecasts.md"),
                str(run_path / "04_results" / "evaluations" / "evaluations.md"),
                str(run_path / "04_results" / "reports" / "vertical_mvp_summary.md"),
            ],
            outputs=["確認結果と次段階へ進む判断"],
        ),
        [
            WorkflowAction(
                action_id="ADD_ANOTHER_SOURCE",
                title="同じ案件へ原文を追加する",
                reason="5～10件のfixtureへ拡張する場合の代替です。",
                executor="[USER]",
                inputs=["追加raw原文"],
                outputs=["追加SOURCEと差分処理"],
            )
        ],
    )


def _sync_workflow_tasks(
    session: Session,
    run_id: str,
    stage: str,
    action: WorkflowAction,
    alternatives: list[WorkflowAction],
    context: _RunContext,
) -> None:
    existing = {
        item.task_key: item
        for item in session.scalars(
            select(WorkflowTaskRecord).where(WorkflowTaskRecord.run_id == run_id)
        )
    }
    desired: list[tuple[str, WorkflowAction, str, int | None]] = [
        (action.action_id, action, "pending", 1)
    ]
    for index, alternative in enumerate(alternatives, start=2):
        desired.append((alternative.action_id, alternative, "pending", index))

    active_keys = {key for key, _, _, _ in desired}
    for key, record in existing.items():
        if key not in active_keys and record.status in {"pending", "running"}:
            review_done = key.startswith("REVIEW_AI") and context.needs_review_unresolved == 0
            eval_done = key.startswith("EVALUATE") and not context.unevaluated_components
            if review_done or eval_done:
                record.status = "resolved"
            else:
                record.status = "superseded"

    for key, item, status, rank in desired:
        current = existing.get(key)
        if current is None:
            session.add(
                WorkflowTaskRecord(
                    workflow_task_id=next_id(
                        session, "WFT-", width=6, sequence_key="WORKFLOW_TASK"
                    ),
                    run_id=run_id,
                    task_key=key,
                    title=item.title,
                    status=status,
                    executor=item.executor,
                    depends_on=[],
                    retryable="yes",
                    recommended_rank=rank,
                    command_or_prompt=item.command_or_prompt,
                    inputs=item.inputs,
                    outputs=item.outputs,
                    details={"stage": stage},
                )
            )
        else:
            current.title = item.title
            current.status = status
            current.executor = item.executor
            current.recommended_rank = rank
            current.command_or_prompt = item.command_or_prompt
            current.inputs = item.inputs
            current.outputs = item.outputs
            current.details = {"stage": stage}


def _component_in_run(session: Session, *, run_id: str, component_id: str) -> bool:
    component = session.get(ForecastComponentRecord, component_id)
    if component is None:
        return False
    issuance = session.get(ForecastIssuanceRecord, component.forecast_issuance_id)
    if issuance is None:
        return False
    if issuance.ai_artifact_id:
        artifact = session.get(AiArtifactRecord, issuance.ai_artifact_id)
        if artifact is not None and artifact.run_id == run_id:
            return True
    if issuance.ai_import_id:
        ai_import = session.get(AiImportRecord, issuance.ai_import_id)
        if ai_import is not None and ai_import.run_id == run_id:
            return True
    return False


def _issues_for(settings: AppSettings, counts: dict[str, int]) -> list[str]:
    issues: list[str] = []
    if settings.cursor_model is None or settings.chatgpt_model is None:
        issues.append(
            "AIモデル名が未設定です。意味判断を実行する前に高性能モデルを設定してください。"
        )
    if counts.get("rejected"):
        issues.append(f"拒否されたAI出力が{counts['rejected']}件あります。")
    if counts.get("unevaluable"):
        issues.append(f"市場データ取得不能による評価不能が{counts['unevaluable']}件あります。")
    return issues


def _write_state_files(run_path: Path, state: WorkflowState) -> None:
    data = state.model_dump(mode="json")
    _atomic_write(
        run_path / "status.yaml",
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
    )
    _atomic_write(
        run_path / "WORKFLOW_STATE.json",
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
    )

    action = state.recommended_action
    alternatives = (
        "\n".join(
            f"- `{item.action_id}` {item.title} — {item.executor}: {item.reason}"
            for item in state.alternatives
        )
        or "- なし"
    )
    component_lines = (
        "\n".join(
            f"- `{item.get('forecast_component_id')}` "
            f"issuance=`{item.get('forecast_issuance_id')}` "
            f"target={item.get('raw_target_label') or '-'} "
            f"symbol={item.get('symbol') or '-'} "
            f"direction={item.get('direction') or '-'} "
            f"mapping={item.get('mapping_status') or '-'} "
            f"eval={item.get('latest_evaluation_status') or '未評価'}"
            for item in state.component_summaries
        )
        or "- なし"
    )
    _atomic_write(
        run_path / "NEXT_ACTIONS.md",
        "# 次の行動\n\n"
        f"## 推奨：`{action.action_id}` {action.title}\n\n"
        f"- 理由：{action.reason}\n"
        f"- 担当：{action.executor}\n"
        f"- 入力：{', '.join(action.inputs) or 'なし'}\n"
        f"- 出力：{', '.join(action.outputs) or 'なし'}\n"
        f"- 操作：{action.command_or_prompt or '上記成果物を確認する'}\n\n"
        f"## 構成予想一覧\n\n{component_lines}\n\n"
        f"## 代替案\n\n{alternatives}\n",
    )
    blocker_lines = "\n".join(f"- {item}" for item in state.blockers) or "- なし"
    issue_lines = "\n".join(f"- {item}" for item in state.issues) or "- なし"
    _atomic_write(
        run_path / "OPEN_ISSUES.md",
        f"# 未解決事項\n\n## ブロッカー\n\n{blocker_lines}\n\n## その他\n\n{issue_lines}\n",
    )


def _atomic_write(path: Path, content: str) -> None:
    from analyst_forecast.application.io_utils import atomic_write_text

    atomic_write_text(path, content)
