from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from analyst_forecast import __version__
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.infrastructure.db.models import (
    AiImportRecord,
    EvaluationRecord,
    ForecastComponentRecord,
    ForecastIssuanceRecord,
    RunRecord,
    RunSourceRecord,
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

    schema_version: str = "1.0.0"
    app_version: str = __version__
    run_id: str
    stage: str
    updated_at: datetime
    counts: dict[str, int]
    recommended_action: WorkflowAction
    alternatives: list[WorkflowAction] = Field(default_factory=list, max_length=2)
    blockers: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


def refresh_workflow(settings: AppSettings, run_id: str) -> WorkflowState:
    session_factory = create_session_factory(settings.database_file)
    with session_factory.begin() as session:
        run = session.get(RunRecord, run_id)
        if run is None:
            raise ValueError(f"案件IDが存在しません: {run_id}")
        run_path = settings.vault_root / Path(run.run_path)

        source_links = list(
            session.scalars(select(RunSourceRecord).where(RunSourceRecord.run_id == run_id))
        )
        issuances = list(
            session.scalars(
                select(ForecastIssuanceRecord)
                .join(
                    AiImportRecord,
                    AiImportRecord.ai_import_id == ForecastIssuanceRecord.ai_import_id,
                )
                .where(AiImportRecord.run_id == run_id)
            )
        )
        issuance_ids = [item.forecast_issuance_id for item in issuances]
        components = (
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
        component_ids = [item.forecast_component_id for item in components]
        evaluations = (
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

        needs_review = _json_file_count(run_path / "03_ai_outputs" / "needs_review")
        rejected = _json_file_count(run_path / "03_ai_outputs" / "rejected")
        unevaluable = sum(item.evaluation_status == "unevaluable" for item in evaluations)
        counts = {
            "sources": len(source_links),
            "forecast_issuances": len(issuances),
            "forecast_components": len(components),
            "evaluations": len(evaluations),
            "needs_review": needs_review,
            "rejected": rejected,
            "unevaluable": unevaluable,
        }
        stage, action, alternatives = _choose_action(run_id, run_path, counts)
        issues = _issues_for(settings, counts)
        blockers: list[str] = []
        run.status = stage

    state = WorkflowState(
        run_id=run_id,
        stage=stage,
        updated_at=datetime.now(UTC),
        counts=counts,
        recommended_action=action,
        alternatives=alternatives,
        blockers=blockers,
        issues=issues,
    )
    _write_state_files(run_path, state)
    return state


def _choose_action(
    run_id: str,
    run_path: Path,
    counts: dict[str, int],
) -> tuple[str, WorkflowAction, list[WorkflowAction]]:
    if counts["sources"] == 0:
        return (
            "awaiting_raw_source",
            WorkflowAction(
                action_id="IMPORT_RAW",
                title="raw原文を取り込む",
                reason="評価へ進むための変更禁止原文がまだ登録されていません。",
                executor="[USER]",
                inputs=["UTF-8原文ファイル", "媒体・URL・発言日時・公開日時"],
                outputs=["02_sources/<medium>/raw/", "SOURCE登録", "SHA-256"],
                command_or_prompt=f"analyst-forecast source import {run_id} <原文ファイル>",
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

    if counts["needs_review"] > 0:
        return (
            "awaiting_ai_review",
            WorkflowAction(
                action_id="REVIEW_AI_OUTPUT",
                title="低確信度のAI出力をレビューする",
                reason="確信度が設定閾値未満のため正式テーブルへ取り込んでいません。",
                executor="[AI Cursor]",
                inputs=["03_ai_outputs/needs_review/", "raw原文", "P09相当の独立レビュー"],
                outputs=["修正版JSONまたは問題点"],
                command_or_prompt="低確信度箇所だけを原文引用に基づいて再検証してください。",
            ),
            [],
        )

    if counts["forecast_issuances"] == 0:
        return (
            "awaiting_forecast_extraction",
            WorkflowAction(
                action_id="EXTRACT_FORECASTS",
                title="AIで予想抽出を実行する",
                reason="raw原文は登録済みですが、検証済みの予想表明がありません。",
                executor="[AI Cursor]",
                inputs=["02_sources/*/raw/", "01_prompts/P05*", "01_prompts/P08*"],
                outputs=["03_ai_outputs/inbox/forecast_extraction.json"],
                command_or_prompt="01_prompts内のP05（必要時）とP08を順に実行してください。",
            ),
            [
                WorkflowAction(
                    action_id="USE_CHATGPT_PROMPT",
                    title="ChatGPT版プロンプトを使う",
                    reason="Cursorで処理しない場合の同一Schema代替です。",
                    executor="[AI ChatGPT]",
                    inputs=["raw原文", "01_prompts/*chatgpt.md"],
                    outputs=["forecast_extraction.json"],
                )
            ],
        )

    if counts["evaluations"] < counts["forecast_components"]:
        return (
            "ready_for_market_evaluation",
            WorkflowAction(
                action_id="EVALUATE_MARKET",
                title="市場データで最小方向評価を実行する",
                reason="検証済み構成予想に対する市場評価が未完了です。",
                executor="[PYTHON]",
                inputs=["forecast_component_id", "固定済みtarget mapping", "市場provider"],
                outputs=["evaluation", "evaluation_snapshot", "市場データcache"],
                command_or_prompt=f"analyst-forecast market evaluate {run_id} <component-id>",
            ),
            [
                WorkflowAction(
                    action_id="IMPORT_MARKET_CSV",
                    title="市場データCSVを用意する",
                    reason="自動providerで取得できない場合の再現可能な代替です。",
                    executor="[USER]",
                    inputs=["日付・OHLC・調整済み始値/終値を含むCSV"],
                    outputs=["検証可能なCSV入力"],
                )
            ],
        )

    if counts["unevaluable"] > 0:
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
            inputs=["04_results/", "SQLite評価", "05_audit/"],
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


def _issues_for(settings: AppSettings, counts: dict[str, int]) -> list[str]:
    issues: list[str] = []
    if settings.cursor_model is None or settings.chatgpt_model is None:
        issues.append(
            "AIモデル名が未設定です。意味判断を実行する前に高性能モデルを設定してください。"
        )
    if counts["rejected"]:
        issues.append(f"拒否されたAI出力が{counts['rejected']}件あります。")
    if counts["unevaluable"]:
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
    _atomic_write(
        run_path / "NEXT_ACTIONS.md",
        "# 次の行動\n\n"
        f"## 推奨：`{action.action_id}` {action.title}\n\n"
        f"- 理由：{action.reason}\n"
        f"- 担当：{action.executor}\n"
        f"- 入力：{', '.join(action.inputs) or 'なし'}\n"
        f"- 出力：{', '.join(action.outputs) or 'なし'}\n"
        f"- 操作：{action.command_or_prompt or '上記成果物を確認する'}\n\n"
        f"## 代替案\n\n{alternatives}\n",
    )
    blocker_lines = "\n".join(f"- {item}" for item in state.blockers) or "- なし"
    issue_lines = "\n".join(f"- {item}" for item in state.issues) or "- なし"
    _atomic_write(
        run_path / "OPEN_ISSUES.md",
        f"# 未解決事項\n\n## ブロッカー\n\n{blocker_lines}\n\n## その他\n\n{issue_lines}\n",
    )


def _json_file_count(path: Path) -> int:
    return sum(1 for item in path.glob("*.json") if item.is_file())


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)
